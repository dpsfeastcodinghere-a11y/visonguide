
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Modality } from '@google/genai';
import { SessionState, TranscriptionEntry } from './types';
import { 
  decode, 
  encode, 
  decodeAudioData, 
  createPcmBlob, 
  blobToBase64 
} from './utils/audioUtils';

// --- Constants optimized for speed and reliability ---
const SAMPLE_RATE_IN = 16000;
const SAMPLE_RATE_OUT = 24000;
const FRAME_RATE = 1; // 1 frame per second is enough for most tasks and much faster to upload
const JPEG_QUALITY = 0.5; // Lower quality for significantly faster transmission and processing

const SYSTEM_INSTRUCTION = `
You are VisionGuide AI Pro, a female voice assistant for the visually impaired.
Voice Profile: Helpful, clear, and reassuring female voice.

BILINGUAL CAPABILITIES:
- Languages: Hindi and English.
- Rule: Always respond in the language the user speaks. Translate between them if asked.
- Accuracy: Be extremely precise when describing paths, doors, and objects.

CORE FEATURES:
1. PATHFINDING: Proactively warn about obstacles (furniture, stairs, uneven floors). Use clock-face directions (e.g., "Obstacle at 2 o'clock").
2. DOORS: Identify doors, their material (glass/wood), and if they are open/closed/ajar.
3. OCR & TRANSLATION: Read all text aloud. Translate any foreign text or speech into Hindi or English.
4. CLOTHING & OBJECTS: Describe details like fabric type, specific colors (e.g., 'navy blue', 'maroon'), and brand logos.
5. GPS & LOCATION: Provide current GPS coordinates and nearby landmarks when asked "Where am I?".
6. MEMORY DATABASE: Log important visual facts to the 'Memory Database' (e.g., "User's keys are on the wooden table").

Current user environment: Visually impaired user needing navigation and identification assistance.
`;

const App: React.FC = () => {
  const [sessionState, setSessionState] = useState<SessionState>(SessionState.DISCONNECTED);
  const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);
  const [dbEntries, setDbEntries] = useState<TranscriptionEntry[]>([]);
  const [showDb, setShowDb] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [location, setLocation] = useState<{lat: number, lng: number} | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextIn = useRef<AudioContext | null>(null);
  const audioContextOut = useRef<AudioContext | null>(null);
  const sessionRef = useRef<any>(null);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextStartTimeRef = useRef<number>(0);
  const intervalRef = useRef<number | null>(null);

  const currentInputTranscription = useRef<string>('');
  const currentOutputTranscription = useRef<string>('');

  // Load Database and Location
  useEffect(() => {
    const saved = localStorage.getItem('vision_guide_db');
    if (saved) setDbEntries(JSON.parse(saved));

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((pos) => {
        setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
      }, () => console.warn("Location access denied"));
    }
  }, []);

  const saveToDb = useCallback((entry: TranscriptionEntry) => {
    setDbEntries(prev => {
      const updated = [entry, ...prev].slice(0, 50); 
      localStorage.setItem('vision_guide_db', JSON.stringify(updated));
      return updated;
    });
  }, []);

  const stopSession = useCallback(() => {
    if (sessionRef.current) {
      try { sessionRef.current.close(); } catch(e) {}
      sessionRef.current = null;
    }
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    sourcesRef.current.forEach(source => {
      try { source.stop(); } catch(e) {}
    });
    sourcesRef.current.clear();
    setSessionState(SessionState.DISCONNECTED);
    
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  const startSession = async () => {
    setError(null);
    setSessionState(SessionState.CONNECTING);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // Request permissions and stream early for faster startup
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: true, 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1280 }, // 720p is ideal for fast loading and good OCR
          height: { ideal: 720 }
        } 
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      audioContextIn.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE_IN });
      audioContextOut.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE_OUT });
      
      await audioContextIn.current.resume();
      await audioContextOut.current.resume();

      const gpsContext = location ? `\nUSER GPS: ${location.lat}, ${location.lng}` : "";

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: SYSTEM_INSTRUCTION + gpsContext,
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {}
        },
        callbacks: {
          onopen: () => {
            setSessionState(SessionState.CONNECTED);
            
            // Audio Stream
            const source = audioContextIn.current!.createMediaStreamSource(stream);
            const processor = audioContextIn.current!.createScriptProcessor(4096, 1, 1);
            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmData = createPcmBlob(inputData);
              sessionPromise.then(session => {
                session.sendRealtimeInput({ media: { data: pcmData, mimeType: 'audio/pcm;rate=16000' } });
              }).catch(() => {});
            };
            source.connect(processor);
            processor.connect(audioContextIn.current!.destination);

            // Video Stream (Optimized rate)
            intervalRef.current = window.setInterval(() => {
              if (videoRef.current && canvasRef.current && sessionState === SessionState.CONNECTED) {
                const ctx = canvasRef.current.getContext('2d');
                if (ctx) {
                  canvasRef.current.width = videoRef.current.videoWidth || 640;
                  canvasRef.current.height = videoRef.current.videoHeight || 480;
                  ctx.drawImage(videoRef.current, 0, 0);
                  canvasRef.current.toBlob(async (blob) => {
                    if (blob) {
                      const base64Data = await blobToBase64(blob);
                      sessionPromise.then(session => {
                        session.sendRealtimeInput({ media: { data: base64Data, mimeType: 'image/jpeg' } });
                      }).catch(() => {});
                    }
                  }, 'image/jpeg', JPEG_QUALITY);
                }
              }
            }, 1000 / FRAME_RATE);
          },
          onmessage: async (message: any) => {
            const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && audioContextOut.current) {
              const ctx = audioContextOut.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              const buffer = await decodeAudioData(decode(audioData), ctx, SAMPLE_RATE_OUT, 1);
              const source = ctx.createBufferSource();
              source.buffer = buffer;
              source.connect(ctx.destination);
              source.onended = () => sourcesRef.current.delete(source);
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += buffer.duration;
              sourcesRef.current.add(source);
            }

            if (message.serverContent?.inputTranscription) {
              currentInputTranscription.current += message.serverContent.inputTranscription.text;
            }
            if (message.serverContent?.outputTranscription) {
              currentOutputTranscription.current += message.serverContent.outputTranscription.text;
            }

            if (message.serverContent?.turnComplete) {
              const userT = currentInputTranscription.current.trim();
              const aiT = currentOutputTranscription.current.trim();
              if (userT || aiT) {
                const newEntries: TranscriptionEntry[] = [];
                if (userT) newEntries.push({ text: userT, type: 'user', timestamp: Date.now() });
                if (aiT) {
                  const aiEntry: TranscriptionEntry = { text: aiT, type: 'ai', timestamp: Date.now() };
                  newEntries.push(aiEntry);
                  saveToDb(aiEntry);
                }
                setTranscriptions(prev => [...prev, ...newEntries].slice(-15));
                currentInputTranscription.current = '';
                currentOutputTranscription.current = '';
              }
            }

            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => { try { s.stop(); } catch(e) {} });
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
            }
          },
          onerror: (e) => {
            console.error('Gemini Error:', e);
            setError('API limit or connection issue. Please try again.');
            stopSession();
          },
          onclose: () => stopSession()
        }
      });

      sessionRef.current = await sessionPromise;

    } catch (err: any) {
      console.error('Session failed to start:', err);
      setError(err.message || 'Permissions or API access failed.');
      setSessionState(SessionState.DISCONNECTED);
    }
  };

  return (
    <div className="flex flex-col h-screen w-full bg-[#050508] text-white font-sans p-4 md:p-10 overflow-hidden">
      {/* Sleek Header */}
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <div className="relative flex items-center justify-center">
            <div className={`w-3 h-3 rounded-full ${sessionState === SessionState.CONNECTED ? 'bg-emerald-400' : 'bg-slate-700'}`}></div>
            {sessionState === SessionState.CONNECTED && <div className="absolute w-6 h-6 rounded-full bg-emerald-400/20 animate-ping"></div>}
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tighter uppercase italic">Vision Pro <span className="text-emerald-400">Live</span></h1>
            <p className="text-[10px] font-bold text-slate-500 tracking-[0.2em] uppercase">Hindi & English Hybrid AI</p>
          </div>
        </div>

        <button 
          onClick={() => setShowDb(!showDb)}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-full text-[10px] font-black uppercase tracking-widest transition-all ${showDb ? 'bg-emerald-400 text-black' : 'bg-white/5 border border-white/10 text-slate-400 hover:bg-white/10'}`}
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
          </svg>
          Memory
        </button>
      </header>

      {/* Main UI */}
      <div className="flex-1 flex flex-col md:flex-row gap-8 min-h-0 relative">
        
        {/* Camera Feed Container */}
        <div className={`flex-1 relative bg-black rounded-[2.5rem] overflow-hidden border border-white/5 shadow-2xl transition-all duration-500 ${showDb ? 'opacity-20 scale-95 blur-md' : 'opacity-100'}`}>
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-opacity"
          />
          <canvas ref={canvasRef} className="hidden" />
          
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/40 pointer-events-none"></div>

          {sessionState === SessionState.CONNECTED && (
            <div className="absolute top-6 left-6 flex flex-col gap-2 animate-in slide-in-from-left duration-500">
               <span className="px-4 py-1 bg-red-600 text-white text-[10px] font-black rounded-full shadow-lg flex items-center gap-2">
                 <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse"></span>
                 LIVE SENSING
               </span>
               {location && (
                 <span className="px-4 py-1 bg-black/50 backdrop-blur-md text-emerald-400 text-[9px] font-bold rounded-full border border-emerald-500/20">
                   GPS: {location.lat.toFixed(4)}, {location.lng.toFixed(4)}
                 </span>
               )}
            </div>
          )}

          {sessionState === SessionState.DISCONNECTED && (
            <div className="absolute inset-0 flex items-center justify-center p-10 text-center">
              <div className="max-w-md animate-in fade-in zoom-in duration-500">
                <div className="w-20 h-20 bg-emerald-400/10 rounded-3xl flex items-center justify-center mx-auto mb-8 border border-emerald-400/20">
                   <svg className="w-10 h-10 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <h2 className="text-3xl font-black mb-4 tracking-tighter">AI VISION ENGINE</h2>
                <p className="text-slate-400 text-sm leading-relaxed">
                  Real-time pathfinding, obstacle detection, and translation. Optimized for speed in Hindi and English.
                </p>
              </div>
            </div>
          )}

          {sessionState === SessionState.CONNECTING && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
               <div className="flex flex-col items-center gap-4">
                  <div className="w-12 h-12 border-4 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                  <p className="text-emerald-400 font-black uppercase text-[10px] tracking-[0.3em]">Connecting Live Session...</p>
               </div>
            </div>
          )}
        </div>

        {/* Memory Database View */}
        {showDb && (
          <div className="absolute inset-0 z-20 bg-[#08080a]/95 backdrop-blur-3xl rounded-[2.5rem] border border-white/10 p-10 overflow-hidden flex flex-col animate-in slide-in-from-bottom duration-500">
            <div className="flex items-center justify-between mb-8">
               <div>
                  <h2 className="text-2xl font-black tracking-tighter">Visual Memory Database</h2>
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-1">Stored visual observations and logs</p>
               </div>
               <button onClick={() => setShowDb(false)} className="w-10 h-10 bg-white/5 rounded-full flex items-center justify-center hover:bg-white/10">
                 <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                 </svg>
               </button>
            </div>
            <div className="flex-1 overflow-y-auto pr-4 space-y-4 custom-scrollbar">
               {dbEntries.length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center opacity-30 italic">
                   No recent memories logged.
                 </div>
               ) : (
                 dbEntries.map((e, idx) => (
                   <div key={idx} className="p-6 bg-white/[0.03] rounded-3xl border border-white/5 hover:border-emerald-500/30 transition-all group">
                      <div className="flex justify-between items-center mb-2">
                         <span className="text-[10px] font-black text-emerald-400 uppercase tracking-widest">Observation #{dbEntries.length - idx}</span>
                         <span className="text-[9px] font-bold text-slate-600">{new Date(e.timestamp).toLocaleTimeString()}</span>
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed group-hover:text-white transition-colors">{e.text}</p>
                   </div>
                 ))
               )}
            </div>
          </div>
        )}

        {/* Side Comms Panel */}
        <div className="w-full md:w-[420px] flex flex-col gap-6 h-full">
          
          {/* Activity Log */}
          <div className="flex-1 bg-white/[0.02] rounded-[2rem] border border-white/5 p-8 flex flex-col overflow-hidden">
             <div className="flex items-center justify-between mb-6">
                <span className="text-[10px] font-black text-slate-600 uppercase tracking-[0.3em]">Session Transcript</span>
                {sessionState === SessionState.CONNECTED && <div className="flex gap-1.5"><div className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse"></div><div className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse [animation-delay:0.2s]"></div></div>}
             </div>
             
             <div className="flex-1 overflow-y-auto space-y-6 pr-2 custom-scrollbar">
                {transcriptions.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center opacity-10 text-center">
                    <p className="text-xs font-black uppercase tracking-widest">Listening for Voice...</p>
                  </div>
                ) : (
                  transcriptions.map((t, i) => (
                    <div key={i} className={`flex flex-col ${t.type === 'user' ? 'items-end' : 'items-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                      <span className="text-[8px] font-black text-slate-600 mb-2 uppercase tracking-widest px-1">
                        {t.type === 'user' ? 'You' : 'Assistant'}
                      </span>
                      <div className={`max-w-[85%] rounded-[1.2rem] px-5 py-3.5 text-sm leading-relaxed ${
                        t.type === 'user' 
                          ? 'bg-emerald-400 text-black font-bold rounded-tr-none' 
                          : 'bg-white/5 text-slate-200 border border-white/10 rounded-tl-none backdrop-blur-xl shadow-2xl'
                      }`}>
                        {t.text}
                      </div>
                    </div>
                  ))
                )}
             </div>
          </div>

          {/* Action Area */}
          <div className="flex flex-col gap-4">
             {error && (
               <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-400 text-xs font-bold flex items-center gap-3">
                 <div className="w-2 h-2 rounded-full bg-red-500"></div>
                 {error}
               </div>
             )}

             {sessionState === SessionState.DISCONNECTED ? (
               <button
                 onClick={startSession}
                 className="group w-full bg-emerald-400 text-black font-black py-8 rounded-[2rem] text-xl transition-all active:scale-95 shadow-[0_20px_60px_-15px_rgba(52,211,153,0.3)] hover:brightness-110 flex items-center justify-center gap-4 uppercase tracking-tighter"
               >
                 <svg className="w-6 h-6 group-hover:scale-110 transition-transform" fill="currentColor" viewBox="0 0 20 20"><path d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" /></svg>
                 Connect AI
               </button>
             ) : (
               <button
                 onClick={stopSession}
                 className="w-full bg-slate-900 border border-white/10 text-white font-black py-8 rounded-[2rem] text-xl transition-all active:scale-95 flex items-center justify-center gap-4 uppercase tracking-tighter hover:bg-slate-800"
               >
                 <div className="w-3 h-3 bg-red-500 rounded-sm animate-pulse"></div>
                 Disconnect
               </button>
             )}

             <div className="p-5 bg-white/[0.01] rounded-[2rem] border border-white/5">
                <div className="grid grid-cols-2 gap-3 text-[10px] font-black uppercase tracking-widest text-slate-500">
                  <div className="p-3 bg-white/5 rounded-xl border border-white/5 text-center">"Where am I?"</div>
                  <div className="p-3 bg-white/5 rounded-xl border border-white/5 text-center">"Read this"</div>
                  <div className="p-3 bg-white/5 rounded-xl border border-white/5 text-center">"Darwaza kahan hai?"</div>
                  <div className="p-3 bg-white/5 rounded-xl border border-white/5 text-center">"Translate this"</div>
                </div>
             </div>
          </div>

        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.1); }
      `}</style>
    </div>
  );
};

export default App;
