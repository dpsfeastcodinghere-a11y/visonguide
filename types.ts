
export interface TranscriptionEntry {
  text: string;
  type: 'user' | 'ai';
  timestamp: number;
}

export enum SessionState {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  ERROR = 'ERROR'
}
