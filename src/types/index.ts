export interface AnalysisResult {
  category: string;
  reasoning: string;
  anomalyFramesB64?: string | null;
  error?: string;
}

export interface RunPodResponse {
  id: string;
  status: string;
  output?: AnalysisResult;
  error?: string;
}

export type AnalysisStatus = 'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'failed';
