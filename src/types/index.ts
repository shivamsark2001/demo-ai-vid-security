export interface TimelineEvent {
  id: string;
  t0: number; // Start time in seconds
  t1: number; // End time in seconds
  label: string;
  score: number; // 0-1 confidence from SigLIP
  severity: 'high' | 'medium' | 'low';
  geminiVerdict: boolean;
  geminiConfidence: number;
  geminiExplanation: string;
  keyframeUrls?: string[];
}

export interface AnalysisResult {
  jobId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  videoUrl: string;
  videoDuration: number;
  events: TimelineEvent[];
  processedAt?: string;
  error?: string;
}

export interface RunPodResponse {
  id: string;
  status: string;
  output?: AnalysisResult;
  error?: string;
}

export type AnalysisStatus = 'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'failed';
