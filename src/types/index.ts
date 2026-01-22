export interface TimelineEvent {
  id: string;
  t0: number;
  t1: number;
  label: string;
  score: number;
  severity: 'high' | 'medium' | 'low';
  geminiVerdict: boolean;
  geminiConfidence: number;
  geminiExplanation: string;
  keyframeUrls?: string[];
}

export interface PromptBanks {
  normalPrompts: string[];
  anomalyPrompts: string[];
  detectionSummary: string;
}

export interface EdgeDetection {
  isAnomaly: boolean;
  scoreDiff: number;
  anomalyScore: number;
  normalScore: number;
  confidence: number;
  perFrameScores: number[];
  topAnomalyPrompt: string;
  topNormalPrompt: string;
}

export interface GeminiVerification {
  isAnomaly: boolean;
  anomalyType: string | null;
  confidence: number;
  reasoning: string;
  keyObservations: string[];
  frameAnalysis: string;
}

export interface AnalysisResult {
  status: string;
  videoDuration: number;
  events: TimelineEvent[];
  
  // Enhanced pipeline data
  promptBanks?: PromptBanks;
  edgeDetection?: EdgeDetection;
  geminiVerification?: GeminiVerification;
  annotatedGridB64?: string;
  frameCount?: number;
  
  error?: string;
}

export interface RunPodResponse {
  id: string;
  status: string;
  output?: AnalysisResult;
  error?: string;
}

export type AnalysisStatus = 'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'failed';
