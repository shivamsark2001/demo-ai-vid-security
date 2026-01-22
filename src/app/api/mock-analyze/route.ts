import { NextRequest, NextResponse } from 'next/server';

// Mock endpoint - simulates sequential scan until first anomaly

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { camera_context, detection_targets } = body;

  await new Promise(resolve => setTimeout(resolve, 2000));

  // Simulate finding an anomaly
  const mockResult = {
    status: 'completed',
    videoDuration: 45,
    events: [],
    
    edgeDetection: {
      isAnomaly: true,
      scoreDiff: 0.0456,
      anomalyScore: 0.72,
      normalScore: 0.67,
      confidence: 0.0456,
      perFrameScores: [0.012, 0.034, 0.067, 0.089, 0.045, 0.023, 0.011, -0.005],
      topAnomalyPrompt: 'person behaving suspiciously near entrance',
      topNormalPrompt: 'person walking normally',
    },
    
    geminiVerification: {
      isAnomaly: true,
      anomalyType: 'Suspicious Surveillance Behavior',
      confidence: 0.84,
      reasoning: `After sequential analysis, an anomaly was detected at approximately 12 seconds into the video. The individual exhibits behavior inconsistent with normal pedestrian traffic - repeatedly pausing, scanning surroundings, and appearing to check camera positions. Frames 3-5 show the clearest evidence of this behavior pattern with anomaly scores exceeding the threshold.`,
      keyObservations: [
        'Individual pauses repeatedly near entrance area',
        'Scanning behavior detected in frames 3-5',
        'Movement pattern inconsistent with normal traffic',
        'Apparent awareness of surveillance positions'
      ],
      frameAnalysis: 'Frames 3-5 show peak anomaly scores (+0.067 to +0.089). Frame 4 captures the subject mid-turn, appearing to check surroundings. The edge detector correctly flagged these frames.'
    },
    
    annotatedGridB64: null,
    frameCount: 8,
    anomalyTimestamp: 12.0,
    framesScanned: 24,
  };

  return NextResponse.json(mockResult);
}
