import { NextRequest, NextResponse } from 'next/server';

// Mock endpoint for local development/testing without RunPod
// Returns simulated analysis results with full pipeline data

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { camera_context, detection_targets } = body;

  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Generate mock events
  const mockEvents = [
    {
      id: 'evt-1',
      t0: 5.0,
      t1: 18.0,
      label: 'Suspicious Activity Detected',
      score: 0.85,
      severity: 'high' as const,
      geminiVerdict: true,
      geminiConfidence: 0.88,
      geminiExplanation: 'Individual observed behaving unusually near the monitored area. The person appears to be checking surroundings repeatedly and moving in an irregular pattern that differs from normal pedestrian traffic.',
      keyframeUrls: [],
    },
  ];

  // Mock prompt banks
  const mockPromptBanks = {
    normalPrompts: [
      'person walking normally through area',
      'employees working at their stations',
      'regular pedestrian traffic flow',
      'people having normal conversation',
      'standard vehicle movement',
      'routine security patrol',
      'customers browsing normally',
      'delivery person with package',
      'maintenance worker with equipment',
      'group of people walking together',
    ],
    anomalyPrompts: [
      'person running away quickly',
      'suspicious loitering behavior',
      'unauthorized access attempt',
      'aggressive confrontation',
      'person concealing items',
      'unusual gathering of people',
      'person checking surroundings nervously',
      'forced entry attempt',
      'person in restricted area',
      'erratic or panicked movement',
    ],
    detectionSummary: camera_context 
      ? `Monitoring for: ${detection_targets || 'suspicious activity'}`
      : 'General security monitoring for anomalous behavior patterns',
  };

  // Mock edge detection results
  const mockEdgeDetection = {
    isAnomaly: true,
    scoreDiff: 0.0342,
    anomalyScore: 0.7234,
    normalScore: 0.6892,
    confidence: 0.0342,
    perFrameScores: [
      -0.012, -0.008, 0.003, 0.018, 0.045, 0.067, 0.089, 0.102,
      0.095, 0.078, 0.056, 0.034, 0.021, 0.008, -0.005, -0.011,
    ],
    topAnomalyPrompt: 'person checking surroundings nervously',
    topNormalPrompt: 'person walking normally through area',
  };

  // Mock Gemini verification
  const mockGeminiVerification = {
    isAnomaly: true,
    anomalyType: 'Suspicious Surveillance Behavior',
    confidence: 0.88,
    reasoning: 'After analyzing the 16-frame sequence, I observe a clear pattern of suspicious behavior. In frames 5-10 (marked with high anomaly scores), an individual is seen repeatedly scanning the environment while lingering near what appears to be an entrance or sensitive area. The person\'s movement pattern is inconsistent with typical pedestrian flow - they pause frequently, change direction abruptly, and appear to be timing their movements. This behavior is consistent with pre-surveillance reconnaissance activity.',
    keyObservations: [
      'Individual exhibits repeated scanning behavior (frames 5-8)',
      'Unusual pause duration at specific locations',
      'Movement pattern inconsistent with normal traffic flow',
      'Apparent awareness of camera positions',
    ],
    frameAnalysis: 'Frames 5-10 show the most significant anomaly indicators. Frame 7 captures the subject mid-turn, appearing to check behind them. Frames 8-9 show extended dwell time near the entrance. The edge detector correctly flagged these frames with positive scores (+0.067 to +0.102).',
  };

  return NextResponse.json({
    status: 'completed',
    videoDuration: 60,
    events: mockEvents,
    
    // Enhanced pipeline data
    promptBanks: mockPromptBanks,
    edgeDetection: mockEdgeDetection,
    geminiVerification: mockGeminiVerification,
    annotatedGridB64: null, // Would be base64 image from real analysis
    frameCount: 16,
  });
}
