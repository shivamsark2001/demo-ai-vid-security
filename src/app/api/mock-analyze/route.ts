import { NextRequest, NextResponse } from 'next/server';

// Mock endpoint for local development/testing without RunPod
// Returns simulated analysis results

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { video_url } = body;

  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Generate mock events
  const mockEvents = [
    {
      id: 'evt-1',
      t0: 5.0,
      t1: 12.0,
      label: 'Suspicious Loitering',
      score: 0.78,
      severity: 'medium',
      geminiVerdict: true,
      geminiConfidence: 0.82,
      geminiExplanation: 'Individual observed lingering near entrance for extended period, repeatedly checking surroundings.',
      keyframeUrls: [],
    },
    {
      id: 'evt-2',
      t0: 25.0,
      t1: 35.0,
      label: 'Unauthorized Access Attempt',
      score: 0.91,
      severity: 'high',
      geminiVerdict: true,
      geminiConfidence: 0.95,
      geminiExplanation: 'Person attempting to manipulate door lock mechanism without proper credentials.',
      keyframeUrls: [],
    },
    {
      id: 'evt-3',
      t0: 48.0,
      t1: 52.0,
      label: 'Quick Exit',
      score: 0.45,
      severity: 'low',
      geminiVerdict: false,
      geminiConfidence: 0.35,
      geminiExplanation: 'Person walking quickly but appears to be normal pedestrian traffic.',
      keyframeUrls: [],
    },
  ];

  return NextResponse.json({
    jobId: 'mock-' + Date.now(),
    status: 'completed',
    videoUrl: video_url,
    videoDuration: 60,
    events: mockEvents,
    processedAt: new Date().toISOString(),
  });
}
