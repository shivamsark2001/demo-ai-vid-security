import { NextRequest, NextResponse } from 'next/server';

// Mock endpoint - simulates the new simplified output format

export async function POST(request: NextRequest) {
  await request.json(); // consume body

  await new Promise(resolve => setTimeout(resolve, 2000));

  // Simulate finding an anomaly with the new simplified format
  const mockResult = {
    category: 'theft',
    reasoning: 'In frames 1-3, a person is seen approaching the counter area with normal posture. By frame 4, their body language shifts - they lean forward unnaturally while their hand moves toward merchandise on the display. Frames 5-7 clearly show concealment behavior: the item is grabbed and quickly moved toward their jacket pocket. Frame 8 shows them stepping back with hands now empty but jacket bulging. This sequence matches the visual pattern of shoplifting.',
    anomalyFramesB64: null, // Would be a base64 2x4 grid in real response
  };

  return NextResponse.json(mockResult);
}
