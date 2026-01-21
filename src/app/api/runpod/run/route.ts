import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;

export async function POST(request: NextRequest) {
  try {
    if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
      return NextResponse.json(
        { error: 'RunPod configuration missing' },
        { status: 500 }
      );
    }

    const body = await request.json();
    const { video_url, context_text } = body;

    if (!video_url) {
      return NextResponse.json(
        { error: 'video_url is required' },
        { status: 400 }
      );
    }

    // Call RunPod async endpoint
    const response = await fetch(
      `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${RUNPOD_API_KEY}`,
        },
        body: JSON.stringify({
          input: {
            video_url,
            context_text: context_text || 'Detect any suspicious or threatening behavior',
            fps: 2, // Extract 2 frames per second
            max_frames: 100, // Cap at 100 frames
          },
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error('RunPod error:', errorText);
      return NextResponse.json(
        { error: 'Failed to start analysis job' },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    return NextResponse.json({
      id: data.id,
      status: data.status,
    });
  } catch (error) {
    console.error('RunPod run error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
