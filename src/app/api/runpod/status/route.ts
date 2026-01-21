import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;

export async function GET(request: NextRequest) {
  try {
    if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
      return NextResponse.json(
        { error: 'RunPod configuration missing' },
        { status: 500 }
      );
    }

    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('jobId');

    if (!jobId) {
      return NextResponse.json(
        { error: 'jobId is required' },
        { status: 400 }
      );
    }

    // Check job status
    const response = await fetch(
      `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${jobId}`,
      {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${RUNPOD_API_KEY}`,
        },
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error('RunPod status error:', errorText);
      return NextResponse.json(
        { error: 'Failed to get job status' },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    return NextResponse.json({
      id: data.id,
      status: data.status,
      output: data.output,
      error: data.error,
      progress: data.progress,
    });
  } catch (error) {
    console.error('RunPod status error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
