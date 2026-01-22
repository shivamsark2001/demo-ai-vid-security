import { NextRequest, NextResponse } from 'next/server';

// Self-hosted API via Cloudflare tunnel
const API_URL = process.env.ANALYSIS_API_URL || 'https://realtor-governor-contractors-manager.trycloudflare.com';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('jobId');

    if (!jobId) {
      return NextResponse.json(
        { error: 'jobId is required' },
        { status: 400 }
      );
    }

    // Poll job status from our API
    const response = await fetch(`${API_URL}/job/${jobId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('API status error:', errorText);
      return NextResponse.json(
        { error: 'Failed to get job status' },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    // Map API response to expected format
    // API returns: { status, result?, error? }
    // Frontend expects: { id, status, output, error, progress }
    
    let mappedStatus = data.status;
    // Normalize status values
    if (data.status === 'pending' || data.status === 'queued') {
      mappedStatus = 'IN_QUEUE';
    } else if (data.status === 'processing' || data.status === 'running') {
      mappedStatus = 'IN_PROGRESS';
    } else if (data.status === 'completed' || data.status === 'complete') {
      mappedStatus = 'COMPLETED';
    } else if (data.status === 'failed' || data.status === 'error') {
      mappedStatus = 'FAILED';
    }
    
    return NextResponse.json({
      id: jobId,
      status: mappedStatus,
      output: data.result || data.output,
      error: data.error,
      progress: data.progress,
    });
  } catch (error) {
    console.error('API status error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
