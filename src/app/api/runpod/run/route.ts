import { NextRequest, NextResponse } from 'next/server';

// Self-hosted API via Cloudflare tunnel
const API_URL = process.env.ANALYSIS_API_URL || 'https://than-connector-additional-surgeons.trycloudflare.com';

// Timeout for video analysis (5 minutes - videos can take a while to process)
const ANALYSIS_TIMEOUT_MS = 5 * 60 * 1000; // 300 seconds

// Allow this serverless function to run for up to 5 minutes (Vercel Pro/Enterprise)
// On Vercel Hobby plan, max is 60 seconds - upgrade if needed for long videos
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { video_url, camera_context, detection_targets } = body;

    if (!video_url) {
      return NextResponse.json(
        { error: 'video_url is required' },
        { status: 400 }
      );
    }

    // SAFEGUARD: Reject browser blob: URLs
    if (video_url.startsWith('blob:')) {
      console.error('Rejected blob: URL:', video_url);
      return NextResponse.json(
        { error: 'Invalid video URL - browser blob URLs are not supported' },
        { status: 400 }
      );
    }

    console.log('Starting analysis:', { 
      video_url: video_url.substring(0, 60) + '...', 
      camera_context: camera_context?.substring(0, 30),
      detection_targets: detection_targets?.substring(0, 30)
    });

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), ANALYSIS_TIMEOUT_MS);

    // Use synchronous /analyze endpoint directly
    const response = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        video_url,
        camera_context: camera_context || 'Security surveillance camera',
        detection_targets: detection_targets || 'Suspicious or threatening behavior',
        sample_fps: 2.0,
        high_threshold: 0.015,
        low_threshold: 0.005,
        min_frames_to_trigger: 2,
        min_frames_to_clear: 3,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('API error:', response.status, errorText);
      return NextResponse.json(
        { error: `Analysis failed: ${errorText}` },
        { status: response.status }
      );
    }

    const result = await response.json();
    console.log('Analysis complete:', result.category);
    
    // Return result directly (no job ID needed for sync)
    return NextResponse.json({
      id: 'sync',
      status: 'COMPLETED',
      output: result,
    });
  } catch (error) {
    console.error('Analysis error:', error);
    
    // Check if it's a timeout error
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        { error: 'Analysis timed out after 5 minutes. Try a shorter video or check if the backend is processing correctly.' },
        { status: 504 }
      );
    }
    
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
