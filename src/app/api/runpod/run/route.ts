import { NextRequest, NextResponse } from 'next/server';

// Self-hosted API via Cloudflare tunnel
const API_URL = process.env.ANALYSIS_API_URL || 'https://realtor-governor-contractors-manager.trycloudflare.com';

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
        low_threshold: 0,
        min_frames_to_trigger: 2,
        min_frames_to_clear: 4,
      }),
    });

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
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
