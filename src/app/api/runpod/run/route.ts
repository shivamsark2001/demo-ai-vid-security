import { NextRequest, NextResponse } from 'next/server';

// Self-hosted API via Cloudflare tunnel
const API_URL = process.env.ANALYSIS_API_URL || 'https://explanation-argument-mono-produced.trycloudflare.com';

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

    console.log('Starting analysis job:', { 
      video_url: video_url.substring(0, 60) + '...', 
      camera_context: camera_context?.substring(0, 30),
      detection_targets: detection_targets?.substring(0, 30)
    });

    // Call async endpoint for job-based processing
    const response = await fetch(`${API_URL}/analyze/async`, {
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
        min_frames_to_trigger: 2,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('API error:', errorText);
      return NextResponse.json(
        { error: 'Failed to start analysis job' },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    return NextResponse.json({
      id: data.job_id || data.id,
      status: data.status || 'IN_QUEUE',
    });
  } catch (error) {
    console.error('Analysis run error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
