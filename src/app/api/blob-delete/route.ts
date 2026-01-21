import { del } from '@vercel/blob';
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { url } = await request.json();
    
    if (!url) {
      return NextResponse.json({ error: 'URL required' }, { status: 400 });
    }
    
    // Only delete if it's a Vercel blob URL
    if (!url.includes('vercel-storage.com') && !url.includes('blob.vercel-storage.com')) {
      return NextResponse.json({ error: 'Not a Vercel blob URL' }, { status: 400 });
    }
    
    await del(url);
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Delete error:', error);
    return NextResponse.json({ error: 'Delete failed' }, { status: 500 });
  }
}
