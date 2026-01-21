import { list, del } from '@vercel/blob';
import { NextResponse } from 'next/server';

// Call this via cron job or after analysis completes
// Deletes blobs older than 24 hours
export async function POST() {
  const MAX_AGE_HOURS = 24;
  const cutoffTime = Date.now() - (MAX_AGE_HOURS * 60 * 60 * 1000);
  
  let deleted = 0;
  let cursor: string | undefined;
  
  try {
    do {
      const { blobs, cursor: nextCursor } = await list({ cursor });
      
      for (const blob of blobs) {
        const uploadTime = new Date(blob.uploadedAt).getTime();
        if (uploadTime < cutoffTime) {
          await del(blob.url);
          deleted++;
          console.log(`Deleted: ${blob.pathname}`);
        }
      }
      
      cursor = nextCursor;
    } while (cursor);
    
    return NextResponse.json({ 
      success: true, 
      deleted,
      message: `Cleaned up ${deleted} old blobs`
    });
  } catch (error) {
    console.error('Cleanup error:', error);
    return NextResponse.json({ error: 'Cleanup failed' }, { status: 500 });
  }
}

// Also allow GET for easy testing/manual trigger
export async function GET() {
  return POST();
}
