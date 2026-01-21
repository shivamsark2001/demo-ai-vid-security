# RunPod Video Security Worker

This is the serverless worker that processes videos for security analysis.

## Pipeline

1. **Download Video** - Fetches video from Vercel Blob URL
2. **Extract Frames** - Uses ffmpeg to extract 1-2 FPS, capped at ~100 frames
3. **SigLIP Scoring** - Scores each frame against threat prompt bank
4. **Event Segmentation** - Groups high-scoring frames into timeline events
5. **Gemini Verification** - Sends keyframes to Gemini for confirmation & explanation

## Building & Deploying

### Build Docker Image

```bash
docker build -t video-security-worker .
```

### Test Locally

```bash
docker run --gpus all -e OPENROUTER_API_KEY=your_key video-security-worker
```

### Push to Docker Hub

```bash
docker tag video-security-worker yourusername/video-security-worker:latest
docker push yourusername/video-security-worker:latest
```

### Deploy to RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create new endpoint
3. Use your Docker image: `yourusername/video-security-worker:latest`
4. Configure:
   - GPU Type: RTX 3090 or better recommended
   - Max Workers: 3
   - Idle Timeout: 5 seconds
5. Add environment variables:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key for Gemini

### Get Endpoint ID

After deployment, copy the Endpoint ID and add it to your Next.js app:

```env
RUNPOD_ENDPOINT_ID=your_endpoint_id
RUNPOD_API_KEY=your_runpod_api_key
```

## API

### Input

```json
{
  "input": {
    "video_url": "https://...",
    "context_text": "Detect unauthorized access",
    "fps": 2,
    "max_frames": 100
  }
}
```

### Output

```json
{
  "jobId": "...",
  "status": "completed",
  "videoUrl": "...",
  "videoDuration": 120.5,
  "events": [
    {
      "id": "uuid",
      "t0": 15.0,
      "t1": 18.5,
      "label": "Breaking Into A Building",
      "score": 0.75,
      "severity": "high",
      "geminiVerdict": true,
      "geminiConfidence": 0.85,
      "geminiExplanation": "Person seen forcing entry through side door"
    }
  ],
  "processedAt": "2024-01-15T10:30:00Z"
}
```

## Threat Prompt Bank

The SigLIP model scores frames against these prompts:

- Breaking into a building
- Climbing over a fence
- Wearing a mask or balaclava
- Running away quickly
- Carrying a weapon
- Fighting or attacking
- Stealing or taking something
- Vandalizing property
- Loitering suspiciously
- Trespassing
- And more...

Custom prompts from user context are also added.
