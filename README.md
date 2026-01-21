# AI Video Security Analyzer

A full-stack application for AI-powered video security analysis using SigLIP vision models and Gemini verification.

![Video Security Analyzer](https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js)
![RunPod](https://img.shields.io/badge/RunPod-Serverless-purple?style=flat-square)
![Vercel](https://img.shields.io/badge/Vercel-Blob-black?style=flat-square&logo=vercel)

## Features

- **📹 Video Upload** - Direct upload to Vercel Blob storage
- **🧠 SigLIP Analysis** - State-of-the-art vision model scores frames against threat prompts
- **✨ Gemini Verification** - Multi-frame context analysis with detailed explanations
- **📊 Timeline View** - Click-to-jump timestamps with severity scoring
- **🎯 Custom Context** - Define specific threats to detect

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Next.js App   │────▶│   Vercel Blob    │     │                 │
│   (Frontend)    │     │   (Storage)      │     │    RunPod       │
│                 │     └──────────────────┘     │   Serverless    │
│  - Upload UI    │                              │                 │
│  - Timeline     │     ┌──────────────────┐     │  - ffmpeg       │
│  - Video Player │────▶│   RunPod API     │────▶│  - SigLIP       │
│                 │     │   /run, /status  │     │  - Gemini       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Getting Started

### Prerequisites

- Node.js 18+
- Vercel account (for Blob storage)
- RunPod account (for GPU workers)
- OpenRouter account (for Gemini API)

### 1. Clone & Install

```bash
git clone <your-repo>
cd demo-ai-vid-security
npm install
```

### 2. Set Environment Variables

Create `.env.local`:

```env
# Vercel Blob Storage
BLOB_READ_WRITE_TOKEN=your_vercel_blob_token

# RunPod Serverless
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id

# OpenRouter (for Gemini)
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 3. Deploy RunPod Worker

```bash
cd runpod-worker
docker build -t your-username/video-security-worker .
docker push your-username/video-security-worker:latest
```

Then create a RunPod Serverless endpoint with this image.

### 4. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Usage

1. **Upload** - Drag & drop or click to upload surveillance video (MP4, MOV, AVI)
2. **Context** - Describe what threats to look for (optional)
3. **Analyze** - Click to start AI analysis
4. **Review** - Browse timeline events, click to jump to timestamps

## Pipeline

### Frame Extraction
- ffmpeg extracts 1-2 FPS from video
- Capped at ~100 frames max
- High-quality JPEG output

### SigLIP Scoring
Each frame is scored against a threat prompt bank:
- Breaking into buildings
- Climbing fences
- Wearing masks
- Fighting/attacking
- Stealing
- Vandalizing
- Loitering
- And more...

Custom prompts from user context are added.

### Event Segmentation
- Frames scoring above threshold (0.3) are grouped
- Consecutive high-scoring frames merge into events
- Each event gets severity rating (high/medium/low)

### Gemini Verification
For each detected event:
- 1-4 keyframes sent to Gemini 2.0 Flash
- Model confirms or rejects detection
- Provides confidence score and explanation

## Deployment

### Vercel (Frontend)

```bash
vercel
```

Add environment variables in Vercel dashboard.

### RunPod (Worker)

See `runpod-worker/README.md` for detailed instructions.

## API Routes

### POST /api/upload
Upload video to Vercel Blob.

### POST /api/runpod/run
Start async analysis job.

### GET /api/runpod/status?jobId=xxx
Poll job status until complete.

## Tech Stack

- **Frontend**: Next.js 16, React 19, Tailwind CSS
- **Storage**: Vercel Blob
- **GPU Worker**: RunPod Serverless, Python, PyTorch
- **AI Models**: SigLIP (vision), Gemini 2.0 Flash (verification)
- **Video**: ffmpeg

## License

MIT
