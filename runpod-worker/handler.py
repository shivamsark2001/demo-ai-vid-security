#!/usr/bin/env python3
"""
RunPod Serverless Handler - Video Security Analysis
====================================================
2-Stage Pipeline: SigLIP2 Edge Detection → Gemini Verification
Returns full pipeline data for enhanced UI
"""

import os
import json
import uuid
import base64
import tempfile
import subprocess
import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from io import BytesIO

import runpod
import requests
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============ Configuration ============
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GEMINI_MODEL = "google/gemini-3-flash-preview"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️  Device: {DEVICE}")
print(f"🔑 API Key: {'Set' if OPENROUTER_API_KEY else 'NOT SET'}")

# ============ Model Loading ============
SIGLIP_MODEL = None
SIGLIP_PREPROCESS = None
SIGLIP_TOKENIZER = None
USE_OPEN_CLIP = False


@dataclass
class TimelineEvent:
    id: str
    t0: float
    t1: float
    label: str
    score: float
    severity: str
    geminiVerdict: bool
    geminiConfidence: float
    geminiExplanation: str
    keyframeUrls: List[str]


def load_siglip():
    """Load SigLIP2 model."""
    global SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER, USE_OPEN_CLIP
    
    if SIGLIP_MODEL is not None:
        return SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER
    
    try:
        import open_clip
        print("📦 Loading SigLIP2 via open_clip...")
        SIGLIP_MODEL, _, SIGLIP_PREPROCESS = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384", pretrained="webli"
        )
        SIGLIP_MODEL = SIGLIP_MODEL.to(DEVICE).eval()
        SIGLIP_TOKENIZER = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
        USE_OPEN_CLIP = True
        print(f"✅ SigLIP2 loaded on {DEVICE}")
    except ImportError:
        from transformers import AutoProcessor, AutoModel
        model_name = "google/siglip-so400m-patch14-384"
        SIGLIP_PREPROCESS = AutoProcessor.from_pretrained(model_name)
        SIGLIP_MODEL = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
        SIGLIP_TOKENIZER = SIGLIP_PREPROCESS
        USE_OPEN_CLIP = False
        print(f"✅ SigLIP loaded via transformers on {DEVICE}")
    
    return SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER


def extract_json_object(text: str) -> dict | None:
    """Robustly extract JSON from LLM response."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    for match in re.finditer(r"\{", text):
        start = match.start()
        for end in range(len(text), start, -1):
            if text[end-1] != "}":
                continue
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    return None


def call_gemini(prompt: str, image_b64: str = None, max_tokens: int = 1000) -> str | dict:
    """Call Gemini via OpenRouter."""
    if not OPENROUTER_API_KEY:
        return {"error": "No OPENROUTER_API_KEY"}
    
    content = [{"type": "text", "text": prompt}]
    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })
    
    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GEMINI_MODEL,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1
                },
                timeout=60
            )
            result = response.json()
            if "error" in result:
                continue
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            if attempt == 2:
                return {"error": str(e)}
    return {"error": "Failed after 3 attempts"}


def generate_prompt_banks(camera_context: str, detection_targets: str) -> tuple:
    """Generate normal/anomaly prompts using Gemini."""
    print("🤖 Generating prompt banks via Gemini...")
    
    prompt = f"""You are an expert in video anomaly detection. Generate prompt banks for a detection pipeline.

CAMERA CONTEXT: {camera_context}
DETECTION TARGETS: {detection_targets}

Generate two sets of text prompts for CLIP/SigLIP image-text matching:

1. NORMAL PROMPTS: 15-20 descriptions of normal, expected behavior
2. ANOMALY PROMPTS: 15-20 descriptions of abnormal behavior to detect

Each prompt: short visual description (5-15 words), focus on what's VISIBLE in a frame.

Respond in JSON only:
{{
    "normal_prompts": ["desc1", "desc2", ...],
    "anomaly_prompts": ["desc1", "desc2", ...],
    "detection_summary": "Brief summary of what will be detected"
}}"""

    response = call_gemini(prompt, max_tokens=1500)
    
    if isinstance(response, dict) and "error" in response:
        return None, f"Error: {response['error']}"
    
    parsed = extract_json_object(response)
    if parsed and "normal_prompts" in parsed and "anomaly_prompts" in parsed:
        print(f"  ✅ Generated {len(parsed['normal_prompts'])} normal + {len(parsed['anomaly_prompts'])} anomaly prompts")
        return parsed, None
    
    return None, f"Parse failed: {response[:300]}"


def download_video(url: str, output_path: str) -> bool:
    """Download video from URL."""
    print(f"📥 Downloading video...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✅ Downloaded {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False


def extract_frames(video_path: str, output_dir: str, fps: float = 2, max_frames: int = 32) -> List[Dict]:
    """Extract frames using ffmpeg."""
    print(f"🎞️  Extracting frames...")
    
    try:
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]).decode().strip()
        duration = float(probe)
    except:
        duration = 60
    
    if duration * fps > max_frames:
        fps = max_frames / duration
    
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vf", f"fps={fps}",
        "-q:v", "2", "-frames:v", str(max_frames), output_pattern
    ], capture_output=True)
    
    frames = []
    for i, f in enumerate(sorted(Path(output_dir).glob("frame_*.jpg"))):
        frames.append({"path": str(f), "timestamp": i / fps, "index": i})
    
    print(f"  ✅ Extracted {len(frames)} frames")
    return frames


def siglip_detect(frames: List[Dict], normal_prompts: List[str], anomaly_prompts: List[str]) -> Dict:
    """Stage 1: Score frames against prompts."""
    print(f"⚡ Running SigLIP detection...")
    
    model, preprocess, tokenizer = load_siglip()
    
    images = []
    for f in frames:
        try:
            img = Image.open(f["path"]).convert("RGB")
            images.append(img)
        except:
            continue
    
    if not images:
        return {"error": "No valid frames"}
    
    if USE_OPEN_CLIP:
        processed = torch.stack([preprocess(img) for img in images]).to(DEVICE)
        
        with torch.no_grad():
            img_features = model.encode_image(processed)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            normal_tokens = tokenizer(normal_prompts).to(DEVICE)
            anomaly_tokens = tokenizer(anomaly_prompts).to(DEVICE)
            
            normal_features = model.encode_text(normal_tokens)
            normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
            
            anomaly_features = model.encode_text(anomaly_tokens)
            anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
        
        normal_sim = (img_features @ normal_features.T).max(dim=1).values
        anomaly_sim = (img_features @ anomaly_features.T).max(dim=1).values
        per_frame_scores = (anomaly_sim - normal_sim).cpu().numpy().tolist()
        
        avg_normal = normal_sim.mean().item()
        avg_anomaly = anomaly_sim.mean().item()
        
        best_anomaly_idx = (img_features @ anomaly_features.T).mean(dim=0).argmax().item()
        best_normal_idx = (img_features @ normal_features.T).mean(dim=0).argmax().item()
    else:
        per_frame_scores = []
        all_anomaly = []
        all_normal = []
        
        for img in images:
            inputs = preprocess(
                text=normal_prompts + anomaly_prompts,
                images=img, return_tensors="pt", padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.sigmoid(logits).cpu().numpy()
            
            n = len(normal_prompts)
            max_normal = float(probs[:n].max())
            max_anomaly = float(probs[n:].max())
            
            per_frame_scores.append(max_anomaly - max_normal)
            all_normal.append(max_normal)
            all_anomaly.append(max_anomaly)
        
        avg_normal = np.mean(all_normal)
        avg_anomaly = np.mean(all_anomaly)
        best_anomaly_idx = 0
        best_normal_idx = 0
    
    score_diff = avg_anomaly - avg_normal
    is_anomaly = score_diff > 0
    
    print(f"  📊 Score diff: {score_diff:+.4f} → {'🚨 ANOMALY' if is_anomaly else '✅ NORMAL'}")
    
    return {
        "isAnomaly": is_anomaly,
        "scoreDiff": round(score_diff, 4),
        "anomalyScore": round(avg_anomaly, 4),
        "normalScore": round(avg_normal, 4),
        "confidence": round(abs(score_diff), 4),
        "perFrameScores": [round(s, 4) for s in per_frame_scores],
        "topAnomalyPrompt": anomaly_prompts[best_anomaly_idx],
        "topNormalPrompt": normal_prompts[best_normal_idx],
    }


def create_annotated_grid(frames: List[Dict], scores: List[float], cols=4, tile_size=220) -> Image.Image:
    """Create annotated frame grid with scores."""
    images = [Image.open(f["path"]).convert("RGB") for f in frames[:16]]
    n = len(images)
    rows = (n + cols - 1) // cols
    header = 32
    
    grid = Image.new('RGB', (cols * tile_size, rows * (tile_size + header)), (9, 9, 11))
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, img in enumerate(images):
        row, col = i // cols, i % cols
        x, y = col * tile_size, row * (tile_size + header)
        
        tile = img.resize((tile_size - 4, tile_size - 4), Image.BICUBIC)
        grid.paste(tile, (x + 2, y + header))
        
        score = scores[i] if i < len(scores) else 0
        if score > 0.015:
            bg, color = (80, 25, 25), (255, 120, 120)
            label = f"⚠ F{i+1}: {score:+.3f}"
        elif score < -0.015:
            bg, color = (25, 60, 25), (120, 255, 120)
            label = f"✓ F{i+1}: {score:+.3f}"
        else:
            bg, color = (30, 30, 40), (160, 160, 170)
            label = f"○ F{i+1}: {score:+.3f}"
        
        draw.rectangle([x, y, x + tile_size - 1, y + header - 1], fill=bg)
        draw.text((x + 6, y + 8), label, fill=color, font=font)
    
    return grid


def image_to_b64(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def gemini_verify(frames: List[Dict], camera_context: str, detection_targets: str, edge_result: Dict, grid_b64: str) -> Dict:
    """Stage 2: Gemini verification."""
    print("🧠 Running Gemini verification...")
    
    per_frame_desc = "\n".join([
        f"  Frame {i+1}: {s:+.3f} ({'anomaly' if s > 0.01 else 'normal' if s < -0.01 else 'uncertain'})"
        for i, s in enumerate(edge_result.get('perFrameScores', [])[:16])
    ])
    
    prompt = f"""VIDEO ANOMALY VERIFICATION

CAMERA: {camera_context}
TARGETS: {detection_targets}

EDGE DETECTOR (SigLIP2) RESULT:
- Assessment: {"ANOMALY" if edge_result.get('isAnomaly') else "NORMAL"}
- Score diff: {edge_result.get('scoreDiff', 0):+.4f}
- Top anomaly match: "{edge_result.get('topAnomalyPrompt', '')}"
- Top normal match: "{edge_result.get('topNormalPrompt', '')}"

PER-FRAME SCORES:
{per_frame_desc}

The image shows a frame grid with scores. Analyze carefully.

Respond in JSON:
{{
    "isAnomaly": true/false,
    "anomalyType": "specific type or null",
    "confidence": 0.0-1.0,
    "reasoning": "detailed analysis of what you see",
    "keyObservations": ["observation 1", "observation 2", "observation 3"],
    "frameAnalysis": "which specific frames show the anomaly and why"
}}"""

    response = call_gemini(prompt, grid_b64, max_tokens=1000)
    
    if isinstance(response, dict) and "error" in response:
        print(f"  ❌ Gemini error: {response['error']}")
        return {"error": response["error"]}
    
    parsed = extract_json_object(response)
    if parsed:
        print(f"  ✅ Gemini verdict: {'🚨 ANOMALY' if parsed.get('isAnomaly') else '✅ NORMAL'}")
        return {
            "isAnomaly": parsed.get("isAnomaly", False),
            "anomalyType": parsed.get("anomalyType"),
            "confidence": parsed.get("confidence", 0),
            "reasoning": parsed.get("reasoning", ""),
            "keyObservations": parsed.get("keyObservations", []),
            "frameAnalysis": parsed.get("frameAnalysis", "")
        }
    
    return {
        "reasoning": response,
        "isAnomaly": "anomaly" in response.lower() and "no anomaly" not in response.lower()
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler - returns full pipeline data.
    """
    job_input = job.get("input", {})
    
    video_url = job_input.get("video_url")
    camera_context = job_input.get("camera_context", "Security surveillance camera")
    detection_targets = job_input.get("detection_targets", "Suspicious activity")
    fps = job_input.get("fps", 2)
    max_frames = job_input.get("max_frames", 32)
    
    if not video_url:
        return {"error": "video_url is required"}
    
    print("\n" + "="*50)
    print("🎬 TRUWO VIDEO ANALYSIS")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "video.mp4")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)
        
        # Step 1: Download
        if not download_video(video_url, video_path):
            return {"error": "Download failed"}
        
        # Step 2: Extract frames
        frames = extract_frames(video_path, frames_dir, fps=fps, max_frames=max_frames)
        if not frames:
            return {"error": "No frames extracted"}
        
        # Step 3: Generate prompts
        prompt_banks, error = generate_prompt_banks(camera_context, detection_targets)
        if error:
            return {"error": f"Prompt generation failed: {error}"}
        
        # Step 4: SigLIP detection
        edge_result = siglip_detect(
            frames,
            prompt_banks["normal_prompts"],
            prompt_banks["anomaly_prompts"]
        )
        
        if "error" in edge_result:
            return {"error": edge_result["error"]}
        
        # Create annotated grid
        grid = create_annotated_grid(frames, edge_result.get("perFrameScores", []))
        grid_b64 = image_to_b64(grid)
        
        # Step 5: Gemini verification
        gemini_result = gemini_verify(frames, camera_context, detection_targets, edge_result, grid_b64)
        
        # Final verdict
        final_is_anomaly = gemini_result.get("isAnomaly", edge_result["isAnomaly"])
        final_confidence = gemini_result.get("confidence", edge_result["confidence"])
        
        print("\n" + "="*50)
        print(f"🎯 VERDICT: {'🚨 ANOMALY' if final_is_anomaly else '✅ NORMAL'}")
        print("="*50)
        
        # Build events
        events = []
        if final_is_anomaly:
            first_ts = frames[0]["timestamp"]
            last_ts = frames[-1]["timestamp"]
            severity = "high" if final_confidence >= 0.7 else "medium" if final_confidence >= 0.5 else "low"
            
            event = TimelineEvent(
                id=str(uuid.uuid4()),
                t0=first_ts,
                t1=last_ts,
                label=gemini_result.get("anomalyType") or edge_result.get("topAnomalyPrompt", "Anomaly"),
                score=final_confidence,
                severity=severity,
                geminiVerdict=True,
                geminiConfidence=final_confidence,
                geminiExplanation=gemini_result.get("reasoning", ""),
                keyframeUrls=[]
            )
            events.append(asdict(event))
        
        video_duration = frames[-1]["timestamp"] if frames else 0
        
        # Return full pipeline data
        return {
            "status": "completed",
            "videoDuration": video_duration,
            "events": events,
            
            # Enhanced data for UI
            "promptBanks": {
                "normalPrompts": prompt_banks.get("normal_prompts", []),
                "anomalyPrompts": prompt_banks.get("anomaly_prompts", []),
                "detectionSummary": prompt_banks.get("detection_summary", "")
            },
            "edgeDetection": edge_result,
            "geminiVerification": {
                "isAnomaly": gemini_result.get("isAnomaly", False),
                "anomalyType": gemini_result.get("anomalyType"),
                "confidence": gemini_result.get("confidence", 0),
                "reasoning": gemini_result.get("reasoning", ""),
                "keyObservations": gemini_result.get("keyObservations", []),
                "frameAnalysis": gemini_result.get("frameAnalysis", "")
            },
            "annotatedGridB64": f"data:image/jpeg;base64,{grid_b64}",
            "frameCount": len(frames)
        }


runpod.serverless.start({"handler": handler})
