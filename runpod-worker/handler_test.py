#!/usr/bin/env python3
"""
Video Security Analysis - Standalone Test Version
==================================================
Run this directly on your A40 pod for fast iteration.

Usage:
    python handler_test.py --video_url "https://..." --camera_context "..." --detection_targets "..."
    
Or edit the TEST_INPUT at the bottom and run:
    python handler_test.py
"""

import os
import json
import uuid
import base64
import tempfile
import subprocess
import re
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from io import BytesIO

import requests
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============ Configuration ============
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")  # Set via environment variable
GEMINI_MODEL = "google/gemini-2.0-flash-001"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️  Device: {DEVICE}")
print(f"🔑 API Key: {'Set' if OPENROUTER_API_KEY else 'NOT SET'}")

# ============ Model Loading ============
SIGLIP_MODEL = None
SIGLIP_PREPROCESS = None
SIGLIP_TOKENIZER = None
USE_OPEN_CLIP = False

def load_siglip():
    """Load SigLIP2 model - tries open_clip first, falls back to transformers."""
    global SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER, USE_OPEN_CLIP
    
    if SIGLIP_MODEL is not None:
        return SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER
    
    # Try open_clip first (better performance)
    try:
        import open_clip
        print("📦 Loading SigLIP2 via open_clip...")
        SIGLIP_MODEL, _, SIGLIP_PREPROCESS = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384", pretrained="webli"
        )
        SIGLIP_MODEL = SIGLIP_MODEL.to(DEVICE).eval()
        SIGLIP_TOKENIZER = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
        USE_OPEN_CLIP = True
        print(f"✅ SigLIP2 loaded via open_clip on {DEVICE}")
    except ImportError:
        print("⚠️  open_clip not found, falling back to transformers...")
        from transformers import AutoProcessor, AutoModel
        model_name = "google/siglip-so400m-patch14-384"
        print(f"📦 Loading {model_name} via transformers...")
        SIGLIP_PREPROCESS = AutoProcessor.from_pretrained(model_name)
        SIGLIP_MODEL = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
        SIGLIP_TOKENIZER = SIGLIP_PREPROCESS
        USE_OPEN_CLIP = False
        print(f"✅ SigLIP loaded via transformers on {DEVICE}")
    
    return SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER


# ============ Robust JSON Extraction ============
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


# ============ Gemini API ============
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


# ============ Prompt Bank Generation ============
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
    "detection_summary": "Brief summary"
}}"""

    response = call_gemini(prompt, max_tokens=1500)
    
    if isinstance(response, dict) and "error" in response:
        return None, f"Error: {response['error']}"
    
    parsed = extract_json_object(response)
    if parsed and "normal_prompts" in parsed and "anomaly_prompts" in parsed:
        print(f"  ✅ Generated {len(parsed['normal_prompts'])} normal + {len(parsed['anomaly_prompts'])} anomaly prompts")
        return parsed, None
    
    return None, f"Parse failed: {response[:300]}"


# ============ Video Processing ============
def download_video(url: str, output_path: str) -> bool:
    """Download video from URL."""
    print(f"📥 Downloading video from {url[:60]}...")
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


def extract_frames(video_path: str, output_dir: str, fps: float = 2, max_frames: int = 64) -> List[Dict]:
    """Extract frames using ffmpeg."""
    print(f"🎞️  Extracting frames (fps={fps}, max={max_frames})...")
    
    # Get duration
    try:
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]).decode().strip()
        duration = float(probe)
    except:
        duration = 60
    
    # Adjust fps
    if duration * fps > max_frames:
        fps = max_frames / duration
    
    # Extract
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


# ============ SigLIP Detection ============
def siglip_detect(frames: List[Dict], normal_prompts: List[str], anomaly_prompts: List[str]) -> Dict:
    """Stage 1: Score frames against prompts."""
    print(f"⚡ Running SigLIP detection on {len(frames)} frames...")
    
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
        # Transformers fallback
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
    print(f"  📊 Per-frame scores: min={min(per_frame_scores):.3f}, max={max(per_frame_scores):.3f}")
    print(f"  📊 Top anomaly prompt: {anomaly_prompts[best_anomaly_idx]}")
    
    return {
        "is_anomaly": is_anomaly,
        "score_diff": score_diff,
        "anomaly_score": avg_anomaly,
        "normal_score": avg_normal,
        "confidence": abs(score_diff),
        "per_frame_scores": per_frame_scores,
        "top_anomaly_prompt": anomaly_prompts[best_anomaly_idx],
        "top_normal_prompt": normal_prompts[best_normal_idx],
    }


# ============ Frame Grid for Gemini ============
def create_annotated_grid(frames: List[Dict], scores: List[float], cols=4, tile_size=256) -> Image.Image:
    """Create annotated frame grid."""
    images = [Image.open(f["path"]).convert("RGB") for f in frames[:16]]
    n = len(images)
    rows = (n + cols - 1) // cols
    header = 36
    
    grid = Image.new('RGB', (cols * tile_size, rows * (tile_size + header)), (12, 12, 20))
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    for i, img in enumerate(images):
        row, col = i // cols, i % cols
        x, y = col * tile_size, row * (tile_size + header)
        
        tile = img.resize((tile_size - 4, tile_size - 4), Image.BICUBIC)
        grid.paste(tile, (x + 2, y + header))
        
        score = scores[i] if i < len(scores) else 0
        if score > 0.015:
            bg, color, icon = (80, 20, 20), (255, 100, 100), "⚠️"
        elif score < -0.015:
            bg, color, icon = (20, 60, 20), (100, 255, 100), "✓"
        else:
            bg, color, icon = (25, 25, 40), (150, 150, 150), "○"
        
        draw.rectangle([x, y, x + tile_size - 1, y + header - 1], fill=bg)
        draw.text((x + 8, y + 8), f"{icon} F{i+1}: {score:+.3f}", fill=color, font=font)
    
    return grid


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ============ Gemini Verification ============
def gemini_verify(frames: List[Dict], camera_context: str, detection_targets: str, edge_result: Dict) -> Dict:
    """Stage 2: Gemini verification."""
    print("🧠 Running Gemini verification...")
    
    grid = create_annotated_grid(frames, edge_result.get("per_frame_scores", []))
    img_b64 = image_to_b64(grid)
    
    per_frame_desc = "\n".join([
        f"  Frame {i+1}: {s:+.3f}" for i, s in enumerate(edge_result.get('per_frame_scores', [])[:16])
    ])
    
    prompt = f"""VIDEO ANOMALY VERIFICATION

CAMERA: {camera_context}
TARGETS: {detection_targets}

EDGE DETECTOR RESULT:
- Assessment: {"ANOMALY" if edge_result.get('is_anomaly') else "NORMAL"}
- Score diff: {edge_result.get('score_diff', 0):+.4f}
- Top anomaly: "{edge_result.get('top_anomaly_prompt', '')}"
- Top normal: "{edge_result.get('top_normal_prompt', '')}"

PER-FRAME SCORES:
{per_frame_desc}

Analyze the frame grid. Respond in JSON:
{{
    "is_anomaly": true/false,
    "anomaly_type": "specific type or null",
    "confidence": 0.0-1.0,
    "reasoning": "what you see",
    "key_observations": ["obs1", "obs2"]
}}"""

    response = call_gemini(prompt, img_b64, max_tokens=800)
    
    if isinstance(response, dict) and "error" in response:
        print(f"  ❌ Gemini error: {response['error']}")
        return {"error": response["error"]}
    
    parsed = extract_json_object(response)
    if parsed:
        print(f"  ✅ Gemini verdict: {'🚨 ANOMALY' if parsed.get('is_anomaly') else '✅ NORMAL'}")
        print(f"  📝 Reasoning: {parsed.get('reasoning', '')[:100]}...")
        return parsed
    
    return {"reasoning": response, "is_anomaly": "anomaly" in response.lower()}


# ============ Main Pipeline ============
def run_pipeline(video_url: str, camera_context: str, detection_targets: str) -> Dict:
    """Run the full detection pipeline."""
    print("\n" + "="*60)
    print("🎬 VIDEO SECURITY ANALYSIS PIPELINE")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "video.mp4")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)
        
        # Step 1: Download
        if not download_video(video_url, video_path):
            return {"error": "Download failed"}
        
        # Step 2: Extract frames
        frames = extract_frames(video_path, frames_dir)
        if not frames:
            return {"error": "No frames extracted"}
        
        # Step 3: Generate prompts
        prompt_banks, error = generate_prompt_banks(camera_context, detection_targets)
        if error:
            return {"error": f"Prompt generation failed: {error}"}
        
        print(f"\n📋 Sample normal prompts: {prompt_banks['normal_prompts'][:3]}")
        print(f"📋 Sample anomaly prompts: {prompt_banks['anomaly_prompts'][:3]}")
        
        # Step 4: SigLIP detection
        edge_result = siglip_detect(
            frames,
            prompt_banks["normal_prompts"],
            prompt_banks["anomaly_prompts"]
        )
        
        if "error" in edge_result:
            return {"error": edge_result["error"]}
        
        # Step 5: Gemini verification
        gemini_result = gemini_verify(frames, camera_context, detection_targets, edge_result)
        
        # Final result
        final_is_anomaly = gemini_result.get("is_anomaly", edge_result["is_anomaly"])
        
        print("\n" + "="*60)
        print(f"🎯 FINAL VERDICT: {'🚨 ANOMALY DETECTED' if final_is_anomaly else '✅ NORMAL'}")
        print("="*60)
        
        return {
            "final_verdict": {
                "is_anomaly": final_is_anomaly,
                "anomaly_type": gemini_result.get("anomaly_type"),
                "confidence": gemini_result.get("confidence", edge_result["confidence"])
            },
            "edge_detection": edge_result,
            "gemini_verification": gemini_result,
            "prompt_banks": prompt_banks
        }


# ============ Test Mode ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Security Analysis Test")
    parser.add_argument("--video_url", type=str, help="URL to video file")
    parser.add_argument("--camera_context", type=str, default="Security surveillance camera")
    parser.add_argument("--detection_targets", type=str, default="Fire, smoke, suspicious activity")
    args = parser.parse_args()
    
    # Use args or fallback to test input
    TEST_INPUT = {
        "video_url": args.video_url or "YOUR_VIDEO_URL_HERE",
        "camera_context": args.camera_context or "Industrial site surveillance camera overlooking equipment yard and machinery",
        "detection_targets": args.detection_targets or "Fire, flames, smoke, explosions, burning, safety hazards"
    }
    
    if TEST_INPUT["video_url"] == "YOUR_VIDEO_URL_HERE":
        print("❌ Please provide a video URL!")
        print("\nUsage:")
        print('  python handler_test.py --video_url "https://..." --camera_context "..." --detection_targets "..."')
        print("\nOr edit TEST_INPUT in this file directly.")
    else:
        result = run_pipeline(
            TEST_INPUT["video_url"],
            TEST_INPUT["camera_context"],
            TEST_INPUT["detection_targets"]
        )
        print("\n📦 FULL RESULT:")
        print(json.dumps(result, indent=2, default=str))
