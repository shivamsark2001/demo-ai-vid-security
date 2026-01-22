#!/usr/bin/env python3
"""
RunPod Serverless Handler - Video Security Analysis
====================================================
Sequential Processing: Analyze frames until first anomaly detected
"""

import os
import json
import uuid
import base64
import tempfile
import subprocess
import re
from typing import List, Dict, Any
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


def load_siglip():
    """Load SigLIP2 model."""
    global SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER, USE_OPEN_CLIP
    
    if SIGLIP_MODEL is not None:
        return SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER
    
    try:
        import open_clip
        print("📦 Loading SigLIP2...")
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
        print(f"✅ SigLIP loaded on {DEVICE}")
    
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
    print("🤖 Generating prompt banks...")
    
    prompt = f"""Generate prompt banks for video anomaly detection.

CAMERA: {camera_context}
DETECT: {detection_targets}

Generate text prompts for image matching:
1. NORMAL: 10 descriptions of normal behavior
2. ANOMALY: 10 descriptions of abnormal behavior

Keep them general so that they can be used for many videos in the context of the detection targets.

JSON only:
{{
    "normal_prompts": ["desc1", ...],
    "anomaly_prompts": ["desc1", ...],
}}"""

    response = call_gemini(prompt, max_tokens=1200)
    
    if isinstance(response, dict) and "error" in response:
        return None, f"Error: {response['error']}"
    
    parsed = extract_json_object(response)
    if parsed and "normal_prompts" in parsed and "anomaly_prompts" in parsed:
        print(f"  ✅ Generated prompts")
        return parsed, None
    
    return None, f"Parse failed"


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


def get_video_info(video_path: str) -> tuple:
    """Get video duration and fps."""
    try:
        result = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json", video_path
        ]).decode()
        info = json.loads(result)
        
        duration = float(info.get("format", {}).get("duration", 60))
        fps_str = info.get("streams", [{}])[0].get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30
        
        return duration, fps
    except:
        return 60, 30


def extract_frame_at_time(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame at specific timestamp."""
    try:
        subprocess.run([
            "ffmpeg", "-ss", str(timestamp), "-i", video_path,
            "-vframes", "1", "-q:v", "2", output_path, "-y"
        ], capture_output=True, check=True)
        return os.path.exists(output_path)
    except:
        return False


def extract_frames_range(video_path: str, start_time: float, end_time: float, output_dir: str, fps: float = 2) -> List[Dict]:
    """Extract frames from a time range."""
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    duration = end_time - start_time
    
    subprocess.run([
        "ffmpeg", "-ss", str(start_time), "-i", video_path,
        "-t", str(duration), "-vf", f"fps={fps}",
        "-q:v", "2", output_pattern, "-y"
    ], capture_output=True)
    
    frames = []
    for i, f in enumerate(sorted(Path(output_dir).glob("frame_*.jpg"))):
        frames.append({
            "path": str(f),
            "timestamp": start_time + (i / fps),
            "index": i
        })
    return frames


def score_frame(image: Image.Image, normal_prompts: List[str], anomaly_prompts: List[str]) -> Dict:
    """Score a single frame against prompts."""
    model, preprocess, tokenizer = load_siglip()
    
    if USE_OPEN_CLIP:
        processed = preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            img_features = model.encode_image(processed)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            normal_tokens = tokenizer(normal_prompts).to(DEVICE)
            anomaly_tokens = tokenizer(anomaly_prompts).to(DEVICE)
            
            normal_features = model.encode_text(normal_tokens)
            normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
            
            anomaly_features = model.encode_text(anomaly_tokens)
            anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
        
        normal_sim = (img_features @ normal_features.T).max().item()
        anomaly_sim = (img_features @ anomaly_features.T).max().item()
        
        best_anomaly_idx = (img_features @ anomaly_features.T).argmax().item()
        best_normal_idx = (img_features @ normal_features.T).argmax().item()
    else:
        inputs = preprocess(
            text=normal_prompts + anomaly_prompts,
            images=image, return_tensors="pt", padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.sigmoid(logits).cpu().numpy()
        
        n = len(normal_prompts)
        normal_sim = float(probs[:n].max())
        anomaly_sim = float(probs[n:].max())
        best_anomaly_idx = int(probs[n:].argmax())
        best_normal_idx = int(probs[:n].argmax())
    
    score_diff = anomaly_sim - normal_sim
    
    return {
        "score_diff": score_diff,
        "is_anomaly": score_diff > 0,
        "anomaly_sim": anomaly_sim,
        "normal_sim": normal_sim,
        "top_anomaly_prompt": anomaly_prompts[best_anomaly_idx],
        "top_normal_prompt": normal_prompts[best_normal_idx],
    }


def create_annotated_grid(frames: List[Dict], scores: List[float], cols=4, tile_size=220) -> Image.Image:
    """Create annotated frame grid."""
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


def gemini_verify(frames: List[Dict], camera_context: str, detection_targets: str, 
                  scores: List[float], top_anomaly: str, top_normal: str) -> Dict:
    """Gemini verification with frame grid."""
    print("🧠 Running Gemini verification...")
    
    grid = create_annotated_grid(frames, scores)
    grid_b64 = image_to_b64(grid)
    
    avg_score = np.mean(scores)
    max_score = max(scores)
    
    prompt = f"""VIDEO ANOMALY VERIFICATION

CAMERA: {camera_context}
TARGETS: {detection_targets}
EDGE DETECTOR RESULT:
- Avg score: {avg_score:+.4f}
- Max score: {max_score:+.4f}
- Top anomaly match: "{top_anomaly}"
- Top normal match: "{top_normal}"

Analyze the frame grid. Are these frames showing an anomaly?

JSON only:
{{
    "isAnomaly": true/false,
    "anomalyType": "specific type or null",
    "confidence": 0.0-1.0,
    "reasoning": "detailed analysis",
    "keyObservations": ["obs1", "obs2", "obs3"],
    "frameAnalysis": "which frames show the anomaly"
}}"""

    response = call_gemini(prompt, grid_b64, max_tokens=800)
    
    if isinstance(response, dict) and "error" in response:
        return {"error": response["error"]}
    
    parsed = extract_json_object(response)
    if parsed:
        print(f"  ✅ Gemini: {'ANOMALY' if parsed.get('isAnomaly') else 'NORMAL'}")
        return {
            "isAnomaly": parsed.get("isAnomaly", False),
            "anomalyType": parsed.get("anomalyType"),
            "confidence": parsed.get("confidence", 0),
            "reasoning": parsed.get("reasoning", ""),
            "keyObservations": parsed.get("keyObservations", []),
            "frameAnalysis": parsed.get("frameAnalysis", "")
        }
    
    return {"reasoning": response, "isAnomaly": "anomaly" in response.lower()}


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sequential processing: scan video until first anomaly detected.
    """
    job_input = job.get("input", {})
    
    video_url = job_input.get("video_url")
    camera_context = job_input.get("camera_context", "Security camera")
    detection_targets = job_input.get("detection_targets", "Suspicious activity")
    
    # Processing params
    scan_fps = job_input.get("scan_fps", 1)  # 1 FPS scanning
    window_size = job_input.get("window_size", 8)  # Frames to analyze when anomaly found
    anomaly_threshold = job_input.get("anomaly_threshold", 0.01)  # Score threshold
    
    if not video_url:
        return {"error": "video_url is required"}
    
    print("\n" + "="*50)
    print("🎬 TRUWO - Sequential Analysis")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "video.mp4")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)
        
        # Download video
        if not download_video(video_url, video_path):
            return {"error": "Download failed"}
        
        # Get video info
        duration, video_fps = get_video_info(video_path)
        print(f"📹 Video: {duration:.1f}s @ {video_fps:.1f}fps")
        
        # Generate prompts
        prompt_banks, error = generate_prompt_banks(camera_context, detection_targets)
        if error:
            return {"error": f"Prompt generation failed: {error}"}
        
        normal_prompts = prompt_banks["normal_prompts"]
        anomaly_prompts = prompt_banks["anomaly_prompts"]
        
        # Sequential scan
        print(f"🔍 Scanning at {scan_fps} fps...")
        
        anomaly_found = False
        anomaly_time = None
        best_score = -1
        best_anomaly_prompt = ""
        best_normal_prompt = ""
        
        scan_interval = 1.0 / scan_fps
        current_time = 0
        frames_scanned = 0
        
        while current_time < duration:
            frame_path = os.path.join(temp_dir, f"scan_{frames_scanned:04d}.jpg")
            
            if extract_frame_at_time(video_path, current_time, frame_path):
                try:
                    img = Image.open(frame_path).convert("RGB")
                    result = score_frame(img, normal_prompts, anomaly_prompts)
                    
                    frames_scanned += 1
                    
                    if result["score_diff"] > best_score:
                        best_score = result["score_diff"]
                        best_anomaly_prompt = result["top_anomaly_prompt"]
                        best_normal_prompt = result["top_normal_prompt"]
                    
                    # Check if anomaly threshold exceeded
                    if result["score_diff"] > anomaly_threshold:
                        print(f"  🚨 Anomaly at {current_time:.1f}s (score: {result['score_diff']:+.4f})")
                        anomaly_found = True
                        anomaly_time = current_time
                        break
                    
                    if frames_scanned % 10 == 0:
                        print(f"  ✓ Scanned {frames_scanned} frames, t={current_time:.1f}s")
                    
                except Exception as e:
                    print(f"  ⚠ Frame error at {current_time:.1f}s: {e}")
            
            current_time += scan_interval
        
        print(f"📊 Scanned {frames_scanned} frames total")
        
        # If anomaly found, extract window and verify with Gemini
        if anomaly_found:
            print(f"🎯 Extracting frames around anomaly at {anomaly_time:.1f}s...")
            
            # Extract frames around anomaly
            start_time = max(0, anomaly_time - 2)
            end_time = min(duration, anomaly_time + 6)
            
            window_frames = extract_frames_range(video_path, start_time, end_time, frames_dir, fps=1)
            
            if window_frames:
                # Score all window frames
                window_scores = []
                for f in window_frames:
                    try:
                        img = Image.open(f["path"]).convert("RGB")
                        result = score_frame(img, normal_prompts, anomaly_prompts)
                        window_scores.append(result["score_diff"])
                    except:
                        window_scores.append(0)
                
                # Gemini verification
                gemini_result = gemini_verify(
                    window_frames, camera_context, detection_targets,
                    window_scores, best_anomaly_prompt, best_normal_prompt
                )
                
                # Create annotated grid
                grid = create_annotated_grid(window_frames, window_scores)
                grid_b64 = image_to_b64(grid)
                
                # Build response
                return {
                    "status": "completed",
                    "videoDuration": duration,
                    "events": [],
                    
                    "edgeDetection": {
                        "isAnomaly": True,
                        "scoreDiff": round(best_score, 4),
                        "anomalyScore": 0,
                        "normalScore": 0,
                        "confidence": round(best_score, 4),
                        "perFrameScores": [round(s, 4) for s in window_scores],
                        "topAnomalyPrompt": best_anomaly_prompt,
                        "topNormalPrompt": best_normal_prompt,
                    },
                    "geminiVerification": gemini_result,
                    "annotatedGridB64": f"data:image/jpeg;base64,{grid_b64}",
                    "frameCount": len(window_frames),
                    "anomalyTimestamp": anomaly_time,
                    "framesScanned": frames_scanned,
                }
        
        # No anomaly found
        print("✅ No anomaly detected")
        return {
            "status": "completed",
            "videoDuration": duration,
            "events": [],
            
            "edgeDetection": {
                "isAnomaly": False,
                "scoreDiff": round(best_score, 4),
                "anomalyScore": 0,
                "normalScore": 0,
                "confidence": 0,
                "perFrameScores": [],
                "topAnomalyPrompt": best_anomaly_prompt,
                "topNormalPrompt": best_normal_prompt,
            },
            "geminiVerification": {
                "isAnomaly": False,
                "anomalyType": None,
                "confidence": 0.95,
                "reasoning": f"Scanned {frames_scanned} frames across {duration:.1f}s video. No anomalies exceeding threshold detected. Video appears normal.",
                "keyObservations": [
                    f"Analyzed {frames_scanned} frames at {scan_fps} fps",
                    f"Maximum anomaly score: {best_score:+.4f}",
                    "No frames exceeded anomaly threshold"
                ],
                "frameAnalysis": ""
            },
            "annotatedGridB64": None,
            "frameCount": frames_scanned,
            "framesScanned": frames_scanned,
        }


runpod.serverless.start({"handler": handler})
