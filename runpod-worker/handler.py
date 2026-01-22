#!/usr/bin/env python3
"""
RunPod Serverless Handler - Video Security Analysis
====================================================
Visual Context Pipeline: Uses reference frame for targeted prompt generation
"""

import os
import json
import base64
import tempfile
import subprocess
import re
from typing import List, Dict, Any, Optional
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


def image_to_b64(img: Image.Image, quality: int = 90) -> str:
    """Convert PIL Image to base64 string."""
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def extract_reference_frame(video_path: str, position: float = 0.20) -> Optional[Image.Image]:
    """
    Extract a single reference frame from the video at given position (0.0-1.0).
    Default is 20% into the video - early enough for normal scene but not the first frame.
    """
    print(f"📸 Extracting reference frame at {position:.0%} of video...")
    try:
        # Get video duration
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]).decode().strip()
        duration = float(probe)
        
        # Calculate timestamp
        timestamp = duration * position
        
        # Extract single frame to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        
        subprocess.run([
            "ffmpeg", "-ss", str(timestamp), "-i", video_path,
            "-vframes", "1", "-q:v", "2", tmp_path, "-y"
        ], capture_output=True, check=True)
        
        # Load and return as PIL Image
        img = Image.open(tmp_path).convert("RGB")
        os.unlink(tmp_path)  # Clean up temp file
        print(f"  ✅ Reference frame: {img.size[0]}x{img.size[1]} at t={timestamp:.1f}s")
        return img
    except Exception as e:
        print(f"  ⚠️ Failed to extract reference frame: {e}")
        return None


def generate_prompt_banks(camera_context: str, detection_targets: str, 
                          reference_frame: Optional[Image.Image] = None) -> tuple:
    """
    Generate normal/anomaly prompts using Gemini.
    If reference_frame is provided, Gemini analyzes the ACTUAL scene for targeted prompts.
    """
    print("🤖 Generating prompt banks...")
    print(f"   Scene: {camera_context[:50]}...")
    print(f"   Targets: {detection_targets[:50]}...")
    
    if reference_frame is not None:
        # ========== VISUAL CONTEXT MODE ==========
        print("   📸 Using reference frame for visual context")
        
        img_b64 = image_to_b64(reference_frame)
        
        prompt = f"""You are an expert in video anomaly detection. I'm showing you a REFERENCE FRAME from the actual video feed.

=== REFERENCE FRAME (ATTACHED IMAGE) ===
Study this image carefully:
- What objects, furniture, structures, or equipment are visible?
- What is the camera angle and perspective?
- What is the lighting condition?
- What area/space is being monitored?

=== USER-PROVIDED CONTEXT ===
Camera description: {camera_context}
Detection target: {detection_targets}

=== YOUR TASK ===
Generate HIGHLY SPECIFIC prompt banks for THIS EXACT scene to detect "{detection_targets}".

**ANOMALY PROMPTS (20 prompts)** - Describe what "{detection_targets}" would LOOK LIKE in THIS specific scene:
- Use the ACTUAL objects/locations visible in the reference frame
- Describe body positions, movements, interactions specific to THIS environment
- What would be visually different if "{detection_targets}" was happening HERE?
- Include before/during/after visual indicators
- Be EXTREMELY specific to what you SEE in the image

**NORMAL PROMPTS (15 prompts)** - Describe NORMAL activity in THIS exact scene:
- Reference the ACTUAL objects, furniture, and layout you see
- What does routine activity look like in THIS specific location?
- Describe people/objects in their expected positions for THIS scene
- What does THIS scene look like when NOTHING abnormal is happening?

=== FORMAT REQUIREMENTS ===
- Each prompt: 5-15 words, visually descriptive
- Reference SPECIFIC elements you see in the image (e.g., "near the counter", "by the door")
- Focus on what a SINGLE FRAME would show
- Use concrete visual descriptions, not abstract concepts

JSON ONLY (no markdown):
{{
    "normal_prompts": ["visual description using actual scene elements", ...],
    "anomaly_prompts": ["visual description of {detection_targets} in this scene", ...],
    "scene_analysis": "Brief description of what you observe in the reference frame",
    "detection_summary": "System detects {detection_targets} in this specific location"
}}"""
        
        response = call_gemini(prompt, image_b64=img_b64, max_tokens=2500)
        
    else:
        # ========== TEXT-ONLY MODE (fallback) ==========
        prompt = f"""You are an expert in video anomaly detection. Generate HIGHLY SPECIFIC prompt banks.

=== SCENE CONTEXT ===
{camera_context}

=== DETECTION TARGET (FOCUS ON THIS) ===
{detection_targets}

=== CRITICAL INSTRUCTIONS ===
Generate prompts that are LASER-FOCUSED on detecting "{detection_targets}" in a "{camera_context}" scene.

**ANOMALY PROMPTS (20 prompts)** - These MUST describe visual signs of "{detection_targets}":
- What does "{detection_targets}" LOOK LIKE in a single frame?
- Body positions, hand movements, object interactions
- Facial expressions, gestures, postures
- Suspicious object placements or movements
- Before/during/after visual indicators
- Be EXTREMELY specific to "{detection_targets}"

**NORMAL PROMPTS (15 prompts)** - Describe what NORMAL activity looks like in this scene:
- Typical behavior for "{camera_context}"
- Expected objects and their normal positions
- Regular human activities and movements
- What this scene looks like when "{detection_targets}" is NOT happening

=== FORMAT REQUIREMENTS ===
- Each prompt: 5-15 words, visually descriptive
- Focus on what a SINGLE FRAME would show
- Use concrete visual descriptions, not abstract concepts
- Anomaly prompts should make "{detection_targets}" easy to detect

JSON ONLY (no markdown):
{{
    "normal_prompts": ["visual description 1", "visual description 2", ...],
    "anomaly_prompts": ["visual description of {detection_targets} 1", ...],
    "detection_summary": "System detects {detection_targets} in {camera_context}"
}}"""
        
        response = call_gemini(prompt, max_tokens=2000)
    
    if isinstance(response, dict) and "error" in response:
        return None, f"Error: {response['error']}"
    
    parsed = extract_json_object(response)
    if parsed and "normal_prompts" in parsed and "anomaly_prompts" in parsed:
        n_normal = len(parsed['normal_prompts'])
        n_anomaly = len(parsed['anomaly_prompts'])
        print(f"  ✅ Generated {n_normal} normal, {n_anomaly} anomaly prompts")
        
        # Log scene analysis if available (from visual context mode)
        if parsed.get('scene_analysis'):
            print(f"  🔍 Scene: {parsed['scene_analysis'][:80]}...")
        
        # Log example prompts
        if parsed['anomaly_prompts']:
            print(f"  📌 Anomaly examples: {parsed['anomaly_prompts'][:2]}")
        
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
            "-show_entries", "stream=r_frame_rate,duration,nb_frames",
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


def extract_single_frame(video_path: str, timestamp: float) -> Optional[Image.Image]:
    """Extract a single frame at a specific timestamp."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        subprocess.run([
            "ffmpeg", "-ss", str(timestamp), "-i", video_path,
            "-vframes", "1", "-q:v", "2", tmp_path, "-y"
        ], capture_output=True, check=True)
        
        img = Image.open(tmp_path).convert("RGB")
        os.unlink(tmp_path)
        return img
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None


def extract_frames_uniform(video_path: str, output_dir: str, num_frames: int = 8) -> List[Dict]:
    """Extract uniformly spaced frames from entire video."""
    print(f"🎞️  Extracting {num_frames} frames uniformly...")
    
    duration, _ = get_video_info(video_path)
    
    frames = []
    for i in range(num_frames):
        timestamp = (i / (num_frames - 1)) * duration if num_frames > 1 else 0
        output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        
        try:
            subprocess.run([
                "ffmpeg", "-ss", str(timestamp), "-i", video_path,
                "-vframes", "1", "-q:v", "2", output_path, "-y"
            ], capture_output=True, check=True)
            
            if os.path.exists(output_path):
                frames.append({
                    "path": output_path,
                    "timestamp": timestamp,
                    "index": i
                })
        except Exception as e:
            print(f"  ⚠ Failed to extract frame at {timestamp:.1f}s: {e}")
    
    print(f"  ✅ Extracted {len(frames)} frames")
    return frames


def siglip_detect_batch(frames: List[Image.Image], normal_prompts: List[str], anomaly_prompts: List[str]) -> Dict:
    """Batch SigLIP2 detection - process all frames together for better accuracy."""
    model, preprocess, tokenizer = load_siglip()
    
    if not normal_prompts or not anomaly_prompts:
        return {"error": "No prompts provided"}
    
    print(f"⚡ Batch scoring {len(frames)} frames...")
    
    if USE_OPEN_CLIP:
        processed = torch.stack([preprocess(f) for f in frames]).to(DEVICE)
        
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
        score_diff = avg_anomaly - avg_normal
        
        best_anomaly_idx = (img_features @ anomaly_features.T).mean(dim=0).argmax().item()
        best_normal_idx = (img_features @ normal_features.T).mean(dim=0).argmax().item()
        
    else:
        all_normal_sims = []
        all_anomaly_sims = []
        per_frame_scores = []
        
        for frame in frames:
            inputs = preprocess(
                text=normal_prompts + anomaly_prompts,
                images=frame, return_tensors="pt", padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.sigmoid(logits).cpu().numpy()
            
            n = len(normal_prompts)
            normal_sim = float(probs[:n].max())
            anomaly_sim = float(probs[n:].max())
            
            all_normal_sims.append(normal_sim)
            all_anomaly_sims.append(anomaly_sim)
            per_frame_scores.append(anomaly_sim - normal_sim)
        
        avg_normal = np.mean(all_normal_sims)
        avg_anomaly = np.mean(all_anomaly_sims)
        score_diff = avg_anomaly - avg_normal
        
        best_anomaly_idx = 0
        best_normal_idx = 0
    
    is_anomaly = score_diff > 0
    confidence = abs(score_diff)
    
    print(f"  📊 Avg normal: {avg_normal:.4f}, Avg anomaly: {avg_anomaly:.4f}")
    print(f"  📊 Score diff: {score_diff:+.4f} -> {'ANOMALY' if is_anomaly else 'NORMAL'}")
    
    return {
        "is_anomaly": is_anomaly,
        "score_diff": score_diff,
        "anomaly_score": avg_anomaly,
        "normal_score": avg_normal,
        "confidence": confidence,
        "per_frame_scores": per_frame_scores,
        "top_anomaly_prompt": anomaly_prompts[best_anomaly_idx],
        "top_normal_prompt": normal_prompts[best_normal_idx],
    }


# ============ Hysteresis-Based Detection ============
def precompute_text_features(normal_prompts: List[str], anomaly_prompts: List[str]) -> tuple:
    """Pre-compute text features for efficiency in sequential processing."""
    model, preprocess, tokenizer = load_siglip()
    
    if not USE_OPEN_CLIP:
        return None, None
    
    with torch.no_grad():
        normal_tokens = tokenizer(normal_prompts).to(DEVICE)
        normal_features = model.encode_text(normal_tokens)
        normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
        
        anomaly_tokens = tokenizer(anomaly_prompts).to(DEVICE)
        anomaly_features = model.encode_text(anomaly_tokens)
        anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
    
    return normal_features, anomaly_features


def siglip_score_single_frame(
    frame: Image.Image, 
    normal_prompts: List[str], 
    anomaly_prompts: List[str],
    normal_features: torch.Tensor = None,
    anomaly_features: torch.Tensor = None
) -> tuple:
    """Score a single frame against prompts. Returns (score_diff, normal_sim, anomaly_sim)."""
    model, preprocess, tokenizer = load_siglip()
    
    if USE_OPEN_CLIP:
        processed = preprocess(frame).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            img_features = model.encode_image(processed)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            if normal_features is None:
                normal_tokens = tokenizer(normal_prompts).to(DEVICE)
                normal_features = model.encode_text(normal_tokens)
                normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
            
            if anomaly_features is None:
                anomaly_tokens = tokenizer(anomaly_prompts).to(DEVICE)
                anomaly_features = model.encode_text(anomaly_tokens)
                anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
        
        normal_sim = (img_features @ normal_features.T).max(dim=1).values.item()
        anomaly_sim = (img_features @ anomaly_features.T).max(dim=1).values.item()
        score_diff = anomaly_sim - normal_sim
        
        return score_diff, normal_sim, anomaly_sim
    else:
        inputs = preprocess(
            text=normal_prompts + anomaly_prompts,
            images=frame, return_tensors="pt", padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.sigmoid(logits).cpu().numpy()
        
        n = len(normal_prompts)
        normal_sim = float(probs[:n].max())
        anomaly_sim = float(probs[n:].max())
        score_diff = anomaly_sim - normal_sim
        
        return score_diff, normal_sim, anomaly_sim


def process_video_with_hysteresis(
    video_path: str,
    normal_prompts: List[str],
    anomaly_prompts: List[str],
    sample_fps: float = 2.0,
    high_threshold: float = 0.02,
    low_threshold: float = -0.01,
    min_frames_to_trigger: int = 3,
    min_frames_to_clear: int = 5,
) -> Dict:
    """
    Process video sequentially with hysteresis-based anomaly detection.
    
    Uses a state machine:
    - NORMAL state: transitions to ANOMALY when score > high_threshold for N consecutive frames
    - ANOMALY state: transitions to NORMAL when score < low_threshold for M consecutive frames
    """
    print(f"\n🔄 Processing video with hysteresis (sample_fps={sample_fps})...")
    print(f"   Thresholds: HIGH={high_threshold}, LOW={low_threshold}")
    print(f"   Trigger: {min_frames_to_trigger} frames, Clear: {min_frames_to_clear} frames")
    
    duration, video_fps = get_video_info(video_path)
    total_samples = int(duration * sample_fps)
    print(f"   Video: {duration:.1f}s @ {video_fps:.1f}fps → {total_samples} samples")
    
    print("   Pre-computing text features...")
    normal_features, anomaly_features = precompute_text_features(normal_prompts, anomaly_prompts)
    
    state = "NORMAL"
    consecutive_high = 0
    consecutive_low = 0
    
    events = []
    current_event_start = None
    current_event_peak = 0
    current_event_scores = []
    
    all_scores = []
    
    print(f"   Processing {total_samples} frames...")
    
    for i in range(total_samples):
        timestamp = i / sample_fps
        
        if i % 10 == 0:
            progress = (i / total_samples) * 100
            print(f"\r   ⏳ Progress: {progress:.0f}% ({i}/{total_samples}) | State: {state}", end="", flush=True)
        
        frame = extract_single_frame(video_path, timestamp)
        if frame is None:
            continue
        
        score, normal_sim, anomaly_sim = siglip_score_single_frame(
            frame, normal_prompts, anomaly_prompts, 
            normal_features, anomaly_features
        )
        
        all_scores.append({
            "timestamp": timestamp,
            "score": score,
            "normal_sim": normal_sim,
            "anomaly_sim": anomaly_sim,
            "state": state
        })
        
        if state == "NORMAL":
            if score > high_threshold:
                consecutive_high += 1
                consecutive_low = 0
                if consecutive_high >= min_frames_to_trigger:
                    state = "ANOMALY"
                    current_event_start = timestamp - ((consecutive_high - 1) / sample_fps)
                    current_event_peak = score
                    current_event_scores = [score]
                    print(f"\n   🚨 ANOMALY STARTED at {current_event_start:.1f}s (score: {score:.4f})")
            else:
                consecutive_high = 0
                
        elif state == "ANOMALY":
            current_event_peak = max(current_event_peak, score)
            current_event_scores.append(score)
            
            if score < low_threshold:
                consecutive_low += 1
                consecutive_high = 0
                if consecutive_low >= min_frames_to_clear:
                    event_end = timestamp - ((consecutive_low - 1) / sample_fps)
                    events.append({
                        "start": current_event_start,
                        "end": event_end,
                        "duration": event_end - current_event_start,
                        "peak_score": current_event_peak,
                        "avg_score": float(np.mean(current_event_scores)),
                    })
                    print(f"\n   ✅ ANOMALY ENDED at {event_end:.1f}s (duration: {event_end - current_event_start:.1f}s)")
                    state = "NORMAL"
                    current_event_start = None
                    current_event_scores = []
            else:
                consecutive_low = 0
    
    print(f"\r   ⏳ Progress: 100% ({total_samples}/{total_samples}) | Complete!        ")
    
    if state == "ANOMALY" and current_event_start is not None:
        events.append({
            "start": current_event_start,
            "end": duration,
            "duration": duration - current_event_start,
            "peak_score": current_event_peak,
            "avg_score": float(np.mean(current_event_scores)) if current_event_scores else current_event_peak,
        })
        print(f"   ⚠️ Video ended during anomaly (started at {current_event_start:.1f}s)")
    
    total_anomaly_duration = sum(e["duration"] for e in events)
    is_anomaly = len(events) > 0
    
    print(f"\n📊 Hysteresis Results:")
    print(f"   Total events: {len(events)}")
    print(f"   Total anomaly duration: {total_anomaly_duration:.1f}s / {duration:.1f}s ({100*total_anomaly_duration/duration:.1f}%)")
    print(f"   Final verdict: {'🚨 ANOMALY' if is_anomaly else '✅ NORMAL'}")
    
    return {
        "is_anomaly": is_anomaly,
        "events": events,
        "all_scores": all_scores,
        "total_anomaly_duration": total_anomaly_duration,
        "video_duration": duration,
        "anomaly_percentage": 100 * total_anomaly_duration / duration if duration > 0 else 0,
        "num_events": len(events),
        "peak_score": max([e["peak_score"] for e in events]) if events else 0,
    }


def create_annotated_grid(frames: List[Dict], scores: List[float], cols=4, tile_size=256) -> Image.Image:
    """Create annotated frame grid for UI display."""
    images = [Image.open(f["path"]).convert("RGB") for f in frames[:16]]
    n = len(images)
    rows = (n + cols - 1) // cols
    header = 32
    
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
        timestamp = frames[i].get("timestamp", 0) if i < len(frames) else 0
        
        if score > 0.015:
            bg, color = (60, 20, 20), (255, 100, 100)
            label = f"⚠️ {timestamp:.1f}s: {score:+.3f}"
        elif score < -0.015:
            bg, color = (20, 50, 20), (100, 255, 100)
            label = f"✓ {timestamp:.1f}s: {score:+.3f}"
        else:
            bg, color = (30, 30, 40), (180, 180, 180)
            label = f"○ {timestamp:.1f}s: {score:+.3f}"
        
        draw.rectangle([x, y, x + tile_size - 1, y + header - 1], fill=bg)
        draw.text((x + 6, y + 8), label, fill=color, font=font)
    
    return grid


def create_gemini_grid(frames: List[Dict], cols=4) -> Image.Image:
    """Create CLEAN frame grid for Gemini (no annotations)."""
    images = [Image.open(f["path"]).convert("RGB") for f in frames[:8]]
    n = len(images)
    rows = (n + cols - 1) // cols
    tile_size = 128
    
    grid = Image.new('RGB', (cols * tile_size, rows * tile_size), (0, 0, 0))
    
    for i, img in enumerate(images):
        row, col = i // cols, i % cols
        x, y = col * tile_size, row * tile_size
        tile = img.resize((tile_size, tile_size), Image.BICUBIC)
        grid.paste(tile, (x, y))
    
    return grid


def gemini_verify(frames: List[Dict], camera_context: str, detection_targets: str, 
                  edge_result: Dict) -> Dict:
    """Gemini verification - HYPERFOCUSED on the specific detection target."""
    print("🧠 Running Gemini verification...")
    print(f"   Looking for: {detection_targets}")
    
    grid = create_gemini_grid(frames)
    grid_b64 = image_to_b64(grid)
    
    prompt = f"""=== FOCUSED VIDEO ANALYSIS ===

**YOUR ONE JOB:** Determine if "{detection_targets}" is happening in these frames.

SCENE: {camera_context}
TARGET TO DETECT: {detection_targets}

EDGE DETECTOR HINT:
- Result: {"ANOMALY LIKELY" if edge_result.get('is_anomaly') else "NORMAL LIKELY"}
- Matched: "{edge_result.get('top_anomaly_prompt', 'N/A')}"

=== ANALYZE THE 8-FRAME SEQUENCE ===

Look at each frame carefully. Ask yourself:
1. Do I see any visual evidence of "{detection_targets}"?
2. What specific actions, objects, or behaviors indicate "{detection_targets}"?
3. What would this scene look like if "{detection_targets}" was NOT happening?

BE SPECIFIC: Describe exactly what you see that indicates "{detection_targets}" OR why this is normal activity.

JSON ONLY:
{{
    "isAnomaly": true/false,
    "anomalyType": "{detection_targets}" or null,
    "confidence": 0.0-1.0,
    "reasoning": "I see [specific visual evidence]. In frame X, [describe what you observe]. This indicates [detection_targets] because [explanation]."
}}"""

    response = call_gemini(prompt, grid_b64, max_tokens=800)
    
    if isinstance(response, dict) and "error" in response:
        return {"error": response["error"]}
    
    parsed = extract_json_object(response)
    if parsed:
        print(f"  ✅ Gemini: {'ANOMALY' if parsed.get('isAnomaly') else 'NORMAL'} (confidence: {parsed.get('confidence', 0):.0%})")
        return {
            "isAnomaly": parsed.get("isAnomaly", False),
            "anomalyType": parsed.get("anomalyType"),
            "confidence": parsed.get("confidence", 0),
            "reasoning": parsed.get("reasoning", ""),
        }
    
    return {
        "reasoning": response, 
        "isAnomaly": "anomaly" in response.lower() and "no anomaly" not in response.lower()
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Visual Context Pipeline with two modes:
    
    MODE: "batch" (default) - Fast uniform sampling
    1. Download video
    2. Extract reference frame at 20% for visual context
    3. Generate prompts using Gemini with the actual scene image
    4. Extract N uniform frames and run batch SigLIP detection
    5. Gemini verification
    
    MODE: "hysteresis" - Dense sequential processing with event detection
    1. Download video
    2. Extract reference frame at 20% for visual context  
    3. Generate prompts using Gemini with the actual scene image
    4. Process video at sample_fps with hysteresis state machine
    5. Return detected events with precise timestamps
    """
    job_input = job.get("input", {})
    
    video_url = job_input.get("video_url")
    camera_context = job_input.get("camera_context", "Security camera")
    detection_targets = job_input.get("detection_targets", "Suspicious activity")
    mode = job_input.get("mode", "hysteresis")  # "hysteresis" or "batch"
    
    # Batch mode params
    num_frames = job_input.get("num_frames", 8)
    
    # Hysteresis mode params - SYMMETRIC AT ZERO
    sample_fps = job_input.get("sample_fps", 2.0)
    high_threshold = job_input.get("high_threshold", 0)      # ANOMALY when score > 0
    low_threshold = job_input.get("low_threshold", 0)        # NORMAL when score < 0
    min_frames_to_trigger = job_input.get("min_frames_to_trigger", 2)  # 2 consecutive frames to trigger
    min_frames_to_clear = job_input.get("min_frames_to_clear", 4)      # 4 consecutive frames to clear
    
    if not video_url:
        return {"error": "video_url is required"}
    
    print("\n" + "="*60)
    print(f"🎬 Video Anomaly Detection - {mode.upper()} Mode")
    print(f"   Scene: {camera_context[:40]}...")
    print(f"   Target: {detection_targets[:40]}...")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "video.mp4")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)
        
        # Step 1: Download video
        if not download_video(video_url, video_path):
            return {"error": "Download failed"}
        
        # Step 2: Get video info
        duration, video_fps = get_video_info(video_path)
        print(f"📹 Video: {duration:.1f}s @ {video_fps:.1f}fps")
        
        # Step 3: Extract reference frame at 20% for visual context
        reference_frame = extract_reference_frame(video_path, position=0.20)
        
        # Step 4: Generate prompts WITH visual context
        prompt_banks, error = generate_prompt_banks(
            camera_context, 
            detection_targets, 
            reference_frame
        )
        if error:
            return {"error": f"Prompt generation failed: {error}"}
        
        normal_prompts = prompt_banks["normal_prompts"]
        anomaly_prompts = prompt_banks["anomaly_prompts"]
        detection_summary = prompt_banks.get("detection_summary", "")
        scene_analysis = prompt_banks.get("scene_analysis", "")
        
        # ========== MODE: HYSTERESIS ==========
        if mode == "hysteresis":
            hysteresis_result = process_video_with_hysteresis(
                video_path,
                normal_prompts,
                anomaly_prompts,
                sample_fps=sample_fps,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
                min_frames_to_trigger=min_frames_to_trigger,
                min_frames_to_clear=min_frames_to_clear,
            )
            
            if "error" in hysteresis_result:
                return {"error": hysteresis_result["error"]}
            
            final_is_anomaly = hysteresis_result["is_anomaly"]
            
            print("\n" + "="*60)
            print(f"🎯 FINAL VERDICT: {'🚨 ANOMALY DETECTED' if final_is_anomaly else '✅ NORMAL'}")
            if final_is_anomaly:
                print(f"   Events: {hysteresis_result['num_events']}")
                print(f"   Duration: {hysteresis_result['total_anomaly_duration']:.1f}s ({hysteresis_result['anomaly_percentage']:.1f}%)")
            print("="*60)
            
            # Condensed score summary (all_scores can be huge)
            score_summary = {
                "num_samples": len(hysteresis_result["all_scores"]),
                "min_score": min(s["score"] for s in hysteresis_result["all_scores"]) if hysteresis_result["all_scores"] else 0,
                "max_score": max(s["score"] for s in hysteresis_result["all_scores"]) if hysteresis_result["all_scores"] else 0,
                "avg_score": float(np.mean([s["score"] for s in hysteresis_result["all_scores"]])) if hysteresis_result["all_scores"] else 0,
            }
            
            return {
                "status": "completed",
                "mode": "hysteresis",
                "videoDuration": duration,
                
                "finalVerdict": {
                    "isAnomaly": final_is_anomaly,
                    "numEvents": hysteresis_result["num_events"],
                    "totalAnomalyDuration": round(hysteresis_result["total_anomaly_duration"], 2),
                    "anomalyPercentage": round(hysteresis_result["anomaly_percentage"], 2),
                    "peakScore": round(hysteresis_result["peak_score"], 4),
                },
                
                "events": [
                    {
                        "start": round(e["start"], 2),
                        "end": round(e["end"], 2),
                        "duration": round(e["duration"], 2),
                        "peakScore": round(e["peak_score"], 4),
                        "avgScore": round(e["avg_score"], 4),
                    }
                    for e in hysteresis_result["events"]
                ],
                
                "scoreSummary": score_summary,
                
                "promptBanks": {
                    "normalPrompts": normal_prompts,
                    "anomalyPrompts": anomaly_prompts,
                    "detectionSummary": detection_summary,
                    "sceneAnalysis": scene_analysis,
                },
                
                "hysteresisParams": {
                    "sampleFps": sample_fps,
                    "highThreshold": high_threshold,
                    "lowThreshold": low_threshold,
                    "minFramesToTrigger": min_frames_to_trigger,
                    "minFramesToClear": min_frames_to_clear,
                }
            }
        
        # ========== MODE: BATCH (default) ==========
        else:
            # Step 5: Extract analysis frames
            frames = extract_frames_uniform(video_path, frames_dir, num_frames)
            
            if len(frames) < 2:
                return {"error": f"Only extracted {len(frames)} frames, need at least 2"}
            
            # Load frame images
            frame_images = []
            for f in frames:
                try:
                    img = Image.open(f["path"]).convert("RGB")
                    frame_images.append(img)
                except Exception as e:
                    print(f"  ⚠ Failed to load {f['path']}: {e}")
            
            if len(frame_images) < 2:
                return {"error": "Failed to load enough frame images"}
            
            # Step 6: SigLIP detection
            print("\n⚡ Stage 1: SigLIP2 Batch Detection")
            edge_result = siglip_detect_batch(frame_images, normal_prompts, anomaly_prompts)
            
            if "error" in edge_result:
                return {"error": f"Edge detection failed: {edge_result['error']}"}
            
            # Step 7: Gemini verification
            print("\n🧠 Stage 2: Gemini Verification")
            gemini_result = gemini_verify(frames, camera_context, detection_targets, edge_result)
            
            # Create annotated grid for response
            annotated_grid = create_annotated_grid(frames, edge_result.get("per_frame_scores", []))
            grid_b64 = image_to_b64(annotated_grid)
            
            # Determine final verdict
            gemini_confidence = gemini_result.get("confidence", 0) if isinstance(gemini_result, dict) else 0
            
            if gemini_confidence >= 0.7:
                final_is_anomaly = gemini_result.get("isAnomaly", edge_result["is_anomaly"])
                final_confidence = gemini_confidence
            else:
                final_is_anomaly = edge_result["is_anomaly"]
                final_confidence = edge_result["confidence"]
            
            print("\n" + "="*60)
            print(f"🎯 FINAL VERDICT: {'🚨 ANOMALY DETECTED' if final_is_anomaly else '✅ NORMAL'}")
            print(f"   Confidence: {final_confidence:.1%}")
            print("="*60)
            
            return {
                "status": "completed",
                "mode": "batch",
                "videoDuration": duration,
                
                "edgeDetection": {
                    "isAnomaly": edge_result["is_anomaly"],
                    "scoreDiff": round(edge_result["score_diff"], 4),
                    "anomalyScore": round(edge_result["anomaly_score"], 4),
                    "normalScore": round(edge_result["normal_score"], 4),
                    "confidence": round(edge_result["confidence"], 4),
                    "perFrameScores": [round(s, 4) for s in edge_result["per_frame_scores"]],
                    "topAnomalyPrompt": edge_result["top_anomaly_prompt"],
                    "topNormalPrompt": edge_result["top_normal_prompt"],
                },
                
                "geminiVerification": gemini_result,
                
                "finalVerdict": {
                    "isAnomaly": final_is_anomaly,
                    "anomalyType": gemini_result.get("anomalyType") if final_is_anomaly else None,
                    "confidence": round(final_confidence, 4),
                },
                
                "promptBanks": {
                    "normalPrompts": normal_prompts,
                    "anomalyPrompts": anomaly_prompts,
                    "detectionSummary": detection_summary,
                    "sceneAnalysis": scene_analysis,
                },
                
                "annotatedGridB64": f"data:image/jpeg;base64,{grid_b64}",
                "frameCount": len(frames),
                "frameTimestamps": [f["timestamp"] for f in frames],
            }


runpod.serverless.start({"handler": handler})
