#!/usr/bin/env python3
"""
RunPod Serverless Handler - Video Security Analysis
====================================================
Batch Processing Pipeline: More accurate aggregate scoring across all frames
"""

import os
import json
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
    """Generate normal/anomaly prompts using Gemini - HYPERFOCUSED on detection targets."""
    print("🤖 Generating prompt banks...")
    print(f"   Scene: {camera_context[:50]}...")
    print(f"   Targets: {detection_targets[:50]}...")
    
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
        
        # Log a few example prompts for debugging
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
    """
    Batch SigLIP2 detection - process all frames together for better accuracy.
    This is more accurate than sequential scoring because:
    1. Batch normalization context across frames
    2. Aggregate scoring is more robust to outliers
    3. All frames contribute to the final decision
    """
    model, preprocess, tokenizer = load_siglip()
    
    if not normal_prompts or not anomaly_prompts:
        return {"error": "No prompts provided"}
    
    print(f"⚡ Batch scoring {len(frames)} frames...")
    
    if USE_OPEN_CLIP:
        # Encode ALL images at once (batch processing)
        processed = torch.stack([preprocess(f) for f in frames]).to(DEVICE)
        
        with torch.no_grad():
            # Batch encode images
            img_features = model.encode_image(processed)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            # Encode prompts (only once, reused for all frames)
            normal_tokens = tokenizer(normal_prompts).to(DEVICE)
            anomaly_tokens = tokenizer(anomaly_prompts).to(DEVICE)
            
            normal_features = model.encode_text(normal_tokens)
            normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
            
            anomaly_features = model.encode_text(anomaly_tokens)
            anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
        
        # Per-frame similarities (vectorized - much faster)
        # Shape: [num_frames, num_prompts] -> max per frame -> [num_frames]
        normal_sim = (img_features @ normal_features.T).max(dim=1).values
        anomaly_sim = (img_features @ anomaly_features.T).max(dim=1).values
        
        # Per-frame score differences
        per_frame_scores = (anomaly_sim - normal_sim).cpu().numpy().tolist()
        
        # Aggregate scores across ALL frames (key difference from hysteresis)
        avg_normal = normal_sim.mean().item()
        avg_anomaly = anomaly_sim.mean().item()
        score_diff = avg_anomaly - avg_normal
        
        # Best matching prompts (averaged across frames)
        best_anomaly_idx = (img_features @ anomaly_features.T).mean(dim=0).argmax().item()
        best_normal_idx = (img_features @ normal_features.T).mean(dim=0).argmax().item()
        
    else:
        # HuggingFace transformers fallback (batch processing)
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
        
        # Simple best prompt selection for fallback
        best_anomaly_idx = 0
        best_normal_idx = 0
    
    # Determine anomaly based on aggregate score
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
    """
    Create CLEAN frame grid for Gemini (no annotations).
    Simpler grid lets VLM focus on visual content without bias from scores.
    """
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


def image_to_b64(img: Image.Image, quality: int = 90) -> str:
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def parse_anomaly_types(detection_targets: str) -> List[str]:
    """Parse detection targets into structured anomaly types."""
    types = []
    for delim in [',', ';', '\n', '•', '-']:
        if delim in detection_targets:
            types = [t.strip() for t in detection_targets.split(delim) if t.strip()]
            break
    
    if not types:
        types = [detection_targets.strip()]
    
    return types


def gemini_verify(frames: List[Dict], camera_context: str, detection_targets: str, 
                  edge_result: Dict) -> Dict:
    """
    Gemini verification - HYPERFOCUSED on the specific detection target.
    """
    print("🧠 Running Gemini verification...")
    print(f"   Looking for: {detection_targets}")
    
    # Create CLEAN grid (no annotations - let Gemini see raw frames)
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
    
    # Fallback parsing
    return {
        "reasoning": response, 
        "isAnomaly": "anomaly" in response.lower() and "no anomaly" not in response.lower()
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch Processing Pipeline:
    - Extracts uniformly sampled frames from entire video
    - Processes ALL frames together for aggregate scoring
    - More accurate than sequential/hysteresis approach
    """
    job_input = job.get("input", {})
    
    video_url = job_input.get("video_url")
    camera_context = job_input.get("camera_context", "Security camera")
    detection_targets = job_input.get("detection_targets", "Suspicious activity")
    
    # Processing params
    num_frames = job_input.get("num_frames", 8)  # Number of frames to sample
    
    if not video_url:
        return {"error": "video_url is required"}
    
    print("\n" + "="*60)
    print("🎬 Video Anomaly Detection - Batch Processing Pipeline")
    print(f"   Sampling {num_frames} frames for aggregate analysis")
    print("="*60)
    
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
        detection_summary = prompt_banks.get("detection_summary", "")
        
        # Extract uniformly spaced frames
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
        
        # Stage 1: Batch SigLIP detection
        print("\n⚡ Stage 1: SigLIP2 Batch Detection")
        edge_result = siglip_detect_batch(frame_images, normal_prompts, anomaly_prompts)
        
        if "error" in edge_result:
            return {"error": f"Edge detection failed: {edge_result['error']}"}
        
        # Stage 2: Gemini verification (always run for accuracy)
        print("\n🧠 Stage 2: Gemini Verification")
        gemini_result = gemini_verify(frames, camera_context, detection_targets, edge_result)
        
        # Create annotated grid for response (with scores for UI display)
        annotated_grid = create_annotated_grid(frames, edge_result.get("per_frame_scores", []))
        grid_b64 = image_to_b64(annotated_grid)
        
        # Determine final verdict
        # Trust Gemini if it has high confidence, otherwise use edge detection
        gemini_confidence = gemini_result.get("confidence", 0) if isinstance(gemini_result, dict) else 0
        
        if gemini_confidence >= 0.7:
            final_is_anomaly = gemini_result.get("isAnomaly", edge_result["is_anomaly"])
            final_confidence = gemini_confidence
        else:
            # Edge detection result with Gemini's input
            final_is_anomaly = edge_result["is_anomaly"]
            final_confidence = edge_result["confidence"]
        
        print("\n" + "="*60)
        print(f"🎯 FINAL VERDICT: {'🚨 ANOMALY DETECTED' if final_is_anomaly else '✅ NORMAL'}")
        print(f"   Confidence: {final_confidence:.1%}")
        print("="*60)
        
        return {
            "status": "completed",
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
            },
            
            "annotatedGridB64": f"data:image/jpeg;base64,{grid_b64}",
            "frameCount": len(frames),
            "frameTimestamps": [f["timestamp"] for f in frames],
        }


runpod.serverless.start({"handler": handler})
