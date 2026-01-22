#!/usr/bin/env python3
"""
RunPod Serverless Handler - Video Security Analysis
====================================================
Hysteresis-based anomaly detection with SigLIP2 + Gemini verification
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
from PIL import Image

# ============ Configuration ============
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GEMINI_MODEL = "google/gemini-2.5-flash-preview"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SigLIP2 model - Shape Optimized 400M, patch14, 384px
SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-384"

print("="*60)
print("🚀 VIDEO SECURITY ANALYSIS - STARTUP")
print("="*60)
print(f"🖥️  Device: {DEVICE}")
print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

if OPENROUTER_API_KEY:
    key_preview = OPENROUTER_API_KEY[:10] + "..." + OPENROUTER_API_KEY[-4:]
    print(f"🔑 OPENROUTER_API_KEY: SET ({key_preview})")
else:
    print(f"❌ OPENROUTER_API_KEY: NOT SET!")

print(f"🤖 Gemini Model: {GEMINI_MODEL}")
print(f"🔬 SigLIP2 Model: {SIGLIP2_MODEL_ID}")
print("="*60)

# ============ Model Loading ============
SIGLIP2_MODEL = None
SIGLIP2_PROCESSOR = None


def load_siglip2():
    """Load SigLIP2 model via HuggingFace transformers."""
    global SIGLIP2_MODEL, SIGLIP2_PROCESSOR
    
    if SIGLIP2_MODEL is not None:
        return SIGLIP2_MODEL, SIGLIP2_PROCESSOR
    
    from transformers import AutoModel, AutoProcessor
    
    print(f"📦 Loading SigLIP2 model: {SIGLIP2_MODEL_ID}...")
    
    SIGLIP2_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
    SIGLIP2_MODEL = AutoModel.from_pretrained(SIGLIP2_MODEL_ID).to(DEVICE).eval()
    
    print(f"✅ SigLIP2 loaded on {DEVICE}")
    return SIGLIP2_MODEL, SIGLIP2_PROCESSOR


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


def call_gemini(prompt: str, image_b64: str = None, max_tokens: int = 1500) -> str | dict:
    """Call Gemini via OpenRouter."""
    if not OPENROUTER_API_KEY:
        print("❌ call_gemini failed: OPENROUTER_API_KEY not set!")
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
                    "temperature": 0.2
                },
                timeout=90
            )
            result = response.json()
            if "error" in result:
                continue
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            if attempt == 2:
                return {"error": str(e)}
    return {"error": "Failed after 3 attempts"}


def image_to_b64(img: Image.Image, quality: int = 90, as_data_url: bool = False) -> str:
    """Convert PIL Image to base64 string."""
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    if as_data_url:
        return f"data:image/jpeg;base64,{b64}"
    return b64


def extract_reference_frame(video_path: str, position: float = 0.20) -> Optional[Image.Image]:
    """Extract a single reference frame from the video at given position (0.0-1.0)."""
    print(f"📸 Extracting reference frame at {position:.0%}...")
    try:
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]).decode().strip()
        duration = float(probe)
        timestamp = duration * position
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        
        subprocess.run([
            "ffmpeg", "-ss", str(timestamp), "-i", video_path,
            "-vframes", "1", "-q:v", "2", tmp_path, "-y"
        ], capture_output=True, check=True)
        
        img = Image.open(tmp_path).convert("RGB")
        os.unlink(tmp_path)
        print(f"  ✅ Reference frame: {img.size[0]}x{img.size[1]} at t={timestamp:.1f}s")
        return img
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")
        return None


def generate_elaborate_prompt_banks(
    camera_context: str, 
    detection_targets: str, 
    reference_frame: Optional[Image.Image] = None
) -> tuple:
    """
    Generate comprehensive prompt banks using 3 Gemini calls:
    1. Scene analysis + baseline normal behavior
    2. Target-specific anomaly patterns  
    3. Edge cases and subtle indicators
    """
    print("🤖 Generating elaborate prompt banks (3 Gemini calls)...")
    
    img_b64 = image_to_b64(reference_frame) if reference_frame else None
    
    # ========== CALL 1: Scene Analysis + Normal Behavior ==========
    print("   📞 Call 1/3: Scene analysis & normal behavior...")
    
    prompt1 = f"""You are an expert in video surveillance and anomaly detection.

{"=== REFERENCE FRAME (ATTACHED) ===" if img_b64 else ""}
{"Study this image carefully - what objects, furniture, equipment, and layout do you see?" if img_b64 else ""}

=== CONTEXT ===
Camera location: {camera_context}
Detection target: {detection_targets}

=== TASK 1: SCENE ANALYSIS ===
Describe what you observe in detail:
- Physical layout and key landmarks
- Lighting conditions
- Expected traffic patterns
- Normal objects and their positions

=== TASK 2: NORMAL BEHAVIOR PROMPTS (25 prompts) ===
Generate 25 prompts describing NORMAL, EXPECTED activity in this scene.
Each prompt should be 5-15 words, visually descriptive.

Categories to cover:
- People in normal positions/postures
- Expected movements and activities
- Normal object placements
- Typical interactions
- Routine operations

JSON ONLY:
{{
    "scene_description": "detailed scene analysis",
    "key_landmarks": ["landmark1", "landmark2", ...],
    "normal_prompts": ["prompt1", "prompt2", ...]
}}"""

    response1 = call_gemini(prompt1, image_b64=img_b64, max_tokens=2000)
    if isinstance(response1, dict) and "error" in response1:
        return None, f"Call 1 error: {response1['error']}"
    
    parsed1 = extract_json_object(response1)
    if not parsed1:
        return None, "Failed to parse Call 1 response"
    
    scene_description = parsed1.get("scene_description", "")
    key_landmarks = parsed1.get("key_landmarks", [])
    normal_prompts = parsed1.get("normal_prompts", [])
    
    print(f"      ✅ Got {len(normal_prompts)} normal prompts")
    
    # ========== CALL 2: Target-Specific Anomaly Patterns ==========
    print("   📞 Call 2/3: Target-specific anomaly patterns...")
    
    prompt2 = f"""You are an expert in detecting "{detection_targets}" in surveillance footage.

=== SCENE CONTEXT ===
Location: {camera_context}
Scene details: {scene_description}
Key landmarks: {', '.join(key_landmarks[:5])}

=== TASK: ANOMALY DETECTION PROMPTS (30 prompts) ===
Generate 30 prompts describing what "{detection_targets}" LOOKS LIKE visually.

Think about the VISUAL INDICATORS at different stages:
1. PREPARATION phase (5 prompts) - body language before the act
2. ACTIVE phase (10 prompts) - the act itself happening
3. EXECUTION phase (10 prompts) - specific actions and movements
4. AFTERMATH phase (5 prompts) - post-incident indicators

For each prompt:
- Be EXTREMELY specific and visual
- Reference the scene landmarks when relevant
- Describe body positions, hand movements, object interactions
- Focus on what a SINGLE FRAME would show

JSON ONLY:
{{
    "preparation_prompts": ["prompt1", ...],
    "active_prompts": ["prompt1", ...],
    "execution_prompts": ["prompt1", ...],
    "aftermath_prompts": ["prompt1", ...]
}}"""

    response2 = call_gemini(prompt2, max_tokens=2500)
    if isinstance(response2, dict) and "error" in response2:
        return None, f"Call 2 error: {response2['error']}"
    
    parsed2 = extract_json_object(response2)
    if not parsed2:
        return None, "Failed to parse Call 2 response"
    
    anomaly_prompts = (
        parsed2.get("preparation_prompts", []) +
        parsed2.get("active_prompts", []) +
        parsed2.get("execution_prompts", []) +
        parsed2.get("aftermath_prompts", [])
    )
    
    print(f"      ✅ Got {len(anomaly_prompts)} anomaly prompts")
    
    # ========== CALL 3: Edge Cases & Subtle Indicators ==========
    print("   📞 Call 3/3: Edge cases & subtle indicators...")
    
    prompt3 = f"""You are refining anomaly detection prompts for "{detection_targets}".

=== CONTEXT ===
Location: {camera_context}
Scene: {scene_description}

=== CURRENT PROMPTS ===
Normal examples: {normal_prompts[:3]}
Anomaly examples: {anomaly_prompts[:3]}

=== TASK: REFINEMENT ===
Generate additional prompts to improve detection accuracy:

1. FALSE POSITIVE PREVENTION (10 prompts)
   Normal activities that might LOOK suspicious but are NOT "{detection_targets}"
   Add these to help distinguish real threats from false alarms.

2. SUBTLE INDICATORS (10 prompts)  
   Easily-missed visual signs of "{detection_targets}" that are subtle:
   - Micro-expressions or body language
   - Peripheral activities
   - Environmental changes
   - Object displacements

3. CONTEXTUAL VARIATIONS (5 prompts)
   How "{detection_targets}" might look different in various conditions:
   - Different lighting
   - Crowded vs empty
   - Different camera angles

JSON ONLY:
{{
    "false_positive_normals": ["prompt1", ...],
    "subtle_anomaly_indicators": ["prompt1", ...],
    "contextual_anomaly_variations": ["prompt1", ...]
}}"""

    response3 = call_gemini(prompt3, max_tokens=2000)
    if isinstance(response3, dict) and "error" in response3:
        return None, f"Call 3 error: {response3['error']}"
    
    parsed3 = extract_json_object(response3)
    if not parsed3:
        return None, "Failed to parse Call 3 response"
    
    # Combine all prompts
    final_normal_prompts = normal_prompts + parsed3.get("false_positive_normals", [])
    final_anomaly_prompts = (
        anomaly_prompts + 
        parsed3.get("subtle_anomaly_indicators", []) +
        parsed3.get("contextual_anomaly_variations", [])
    )
    
    print(f"  ✅ Final: {len(final_normal_prompts)} normal, {len(final_anomaly_prompts)} anomaly prompts")
    
    return {
        "normal_prompts": final_normal_prompts,
        "anomaly_prompts": final_anomaly_prompts,
        "scene_description": scene_description,
        "key_landmarks": key_landmarks,
    }, None


def download_video(url: str, output_path: str) -> bool:
    """Download video from URL."""
    print(f"📥 Downloading video...")
    
    if url.startswith('blob:'):
        print(f"  ❌ Cannot download browser blob: URL!")
        return False
    
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


def precompute_text_features(normal_prompts: List[str], anomaly_prompts: List[str]) -> tuple:
    """Pre-compute text features for SigLIP2."""
    model, processor = load_siglip2()
    
    with torch.no_grad():
        normal_inputs = processor(text=normal_prompts, padding="max_length", return_tensors="pt").to(DEVICE)
        normal_features = model.get_text_features(**normal_inputs)
        normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
        
        anomaly_inputs = processor(text=anomaly_prompts, padding="max_length", return_tensors="pt").to(DEVICE)
        anomaly_features = model.get_text_features(**anomaly_inputs)
        anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
    
    return normal_features, anomaly_features


def siglip2_score_single_frame(
    frame: Image.Image,
    normal_features: torch.Tensor,
    anomaly_features: torch.Tensor
) -> tuple:
    """Score a single frame using SigLIP2. Returns (score_diff, normal_sim, anomaly_sim)."""
    model, processor = load_siglip2()
    
    inputs = processor(images=frame, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        img_features = model.get_image_features(**inputs)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    
    normal_sim = (img_features @ normal_features.T).max(dim=1).values.item()
    anomaly_sim = (img_features @ anomaly_features.T).max(dim=1).values.item()
    score_diff = anomaly_sim - normal_sim
    
    return score_diff, normal_sim, anomaly_sim


def process_video_hysteresis(
    video_path: str,
    normal_prompts: List[str],
    anomaly_prompts: List[str],
    sample_fps: float = 2.0,
    high_threshold: float = 0.0,
    low_threshold: float = 0.0,
    min_frames_to_trigger: int = 1,
    min_frames_to_clear: int = 4,
) -> Dict:
    """
    Process video with hysteresis-based anomaly detection.
    
    State machine:
    - NORMAL → ANOMALY: score > high_threshold for N consecutive frames
    - ANOMALY → NORMAL: score < low_threshold for M consecutive frames
    """
    print(f"\n🔄 Hysteresis detection (sample_fps={sample_fps})...")
    print(f"   Thresholds: HIGH={high_threshold}, LOW={low_threshold}")
    
    duration, video_fps = get_video_info(video_path)
    total_samples = int(duration * sample_fps)
    print(f"   Video: {duration:.1f}s → {total_samples} samples")
    
    print("   Pre-computing text features...")
    normal_features, anomaly_features = precompute_text_features(normal_prompts, anomaly_prompts)
    
    state = "NORMAL"
    consecutive_high = 0
    consecutive_low = 0
    
    events = []
    current_event_start = None
    current_event_peak = 0
    current_event_scores = []
    current_event_frames = []  # Store frames during anomaly
    
    all_scores = []
    
    print(f"   Processing {total_samples} frames...")
    
    for i in range(total_samples):
        timestamp = i / sample_fps
        
        if i % 10 == 0:
            progress = (i / total_samples) * 100
            print(f"\r   ⏳ {progress:.0f}% ({i}/{total_samples}) | State: {state}", end="", flush=True)
        
        frame = extract_single_frame(video_path, timestamp)
        if frame is None:
            continue
        
        score, normal_sim, anomaly_sim = siglip2_score_single_frame(
            frame, normal_features, anomaly_features
        )
        
        all_scores.append({
            "timestamp": timestamp,
            "score": score,
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
                    current_event_frames = [(timestamp, frame)]
                    print(f"\n   🚨 ANOMALY at {current_event_start:.1f}s")
            else:
                consecutive_high = 0
                
        elif state == "ANOMALY":
            current_event_peak = max(current_event_peak, score)
            current_event_scores.append(score)
            current_event_frames.append((timestamp, frame))
            
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
                        "frames": current_event_frames,
                    })
                    print(f"\n   ✅ Event ended at {event_end:.1f}s")
                    state = "NORMAL"
                    current_event_start = None
                    current_event_frames = []
            else:
                consecutive_low = 0
    
    print(f"\r   ⏳ 100% Complete!                              ")
    
    # Handle video ending during anomaly
    if state == "ANOMALY" and current_event_start is not None:
        events.append({
            "start": current_event_start,
            "end": duration,
            "duration": duration - current_event_start,
            "peak_score": current_event_peak,
            "avg_score": float(np.mean(current_event_scores)) if current_event_scores else current_event_peak,
            "frames": current_event_frames,
        })
    
    total_anomaly_duration = sum(e["duration"] for e in events)
    
    print(f"\n📊 Results: {len(events)} events, {total_anomaly_duration:.1f}s anomaly time")
    
    return {
        "is_anomaly": len(events) > 0,
        "events": events,
        "all_scores": all_scores,
        "total_anomaly_duration": total_anomaly_duration,
        "video_duration": duration,
    }


def gemini_verify_8_frames(
    event_frames: List[tuple],  # [(timestamp, PIL.Image), ...]
    camera_context: str,
    detection_targets: str,
) -> Dict:
    """
    Gemini verification using 8 frames from detected anomaly.
    Returns reasoning + one-word category.
    """
    print("🧠 Gemini verification with 8 frames...")
    
    # Select up to 8 frames evenly distributed
    if len(event_frames) <= 8:
        selected = event_frames
    else:
        indices = np.linspace(0, len(event_frames) - 1, 8, dtype=int)
        selected = [event_frames[i] for i in indices]
    
    # Create 2x4 grid for display
    tile_size = 384
    grid = Image.new('RGB', (tile_size * 4, tile_size * 2), (0, 0, 0))
    
    for idx, (ts, img) in enumerate(selected[:8]):
        resized = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        x = (idx % 4) * tile_size
        y = (idx // 4) * tile_size
        grid.paste(resized, (x, y))
    
    grid_b64 = image_to_b64(grid)
    
    prompt = f"""=== VIDEO ANOMALY VERIFICATION ===

You are analyzing 8 sequential frames from a security camera that detected potential "{detection_targets}".

SCENE: {camera_context}
TARGET: {detection_targets}

=== YOUR TASK ===
Examine each frame carefully from left to right, top to bottom (chronological order).

1. What do you SEE happening across these frames?
2. Is this actually "{detection_targets}" or something else?
3. If it IS an anomaly, what specific visual evidence confirms it?

=== RESPONSE FORMAT ===
JSON ONLY:
{{
    "category": "ONE WORD describing the anomaly type (e.g., 'theft', 'assault', 'vandalism', 'trespassing', 'normal')",
    "reasoning": "2-3 sentences explaining what you observe in the frames and why this is or isn't {detection_targets}. Be specific about which frames show what."
}}"""

    response = call_gemini(prompt, grid_b64, max_tokens=500)
    
    if isinstance(response, dict) and "error" in response:
        return {"category": "unknown", "reasoning": f"Verification failed: {response['error']}"}
    
    parsed = extract_json_object(response)
    if parsed:
        return {
            "category": parsed.get("category", "unknown"),
            "reasoning": parsed.get("reasoning", ""),
            "frames_grid_b64": image_to_b64(grid, as_data_url=True),
        }
    
    return {
        "category": "unknown",
        "reasoning": response,
        "frames_grid_b64": image_to_b64(grid, as_data_url=True),
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hysteresis-based Video Anomaly Detection Pipeline:
    
    1. Download video
    2. Extract reference frame for visual context
    3. Generate elaborate prompts (3 Gemini calls)
    4. Process video with hysteresis state machine
    5. Verify anomalies with Gemini using 8 frames
    6. Return: category (one word), reasoning, anomaly frames
    """
    job_input = job.get("input", {})
    
    video_url = job_input.get("video_url")
    camera_context = job_input.get("camera_context", "Security camera")
    detection_targets = job_input.get("detection_targets", "Suspicious activity")
    
    # Hysteresis parameters
    sample_fps = job_input.get("sample_fps", 2.0)
    high_threshold = job_input.get("high_threshold", 0.0)
    low_threshold = job_input.get("low_threshold", 0.0)
    min_frames_to_trigger = job_input.get("min_frames_to_trigger", 1)
    min_frames_to_clear = job_input.get("min_frames_to_clear", 4)
    
    if not video_url:
        return {"error": "video_url is required"}
    
    print("\n" + "="*60)
    print(f"🎬 Video Anomaly Detection")
    print(f"   Scene: {camera_context[:40]}...")
    print(f"   Target: {detection_targets[:40]}...")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "video.mp4")
        
        # Step 1: Download video
        if not download_video(video_url, video_path):
            return {"error": "Download failed"}
        
        duration, _ = get_video_info(video_path)
        print(f"📹 Video: {duration:.1f}s")
        
        # Step 2: Extract reference frame
        reference_frame = extract_reference_frame(video_path, position=0.20)
        
        # Step 3: Generate elaborate prompts (3 Gemini calls)
        prompt_banks, error = generate_elaborate_prompt_banks(
            camera_context, 
            detection_targets, 
            reference_frame
        )
        if error:
            return {"error": f"Prompt generation failed: {error}"}
        
        normal_prompts = prompt_banks["normal_prompts"]
        anomaly_prompts = prompt_banks["anomaly_prompts"]
        
        # Step 4: Hysteresis detection
        result = process_video_hysteresis(
            video_path,
            normal_prompts,
            anomaly_prompts,
            sample_fps=sample_fps,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            min_frames_to_trigger=min_frames_to_trigger,
            min_frames_to_clear=min_frames_to_clear,
        )
        
        if not result["is_anomaly"]:
            print("\n✅ VERDICT: NORMAL - No anomalies detected")
            return {
                "category": "normal",
                "reasoning": "No anomalous activity detected in the video. All frames showed normal behavior consistent with expected activity in this scene.",
                "anomalyFramesB64": None,
            }
        
        # Step 5: Gemini verification with 8 frames
        # Collect all frames from all events
        all_event_frames = []
        for event in result["events"]:
            all_event_frames.extend(event.get("frames", []))
        
        if not all_event_frames:
            # Fallback: extract frames from event timestamps
            main_event = result["events"][0]
            for i in range(8):
                t = main_event["start"] + (i / 7) * main_event["duration"]
                frame = extract_single_frame(video_path, t)
                if frame:
                    all_event_frames.append((t, frame))
        
        verification = gemini_verify_8_frames(
            all_event_frames,
            camera_context,
            detection_targets
        )
        
        print(f"\n🎯 VERDICT: {verification['category'].upper()}")
        print(f"   {verification['reasoning'][:100]}...")
        
        return {
            "category": verification["category"],
            "reasoning": verification["reasoning"],
            "anomalyFramesB64": verification.get("frames_grid_b64"),
        }


runpod.serverless.start({"handler": handler})
