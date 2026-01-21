"""
RunPod Serverless Handler for Video Security Analysis
Pipeline: Download video → Extract frames → SigLIP scoring → Gemini verification
"""

import os
import uuid
import base64
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import runpod
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SIGLIP_MODEL = "google/siglip-base-patch16-384"
GEMINI_MODEL = "google/gemini-2.0-flash-001"

# Threat prompt bank for SigLIP scoring
THREAT_PROMPTS = [
    "a person breaking into a building",
    "a person climbing over a fence",
    "a person wearing a mask or balaclava",
    "a person running away quickly",
    "a person carrying a weapon",
    "a person fighting or attacking someone",
    "a person stealing or taking something",
    "a person vandalizing property",
    "a person loitering suspiciously",
    "a person hiding or ducking",
    "a person trespassing in restricted area",
    "a person looking around nervously",
    "a person forcing open a door or window",
    "a person spray painting graffiti",
    "a group of people surrounding someone",
]


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


def download_video(url: str, output_path: str) -> bool:
    """Download video from URL to local path."""
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False


def extract_frames(video_path: str, output_dir: str, fps: float = 2, max_frames: int = 100) -> List[Dict]:
    """
    Extract frames from video using ffmpeg.
    Returns list of dicts with frame path and timestamp.
    """
    frames = []
    
    # Get video duration
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
    except Exception as e:
        print(f"Error getting video duration: {e}")
        duration = 60  # Default fallback
    
    # Calculate actual fps to not exceed max_frames
    total_expected_frames = duration * fps
    if total_expected_frames > max_frames:
        fps = max_frames / duration
    
    # Extract frames
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    extract_cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality JPEG
        "-frames:v", str(max_frames),
        output_pattern
    ]
    
    try:
        subprocess.run(extract_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        return frames
    
    # Collect frame info
    for i, frame_file in enumerate(sorted(Path(output_dir).glob("frame_*.jpg"))):
        timestamp = i / fps
        frames.append({
            "path": str(frame_file),
            "timestamp": timestamp,
            "index": i
        })
    
    return frames


def load_siglip_model():
    """Load SigLIP model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL).to(device)
    return model, processor, device


def score_frames_siglip(
    frames: List[Dict],
    prompts: List[str],
    model,
    processor,
    device: str,
    context_text: str = ""
) -> List[Dict]:
    """
    Score each frame against threat prompts using SigLIP.
    Returns frames with scores for each prompt.
    """
    # Add context-specific prompts if provided
    all_prompts = list(prompts)
    if context_text:
        # Parse context into additional prompts
        context_prompts = [
            f"a person {context_text.lower()}",
            context_text.lower(),
        ]
        all_prompts.extend(context_prompts)
    
    scored_frames = []
    
    for frame_info in frames:
        try:
            image = Image.open(frame_info["path"]).convert("RGB")
            
            # Process image and texts
            inputs = processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Get similarity scores (logits per image)
                logits = outputs.logits_per_image[0]
                probs = torch.sigmoid(logits).cpu().numpy()
            
            # Get max score and corresponding prompt
            max_idx = probs.argmax()
            max_score = float(probs[max_idx])
            max_prompt = all_prompts[max_idx]
            
            scored_frames.append({
                **frame_info,
                "scores": {p: float(s) for p, s in zip(all_prompts, probs)},
                "max_score": max_score,
                "max_prompt": max_prompt,
            })
            
        except Exception as e:
            print(f"Error scoring frame {frame_info['path']}: {e}")
            continue
    
    return scored_frames


def build_event_segments(
    scored_frames: List[Dict],
    threshold: float = 0.3,
    merge_gap: float = 2.0
) -> List[Dict]:
    """
    Build event segments from scored frames.
    Groups consecutive high-scoring frames into events.
    """
    events = []
    current_event = None
    
    for frame in scored_frames:
        if frame["max_score"] >= threshold:
            if current_event is None:
                # Start new event
                current_event = {
                    "id": str(uuid.uuid4()),
                    "t0": frame["timestamp"],
                    "t1": frame["timestamp"],
                    "label": frame["max_prompt"],
                    "scores": [frame["max_score"]],
                    "frames": [frame],
                }
            elif frame["timestamp"] - current_event["t1"] <= merge_gap:
                # Extend current event
                current_event["t1"] = frame["timestamp"]
                current_event["scores"].append(frame["max_score"])
                current_event["frames"].append(frame)
                # Update label to most common high-scoring prompt
            else:
                # Save current event and start new one
                events.append(finalize_event(current_event))
                current_event = {
                    "id": str(uuid.uuid4()),
                    "t0": frame["timestamp"],
                    "t1": frame["timestamp"],
                    "label": frame["max_prompt"],
                    "scores": [frame["max_score"]],
                    "frames": [frame],
                }
        else:
            if current_event is not None:
                events.append(finalize_event(current_event))
                current_event = None
    
    # Don't forget the last event
    if current_event is not None:
        events.append(finalize_event(current_event))
    
    return events


def finalize_event(event: Dict) -> Dict:
    """Calculate final score and severity for an event."""
    avg_score = sum(event["scores"]) / len(event["scores"])
    max_score = max(event["scores"])
    
    # Determine severity
    if max_score >= 0.7:
        severity = "high"
    elif max_score >= 0.5:
        severity = "medium"
    else:
        severity = "low"
    
    # Select keyframes (first, middle, last + highest scoring)
    frames = event["frames"]
    keyframes = []
    
    if len(frames) >= 1:
        keyframes.append(frames[0])
    if len(frames) >= 3:
        keyframes.append(frames[len(frames) // 2])
    if len(frames) >= 2:
        keyframes.append(frames[-1])
    
    # Add highest scoring frame if not already included
    highest_frame = max(frames, key=lambda f: f["max_score"])
    if highest_frame not in keyframes:
        keyframes.insert(1, highest_frame)
    
    # Limit to 4 keyframes
    keyframes = keyframes[:4]
    
    return {
        "id": event["id"],
        "t0": event["t0"],
        "t1": max(event["t1"], event["t0"] + 1),  # Minimum 1 second duration
        "label": event["label"].replace("a person ", "").title(),
        "score": avg_score,
        "severity": severity,
        "keyframes": keyframes,
    }


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for Gemini API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def verify_event_with_gemini(event: Dict, context_text: str) -> Dict:
    """
    Use Gemini to verify and explain the detected event.
    Sends keyframes for multi-frame context analysis.
    """
    if not OPENROUTER_API_KEY:
        return {
            "verdict": True,
            "confidence": event["score"],
            "explanation": f"Detected: {event['label']} (Gemini verification unavailable)"
        }
    
    # Prepare images for Gemini
    image_contents = []
    for kf in event["keyframes"]:
        b64 = encode_image_base64(kf["path"])
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })
    
    # Build prompt
    prompt = f"""You are a security analyst reviewing surveillance footage frames.

Context from user: {context_text}

Initial detection: {event['label']}
Detection confidence: {event['score']:.0%}
Time range: {event['t0']:.1f}s - {event['t1']:.1f}s

Analyze these {len(event['keyframes'])} keyframes from the detected event and determine:
1. Is this a genuine security concern or a false positive?
2. How confident are you in your assessment (0-100%)?
3. Provide a one-line explanation suitable for a security alert.

Respond in this exact JSON format:
{{"verdict": true/false, "confidence": 0.0-1.0, "explanation": "one line explanation"}}

Be conservative - only mark as true if there's clear evidence of concerning behavior."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_contents
            ]
        }
    ]
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GEMINI_MODEL,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.1,
            },
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse JSON response
        import json
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        parsed = json.loads(content.strip())
        
        return {
            "verdict": bool(parsed.get("verdict", True)),
            "confidence": float(parsed.get("confidence", event["score"])),
            "explanation": str(parsed.get("explanation", event["label"]))
        }
        
    except Exception as e:
        print(f"Gemini verification error: {e}")
        return {
            "verdict": True,
            "confidence": event["score"],
            "explanation": f"Detected: {event['label']} (verification error)"
        }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler.
    Input: { video_url, context_text, fps, max_frames }
    Output: { events: [...], videoDuration, processedAt }
    """
    job_input = job.get("input", {})
    
    video_url = job_input.get("video_url")
    context_text = job_input.get("context_text", "Detect suspicious behavior")
    fps = job_input.get("fps", 2)
    max_frames = job_input.get("max_frames", 100)
    
    if not video_url:
        return {"error": "video_url is required"}
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "input_video.mp4")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Step 1: Download video
        print("Downloading video...")
        if not download_video(video_url, video_path):
            return {"error": "Failed to download video"}
        
        # Step 2: Extract frames
        print("Extracting frames...")
        frames = extract_frames(video_path, frames_dir, fps=fps, max_frames=max_frames)
        
        if not frames:
            return {"error": "Failed to extract frames from video"}
        
        print(f"Extracted {len(frames)} frames")
        
        # Step 3: Load SigLIP and score frames
        print("Loading SigLIP model...")
        model, processor, device = load_siglip_model()
        
        print("Scoring frames with SigLIP...")
        scored_frames = score_frames_siglip(
            frames, THREAT_PROMPTS, model, processor, device, context_text
        )
        
        # Step 4: Build event segments
        print("Building event segments...")
        events = build_event_segments(scored_frames, threshold=0.3)
        
        print(f"Found {len(events)} potential events")
        
        # Step 5: Verify each event with Gemini
        print("Verifying events with Gemini...")
        final_events = []
        
        for event in events:
            verification = verify_event_with_gemini(event, context_text)
            
            final_event = TimelineEvent(
                id=event["id"],
                t0=event["t0"],
                t1=event["t1"],
                label=event["label"],
                score=event["score"],
                severity=event["severity"],
                geminiVerdict=verification["verdict"],
                geminiConfidence=verification["confidence"],
                geminiExplanation=verification["explanation"],
                keyframeUrls=[],  # URLs would need to be uploaded separately
            )
            final_events.append(asdict(final_event))
        
        # Get video duration from last frame timestamp
        video_duration = frames[-1]["timestamp"] if frames else 0
        
        return {
            "jobId": job.get("id", ""),
            "status": "completed",
            "videoUrl": video_url,
            "videoDuration": video_duration,
            "events": final_events,
            "processedAt": str(os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip()),
        }


# RunPod serverless handler
runpod.serverless.start({"handler": handler})
