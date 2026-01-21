"""
RunPod Serverless Handler for Video Security Analysis
======================================================
2-Stage Pipeline with AI-Generated Prompt Banks

Stage 1: SigLIP2 Edge Detection
  - Dynamic prompt banks from Gemini based on camera context
  - Score frames against normal vs anomaly prompts
  - Per-frame anomaly scoring

Stage 2: Gemini Verification  
  - Annotated frame grid with edge scores
  - Detailed reasoning and verification
  - Final verdict with confidence
"""

import os
import json
import uuid
import base64
import tempfile
import subprocess
import hashlib
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from io import BytesIO

import runpod
import requests
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Try to import open_clip for SigLIP2
try:
    import open_clip
    USE_OPEN_CLIP = True
except ImportError:
    from transformers import AutoProcessor, AutoModel
    USE_OPEN_CLIP = False

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GEMINI_MODEL = "google/gemini-3.0-flash-preview"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model cache
SIGLIP_MODEL = None
SIGLIP_PREPROCESS = None
SIGLIP_TOKENIZER = None


# ============ Robust JSON Extraction ============

def extract_json_object(text: str) -> dict | None:
    """Robustly extract JSON object from LLM response."""
    if not text:
        return None
    
    text = text.strip()
    
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Fallback: scan for valid JSON object
    for match in re.finditer(r"\{", text):
        start = match.start()
        for end in range(len(text), start, -1):
            if text[end-1] != "}":
                continue
            snippet = text[start:end]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
    
    return None


# ============ Gemini API ============

def call_gemini(prompt: str, image_b64: str = None, max_tokens: int = 1000) -> str | dict:
    """Call Gemini API via OpenRouter with retry logic."""
    if not OPENROUTER_API_KEY:
        return {"error": "No OPENROUTER_API_KEY configured"}
    
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
                if attempt < 2:
                    continue
                return {"error": result.get("error", {}).get("message", str(result))}
            
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.Timeout:
            if attempt < 2:
                continue
            return {"error": "API timeout after 3 attempts"}
        except Exception as e:
            if attempt < 2:
                continue
            return {"error": str(e)}
    
    return {"error": "Failed after 3 attempts"}


# ============ Dynamic Prompt Bank Generation ============

def generate_prompt_banks(camera_context: str, detection_targets: str) -> tuple:
    """
    Use Gemini to generate normal/anomaly prompt banks for SigLIP2.
    Returns (prompt_banks_dict, error_message)
    """
    prompt = f"""You are an expert in video anomaly detection systems. Generate prompt banks for a 2-stage detection pipeline.

=== CAMERA CONTEXT ===
{camera_context}

=== WHAT USER WANTS TO DETECT ===
{detection_targets}

=== YOUR TASK ===
Generate two sets of text prompts optimized for CLIP/SigLIP image-text matching:

1. **NORMAL PROMPTS**: 15-20 descriptions of normal, expected behavior in this camera's context
2. **ANOMALY PROMPTS**: 15-20 descriptions of abnormal/suspicious behavior the user wants to detect

IMPORTANT: 
- Each prompt should be a short, visual description (5-15 words)
- Focus on what can be SEEN in a single frame
- Be specific to the camera context provided
- Normal prompts should capture typical activity
- Anomaly prompts should describe visible signs of the target anomalies

Respond in valid JSON format ONLY (no markdown, no explanation):
{{
    "normal_prompts": [
        "description 1",
        "description 2"
    ],
    "anomaly_prompts": [
        "description 1", 
        "description 2"
    ],
    "detection_summary": "Brief summary of what the system will detect"
}}"""

    response = call_gemini(prompt, max_tokens=1500)
    
    if isinstance(response, dict) and "error" in response:
        return None, f"Error: {response['error']}"
    
    parsed = extract_json_object(response)
    
    if parsed and "normal_prompts" in parsed and "anomaly_prompts" in parsed:
        return parsed, None
    
    return None, f"Failed to parse prompt banks: {response[:500]}"


# ============ Model Loading ============

def load_siglip():
    """Load SigLIP2 model using open_clip or transformers fallback."""
    global SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER
    
    if SIGLIP_MODEL is not None:
        return SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER
    
    print(f"Loading SigLIP model on {DEVICE}...")
    
    if USE_OPEN_CLIP:
        # Use open_clip for SigLIP2 (better performance)
        SIGLIP_MODEL, _, SIGLIP_PREPROCESS = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384", pretrained="webli"
        )
        SIGLIP_MODEL = SIGLIP_MODEL.to(DEVICE).eval()
        SIGLIP_TOKENIZER = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
        print("✓ SigLIP2 loaded via open_clip")
    else:
        # Fallback to transformers
        from transformers import AutoProcessor, AutoModel
        model_name = "google/siglip-so400m-patch14-384"
        SIGLIP_PREPROCESS = AutoProcessor.from_pretrained(model_name)
        SIGLIP_MODEL = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
        SIGLIP_TOKENIZER = SIGLIP_PREPROCESS  # Use processor for tokenization
        print("✓ SigLIP loaded via transformers")
    
    return SIGLIP_MODEL, SIGLIP_PREPROCESS, SIGLIP_TOKENIZER


# ============ Frame Extraction ============

def download_video(url: str, output_path: str) -> bool:
    """Download video from URL."""
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
    """Extract frames from video using ffmpeg."""
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
    except Exception:
        duration = 60
    
    # Adjust fps to not exceed max_frames
    total_expected = duration * fps
    if total_expected > max_frames:
        fps = max_frames / duration
    
    # Extract frames
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    extract_cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
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


# ============ Stage 1: SigLIP2 Edge Detection ============

def siglip_detect(frames: List[Dict], normal_prompts: List[str], anomaly_prompts: List[str]) -> Dict:
    """
    Stage 1: SigLIP2 detection with dynamic prompt banks.
    Scores frames against normal vs anomaly prompts.
    """
    model, preprocess, tokenizer = load_siglip()
    
    if not normal_prompts or not anomaly_prompts:
        return {"error": "No prompts provided"}
    
    # Load and preprocess images
    images = []
    for frame_info in frames:
        try:
            img = Image.open(frame_info["path"]).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Error loading frame {frame_info['path']}: {e}")
            continue
    
    if not images:
        return {"error": "No valid frames"}
    
    if USE_OPEN_CLIP:
        # Open_clip path
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
        
        # Per-frame similarities (max across prompts)
        normal_sim = (img_features @ normal_features.T).max(dim=1).values
        anomaly_sim = (img_features @ anomaly_features.T).max(dim=1).values
        per_frame_scores = (anomaly_sim - normal_sim).cpu().numpy().tolist()
        
        # Aggregate scores
        avg_normal = normal_sim.mean().item()
        avg_anomaly = anomaly_sim.mean().item()
        
        # Best matching prompts
        best_anomaly_idx = (img_features @ anomaly_features.T).mean(dim=0).argmax().item()
        best_normal_idx = (img_features @ normal_features.T).mean(dim=0).argmax().item()
        
    else:
        # Transformers path
        per_frame_scores = []
        all_anomaly_scores = []
        all_normal_scores = []
        
        for img in images:
            inputs = preprocess(
                text=normal_prompts + anomaly_prompts,
                images=img,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.sigmoid(logits).cpu().numpy()
            
            n_normal = len(normal_prompts)
            normal_scores = probs[:n_normal]
            anomaly_scores = probs[n_normal:]
            
            max_normal = float(normal_scores.max())
            max_anomaly = float(anomaly_scores.max())
            
            per_frame_scores.append(max_anomaly - max_normal)
            all_normal_scores.append(max_normal)
            all_anomaly_scores.append(max_anomaly)
        
        avg_normal = np.mean(all_normal_scores)
        avg_anomaly = np.mean(all_anomaly_scores)
        best_anomaly_idx = 0
        best_normal_idx = 0
    
    score_diff = avg_anomaly - avg_normal
    
    return {
        "is_anomaly": score_diff > 0,
        "score_diff": score_diff,
        "anomaly_score": avg_anomaly,
        "normal_score": avg_normal,
        "confidence": abs(score_diff),
        "per_frame_scores": per_frame_scores,
        "top_anomaly_prompt": anomaly_prompts[best_anomaly_idx] if anomaly_prompts else "",
        "top_normal_prompt": normal_prompts[best_normal_idx] if normal_prompts else "",
    }


# ============ Frame Grid Creation ============

def create_annotated_grid(frames: List[Dict], scores: List[float] = None, cols: int = 4, tile_size: int = 256) -> Image.Image:
    """Create annotated frame grid with per-frame scores for Gemini verification."""
    if not frames:
        return None
    
    images = []
    for frame_info in frames:
        try:
            img = Image.open(frame_info["path"]).convert("RGB")
            images.append(img)
        except:
            continue
    
    if not images:
        return None
    
    n = len(images)
    rows = (n + cols - 1) // cols
    header_height = 36
    
    grid = Image.new('RGB', (cols * tile_size, rows * (tile_size + header_height)), (12, 12, 20))
    draw = ImageDraw.Draw(grid)
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    for i, img in enumerate(images):
        row, col = i // cols, i % cols
        x, y = col * tile_size, row * (tile_size + header_height)
        
        # Resize and paste frame
        tile = img.resize((tile_size - 4, tile_size - 4), Image.BICUBIC)
        grid.paste(tile, (x + 2, y + header_height))
        
        # Draw score header
        header_bg_color = (25, 25, 40)
        text_color = (150, 150, 150)
        icon = "○"
        
        if scores and i < len(scores):
            score = scores[i]
            if score > 0.015:
                header_bg_color = (80, 20, 20)
                text_color = (255, 100, 100)
                icon = "⚠️"
            elif score < -0.015:
                header_bg_color = (20, 60, 20)
                text_color = (100, 255, 100)
                icon = "✓"
            label = f"{icon} F{i+1}: {score:+.3f}"
        else:
            label = f"Frame {i+1}"
        
        draw.rectangle([x, y, x + tile_size - 1, y + header_height - 1], fill=header_bg_color)
        draw.text((x + 8, y + 8), label, fill=text_color, font=font)
    
    return grid


def image_to_b64(img: Image.Image, quality: int = 90) -> str:
    """Convert PIL image to base64."""
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ============ Stage 2: Gemini Verification ============

def gemini_verify(frames: List[Dict], camera_context: str, detection_targets: str, edge_result: Dict) -> Dict:
    """
    Stage 2: Gemini verification with annotated frame grid.
    Provides detailed reasoning and final verdict.
    """
    # Create annotated grid
    grid = create_annotated_grid(frames, edge_result.get("per_frame_scores"))
    if grid is None:
        return {"error": "Failed to create frame grid"}
    
    img_b64 = image_to_b64(grid)
    
    # Build per-frame description
    per_frame_desc = "\n".join([
        f"  Frame {i+1}: score={s:+.3f} ({'anomaly likely' if s > 0.01 else 'normal likely' if s < -0.01 else 'uncertain'})"
        for i, s in enumerate(edge_result.get('per_frame_scores', []))
    ])
    
    prompt = f"""=== VIDEO ANOMALY VERIFICATION ===

CAMERA CONTEXT: {camera_context}

USER DETECTION TARGETS: {detection_targets}

EDGE DETECTOR (SigLIP2) RESULT:
- Initial Assessment: {"ANOMALY DETECTED" if edge_result.get('is_anomaly') else "NORMAL"}
- Confidence Score: {edge_result.get('confidence', 0):.3f}
- Top Anomaly Match: "{edge_result.get('top_anomaly_prompt', 'N/A')}"
- Top Normal Match: "{edge_result.get('top_normal_prompt', 'N/A')}"
- Score Difference (anomaly - normal): {edge_result.get('score_diff', 0):+.4f}

PER-FRAME SCORES (shown in image headers):
{per_frame_desc}

=== VERIFICATION TASK ===
The image shows a frame grid with frame numbers and edge detector scores in the headers.
- Positive scores (⚠️ red) indicate anomaly likelihood
- Negative scores (✓ green) indicate normal likelihood

1. VERIFY: Is the edge detector's assessment correct? Look carefully at the visual evidence.
2. CLASSIFY: If anomaly, what specific type matches the user's detection targets?
3. EXPLAIN: Provide detailed reasoning about what you observe.
4. CONFIDENCE: Rate your confidence in this assessment.

Consider:
- False positives: Normal activity that might look suspicious
- False negatives: Subtle anomalies the edge detector might miss
- Context: What would be normal vs abnormal for THIS specific camera location

Respond in JSON format ONLY:
{{
    "is_anomaly": true/false,
    "anomaly_type": "specific type or null if normal",
    "confidence": 0.0-1.0,
    "reasoning": "detailed analysis of visual evidence",
    "edge_assessment": "agree/disagree with edge detector and why",
    "key_observations": ["observation 1", "observation 2"]
}}"""

    response = call_gemini(prompt, img_b64, max_tokens=800)
    
    if isinstance(response, dict) and "error" in response:
        return {"error": response["error"]}
    
    parsed = extract_json_object(response)
    
    if parsed:
        return {
            "is_anomaly": parsed.get("is_anomaly", False),
            "anomaly_type": parsed.get("anomaly_type"),
            "confidence": parsed.get("confidence", 0),
            "reasoning": parsed.get("reasoning", ""),
            "edge_assessment": parsed.get("edge_assessment", ""),
            "key_observations": parsed.get("key_observations", []),
        }
    
    # Fallback
    return {
        "reasoning": response,
        "is_anomaly": "anomaly" in response.lower() and "no anomaly" not in response.lower()
    }


# ============ Main Handler ============

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


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler.
    
    Input: {
        video_url: str,
        camera_context: str,      # NEW: Where is the camera, what does it normally see
        detection_targets: str,   # NEW: What anomalies to detect
        fps: float (optional),
        max_frames: int (optional)
    }
    
    Output: {
        events: [...],
        prompt_banks: {...},
        edge_detection: {...},
        gemini_verification: {...},
        videoDuration: float
    }
    """
    job_input = job.get("input", {})
    
    video_url = job_input.get("video_url")
    camera_context = job_input.get("camera_context", job_input.get("context_text", "Security surveillance camera"))
    detection_targets = job_input.get("detection_targets", job_input.get("context_text", "Suspicious or threatening behavior"))
    fps = job_input.get("fps", 2)
    max_frames = job_input.get("max_frames", 64)
    
    if not video_url:
        return {"error": "video_url is required"}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "input_video.mp4")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Step 1: Download video
        print("📥 Downloading video...")
        if not download_video(video_url, video_path):
            return {"error": "Failed to download video"}
        
        # Step 2: Extract frames
        print("🎞️ Extracting frames...")
        frames = extract_frames(video_path, frames_dir, fps=fps, max_frames=max_frames)
        
        if not frames:
            return {"error": "Failed to extract frames"}
        
        print(f"  → Extracted {len(frames)} frames")
        
        # Step 3: Generate dynamic prompt banks from Gemini
        print("🤖 Generating prompt banks...")
        prompt_banks, error = generate_prompt_banks(camera_context, detection_targets)
        
        if error:
            return {"error": f"Prompt generation failed: {error}"}
        
        normal_prompts = prompt_banks.get("normal_prompts", [])
        anomaly_prompts = prompt_banks.get("anomaly_prompts", [])
        
        print(f"  → Normal prompts: {len(normal_prompts)}")
        print(f"  → Anomaly prompts: {len(anomaly_prompts)}")
        
        # Step 4: Stage 1 - SigLIP2 Edge Detection
        print("⚡ Stage 1: SigLIP2 edge detection...")
        edge_result = siglip_detect(frames, normal_prompts, anomaly_prompts)
        
        if "error" in edge_result:
            return {"error": f"Edge detection failed: {edge_result['error']}"}
        
        print(f"  → Initial: {'ANOMALY' if edge_result['is_anomaly'] else 'NORMAL'}")
        print(f"  → Score diff: {edge_result['score_diff']:+.4f}")
        
        # Step 5: Stage 2 - Gemini Verification
        print("🧠 Stage 2: Gemini verification...")
        gemini_result = gemini_verify(frames, camera_context, detection_targets, edge_result)
        
        if "error" in gemini_result:
            print(f"  ⚠️ Gemini error: {gemini_result['error']}")
            # Continue with edge result only
            final_is_anomaly = edge_result.get("is_anomaly", False)
            final_confidence = edge_result.get("confidence", 0)
            anomaly_type = edge_result.get("top_anomaly_prompt", "")
            explanation = f"Edge detection: {edge_result.get('top_anomaly_prompt', 'Unknown')}"
        else:
            final_is_anomaly = gemini_result.get("is_anomaly", edge_result.get("is_anomaly", False))
            final_confidence = gemini_result.get("confidence", edge_result.get("confidence", 0))
            anomaly_type = gemini_result.get("anomaly_type", "")
            explanation = gemini_result.get("reasoning", "")
        
        print(f"  → Final: {'ANOMALY' if final_is_anomaly else 'NORMAL'}")
        print(f"  → Confidence: {final_confidence:.0%}")
        
        # Build timeline events
        events = []
        if final_is_anomaly:
            # Find the segment with highest anomaly scores
            scores = edge_result.get("per_frame_scores", [])
            if scores:
                # Find contiguous anomaly regions
                anomaly_start = None
                for i, score in enumerate(scores):
                    if score > 0:
                        if anomaly_start is None:
                            anomaly_start = i
                    else:
                        if anomaly_start is not None:
                            # End of anomaly region
                            t0 = frames[anomaly_start]["timestamp"]
                            t1 = frames[min(i, len(frames)-1)]["timestamp"]
                            avg_score = np.mean(scores[anomaly_start:i])
                            
                            severity = "high" if avg_score > 0.05 else "medium" if avg_score > 0.02 else "low"
                            
                            events.append(asdict(TimelineEvent(
                                id=str(uuid.uuid4()),
                                t0=t0,
                                t1=max(t1, t0 + 1),
                                label=anomaly_type or edge_result.get("top_anomaly_prompt", "Anomaly"),
                                score=float(avg_score),
                                severity=severity,
                                geminiVerdict=final_is_anomaly,
                                geminiConfidence=final_confidence,
                                geminiExplanation=explanation,
                                keyframeUrls=[]
                            )))
                            anomaly_start = None
                
                # Handle trailing anomaly
                if anomaly_start is not None:
                    t0 = frames[anomaly_start]["timestamp"]
                    t1 = frames[-1]["timestamp"]
                    avg_score = np.mean(scores[anomaly_start:])
                    severity = "high" if avg_score > 0.05 else "medium" if avg_score > 0.02 else "low"
                    
                    events.append(asdict(TimelineEvent(
                        id=str(uuid.uuid4()),
                        t0=t0,
                        t1=max(t1, t0 + 1),
                        label=anomaly_type or edge_result.get("top_anomaly_prompt", "Anomaly"),
                        score=float(avg_score),
                        severity=severity,
                        geminiVerdict=final_is_anomaly,
                        geminiConfidence=final_confidence,
                        geminiExplanation=explanation,
                        keyframeUrls=[]
                    )))
        
        # Get video duration
        video_duration = frames[-1]["timestamp"] if frames else 0
        
        return {
            "jobId": job.get("id", ""),
            "status": "completed",
            "videoUrl": video_url,
            "videoDuration": video_duration,
            "events": events,
            "prompt_banks": {
                "normal_prompts": normal_prompts,
                "anomaly_prompts": anomaly_prompts,
                "detection_summary": prompt_banks.get("detection_summary", "")
            },
            "edge_detection": {
                "is_anomaly": edge_result.get("is_anomaly", False),
                "score_diff": edge_result.get("score_diff", 0),
                "confidence": edge_result.get("confidence", 0),
                "top_anomaly_prompt": edge_result.get("top_anomaly_prompt", ""),
                "top_normal_prompt": edge_result.get("top_normal_prompt", ""),
                "per_frame_scores": edge_result.get("per_frame_scores", [])
            },
            "gemini_verification": gemini_result if "error" not in gemini_result else None,
            "final_verdict": {
                "is_anomaly": final_is_anomaly,
                "anomaly_type": anomaly_type,
                "confidence": final_confidence
            },
            "processedAt": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).decode().strip()
        }


# RunPod serverless handler
runpod.serverless.start({"handler": handler})
