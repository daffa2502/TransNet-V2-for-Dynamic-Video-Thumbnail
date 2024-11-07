''' model available on https://github.com/soCzech/TransNetV2 '''

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import tensorflow as tf
import cv2
import requests
import tempfile
import os
from PIL import Image
import numpy as np
import hashlib
import threading

app = FastAPI()

class TransNetV2:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)

    def predict(self, frames):
        predictions = self.model(frames, training=False)
        return predictions[0].numpy()

model = TransNetV2("./transnetv2")
cache = {}
cache_lock = threading.Lock()

def download_video(url: str) -> str:
    """Downloads video from URL and returns the path to the downloaded file."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not download video: {e}")

    temp_video_path = tempfile.mktemp(suffix=".mp4")
    with open(temp_video_path, 'wb') as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)
    return temp_video_path

def get_video_frames(video_path: str):
    """Extracts frames from the video and returns them as a numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    frames = []
    success, frame = cap.read()
    while success:
        resized_frame = cv2.resize(frame, (48, 27))
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_resized_frame)
        success, frame = cap.read()
    cap.release()

    if not frames:
        raise ValueError("No frames were read from the video.")
    return np.array(frames, dtype=np.float32)

def detect_scenes(model, frames, batch_size=500):
    """Detects scene changes in the video frames."""
    predictions = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch = np.expand_dims(batch, 0)
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions.reshape(batch_predictions.shape[1]).tolist())
    
    scene_changes = [i for i, score in enumerate(predictions) if 1 / (1 + np.exp(-score)) > 0.5]
    return scene_changes

def create_gif(video_path: str, frame_indices, output_path='thumbnail.gif', duration=1000):
    """Creates an animated GIF from the specified frame indices."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()

    if frames:
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    else:
        raise ValueError("No frames selected for GIF creation.")

def generate_cache_key(video_url: str) -> str:
    """Generates a unique hash key for caching based on the video URL."""
    return hashlib.sha256(video_url.encode()).hexdigest()

@app.post("/generate-thumbnail")
async def generate_thumbnail(video_url: str):
    cache_key = generate_cache_key(video_url)

    # Check cache
    with cache_lock:
        if cache_key in cache:
            return FileResponse(cache[cache_key], media_type="image/gif")

    # Download and process video
    try:
        video_path = download_video(video_url)
    except HTTPException as e:
        return e

    try:
        frames = get_video_frames(video_path)
    except ValueError as e:
        os.remove(video_path)
        raise HTTPException(status_code=400, detail=f"Error processing video frames: {e}")

    try:
        scene_changes = detect_scenes(model, frames)
    except Exception as e:
        os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"Scene detection failed: {e}")

    # Determine key frames for GIF
    if len(scene_changes) <= 5:
        key_frames = [(i+1) * (len(frames) // 8) for i in range(7)]
    else:
        n_scene = min(8, len(scene_changes)) - 1
        differences = [(i, scene_changes[i+1] - scene_changes[i]) for i in range(len(scene_changes) - 1)]
        differences.sort(key=lambda x: x[1], reverse=True)
        differences = sorted([diff for diff in differences[:n_scene]])
        key_frames = [scene_changes[differences[i][0]] + (differences[i][1] // 2) for i in range(n_scene)]

    # Generate and cache the GIF
    gif_path = tempfile.mktemp(suffix=".gif")
    try:
        create_gif(video_path, key_frames, gif_path)
    except ValueError as e:
        os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"GIF creation failed: {e}")
    finally:
        os.remove(video_path)

    # Cache the GIF path and return it
    with cache_lock:
        cache[cache_key] = gif_path
    return FileResponse(gif_path, media_type="image/gif")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)