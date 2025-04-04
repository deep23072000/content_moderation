from django.shortcuts import render, redirect,get_object_or_404
from .models import Video
from .forms import VideoForm
from django.conf import settings
from django.http import HttpResponse
from django.core.files.storage import default_storage
import os
import cv2
import numpy as np
import ffmpeg


def home(request):
    return render(request, 'home.html')

def video_list(request):
    videos = Video.objects.all()
    return render(request, 'video_list.html', {'videos': videos})

def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('video_list')
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})

def extract_frames(video_path, video_id):
    output_dir = os.path.join(settings.MEDIA_ROOT, f"frames/{video_id}")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return [f"frames/{video_id}/frame_{i}.jpg" for i in range(frame_count)]

# Extract audio using FFmpeg
def extract_audio(video_path, video_id):
    output_audio = os.path.join(settings.MEDIA_ROOT, f"audio/{video_id}.mp3")
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)

    (
        ffmpeg
        .input(video_path)
        .output(output_audio, format="mp3", acodec="libmp3lame")
        .run(overwrite_output=True)
    )

    return f"audio/{video_id}.mp3"

# Preprocess video and show results
def preprocess_video(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    video_path = video.file.path

    # Extract frames and audio
    frames = extract_frames(video_path, video_id)
    audio_path = extract_audio(video_path, video_id)

    return render(request, "preprocess_result.html", {"video": video, "frames": frames, "audio": audio_path})


