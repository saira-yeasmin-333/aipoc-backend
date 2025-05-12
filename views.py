import json
from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

import os
import torch
import clip
import cv2
from PIL import Image
import io
import torch.nn.functional as F
import base64

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


video_path = os.path.join(os.getcwd(), 'test_4.mp4')


def extract_frames(video_path, fps=1):
    vidcap = cv2.VideoCapture(video_path)
    total_fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(total_fps // fps)
    frames = []
    count = 0
    success, image = vidcap.read()
    while success:
        if count % interval == 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(image_rgb))
        success, image = vidcap.read()
        count += 1
    return frames

@csrf_exempt
def matched_frame(request):
    print('in matched frame')
    data = json.loads(request.body)
    text_query = data.get('query', '')

    print('text query:', text_query)
    print('video path:', video_path)

    if video_path is None:
        return JsonResponse({'error': 'No video path provided'}, status=400)
    # Extract frames
    frames = extract_frames(video_path)
    print('frames extracted size:', len(frames))
    if not frames:
        return JsonResponse({'error': 'No frames extracted'}, status=500)
    frame_embeddings = []

    for frame in frames:
        image_tensor = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        frame_embeddings.append(image_features)

    if not frame_embeddings:
        return JsonResponse({'error': 'No frames extracted'}, status=500)

    frame_embeddings_tensor = torch.cat(frame_embeddings, dim=0).to(device)
    text_tokens = clip.tokenize([text_query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = F.cosine_similarity(frame_embeddings_tensor, text_features)
    best_idx = similarities.argmax().item()
    print('best index:', best_idx)
    best_frame = frames[best_idx]

    buffer = io.BytesIO()
    best_frame.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return JsonResponse({'frame': img_base64})

@csrf_exempt
def upload_video(request):
    print('getting the call')
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        if video_file:
            print('in the api')
            video = request.FILES['video']
            video_path = os.path.join('', video.name)

            with open(video_path, 'wb+') as destination:
                for chunk in video.chunks():
                    destination.write(chunk)

            return JsonResponse({'message': 'Video uploaded successfully'})
        
        return JsonResponse({'error': 'No video uploaded'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def hello_world(request):
    return JsonResponse({'message': 'Hello, world!'})


@csrf_exempt  # disable CSRF for testing (not recommended in production)
def hello_api(request):
    print('in hello api')
    if request.method == 'POST':
        print('in hello api')
        return JsonResponse({'message': 'Hello from Django API!'})
    return JsonResponse({'error': 'Only POST allowed'}, status=405)
