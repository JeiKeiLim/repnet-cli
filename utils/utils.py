from repnet import get_repnet_model
from tqdm import tqdm
import cv2
import numpy as np
import requests
import os


def read_video(video_filename, width=224, height=224, rot=None):
    """Read video from file."""
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=n_frames, desc=f"Getting frames from {video_filename} ...")

    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (width, height))
            if rot:
                frame_rgb = cv2.rotate(frame_rgb, rot)
            frames.append(frame_rgb)

            pbar.update()
    pbar.close()
    frames = np.asarray(frames)
    return frames, fps


def wget(url, path):
    """
    Source from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    Args:
        url (str):
        path (str):

    Returns:

    """
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {url} to {path} ...")
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def get_model(weight_root):
    os.makedirs(weight_root, exist_ok=True)

    weight_urls = [
        "https://storage.googleapis.com/repnet_ckpt/checkpoint",
        "https://storage.googleapis.com/repnet_ckpt/ckpt-88.data-00000-of-00002",
        "https://storage.googleapis.com/repnet_ckpt/ckpt-88.data-00001-of-00002",
        "https://storage.googleapis.com/repnet_ckpt/ckpt-88.index"
    ]
    for url in weight_urls:
        path = f"{weight_root}/{url.split('/')[-1]}"
        if os.path.isfile(path):
            continue

        wget(url, path)

    return get_repnet_model(weight_root)
