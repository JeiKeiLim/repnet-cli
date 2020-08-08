import cv2
import numpy as np
from repnet import get_repnet_model, get_counts, create_count_video
from tqdm import tqdm
import requests
import os
import argparse
import time


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


def process_video_file(repnet_model, main_args):
    rot_dicts = {"none": None,
                 "cw": cv2.ROTATE_90_CLOCKWISE,
                 "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
                 "180": cv2.ROTATE_180}

    if os.path.isdir(main_args.path):
        _, _, files = next(os.walk(main_args.path))
        paths = [f"{main_args.path}/{file}" for file in files]
        outs = [f"{main_args.out}/{file[:-4]}.mp4" for file in files]
        os.makedirs(main_args.out, exist_ok=True)
    else:
        paths = [main_args.path]
        outs = [main_args.out]

    for path, out in tqdm(zip(paths, outs), desc="Processing ..."):
        imgs, vid_fps = read_video(path, rot=rot_dicts[main_args.rot])

        s_time = time.time()
        (pred_period, pred_score, within_period,
         per_frame_counts, chosen_stride) = get_counts(
            repnet_model,
            imgs,
            strides=[1, 2, 3, 4],
            batch_size=20,
            threshold=main_args.threshold,
            within_period_threshold=main_args.in_threshold,
            constant_speed=main_args.constant_speed,
            median_filter=main_args.median_filter,
            fully_periodic=main_args.fully_periodic)
        print(f"Inference time: {time.time()-s_time:.02f}s. Visualizing ...")
        create_count_video(imgs, per_frame_counts, within_period, score=pred_score, fps=vid_fps, output_file=out,
                           delay=1000/vid_fps, plot_count=True, plot_within_period=True, plot_score=True, vizualize_reps=main_args.viz_reps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, help="Input video file path or root.")
    parser.add_argument("--out", default="./export", help="Output video file path or root")
    parser.add_argument("--mode", default="video", help="('video', 'webcam')")
    parser.add_argument("--ckpt", default="./weights", type=str, help="Checkpoint weights root.")
    parser.add_argument("--threshold", default=0.2, type=float, help="RepNet threshold.")
    parser.add_argument("--in-threshold", default=0.5, type=float, help="RepNet within period threshold.")
    parser.add_argument("--constant-speed", default=False, action='store_true', help="RepNet constant speed parameter.")
    parser.add_argument("--median-filter", dest="median_filter", default=True, action='store_true',
                        help="RepNet median filter parameter.")
    parser.add_argument("--no-median-filter", dest="median_filter", action='store_false',
                        help="RepNet median filter parameter.")
    parser.add_argument("--fully-periodic", default=False, action='store_true', help="RepNet fully periodic parameter.")
    parser.add_argument("--viz-reps", dest="viz_reps", default=True, action="store_true", help="Vizualitation repetition mode")
    parser.add_argument("--no-viz-reps", dest="viz_reps", action="store_false", help="Vizualitation repetition mode")
    parser.add_argument("--rot", default="none", type=str, help="Rotate videos. (none, cw, ccw, 180)")
    args = parser.parse_args()

    model = get_model(args.ckpt)

    if args.mode == "video":
        process_video_file(model, args)
