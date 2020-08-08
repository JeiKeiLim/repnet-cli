import cv2
from repnet import get_counts, create_count_video
from tqdm import tqdm
import os
import argparse
import time
from utils import get_model, read_video
import numpy as np
import matplotlib.pyplot as plt


def inference(model, args, imgs):
    s_time = time.time()
    (pred_period, pred_score, within_period,
     per_frame_counts, chosen_stride) = get_counts(
        model,
        imgs,
        strides=[1, 2, 3, 4],
        batch_size=20,
        threshold=args.threshold,
        within_period_threshold=args.in_threshold,
        constant_speed=args.constant_speed,
        median_filter=args.median_filter,
        fully_periodic=args.fully_periodic)
    infer_time = time.time()-s_time

    return pred_period, pred_score, within_period, per_frame_counts, chosen_stride, infer_time


def process_video_file(model, args):
    rot_dicts = {"none": None,
                 "cw": cv2.ROTATE_90_CLOCKWISE,
                 "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
                 "180": cv2.ROTATE_180}

    if os.path.isdir(args.path):
        _, _, files = next(os.walk(args.path))
        paths = [f"{args.path}/{file}" for file in files]
        outs = [f"{args.out}/{file[:-4]}.mp4" for file in files]
        os.makedirs(args.out, exist_ok=True)
    else:
        paths = [args.path]
        outs = [args.out]

    for path, out in tqdm(zip(paths, outs), desc="Processing ..."):
        imgs, vid_fps = read_video(path, rot=rot_dicts[args.rot])

        (pred_period, pred_score, within_period,
         per_frame_counts, chosen_stride, infer_time) = inference(model, args, imgs)
        print(f"Inference time: {infer_time:.02f}s. Visualizing ...")
        create_count_video(imgs, per_frame_counts, within_period, score=pred_score, fps=vid_fps, output_file=out,
                           delay=1000/vid_fps, plot_count=True, plot_within_period=True, plot_score=True, vizualize_reps=args.viz_reps)


def process_webcam(model, args, width=224, height=224):
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    win_size = int(args.win_size * fps)
    strides = int(win_size * args.stride_ratio)

    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (width, height))

            frames.append(frame_rgb)

            if len(frames) == win_size:
                imgs = np.asarray(frames)
                (pred_period, pred_score, within_period,
                 per_frame_counts, chosen_stride, infer_time) = inference(model, args, imgs)

                plt.plot(np.cumsum(per_frame_counts))
                plt.show()

                while len(frames) > (win_size-strides):
                    frames.pop(0)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("window", frame_bgr)

            key_in = cv2.waitKey(25) & 0xFF

            if key_in == ord('q'):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, help="Input video file path or root. If 0 is given, it runs on webcam mode(EXPERIMENTAL)")
    parser.add_argument("--out", default="./export", help="Output video file path or root")
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
    parser.add_argument("--win-size", default=10, type=int, help="Window size for webcam mode")
    parser.add_argument("--stride-ratio", default=0.5, type=float, help="Window stride ratio respect to win-size for webcam mode")
    main_args = parser.parse_args()

    repnet_model = get_model(main_args.ckpt)

    if main_args.path == "0":
        process_webcam(repnet_model, main_args)
    else:
        process_video_file(repnet_model, main_args)
