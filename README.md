# RepNet python-CLI
RepNet python-CLI is command line interface implementation of RepNet - https://sites.google.com/view/repnet

# Environments
Tested on `python 3.8` and `Tensorflow 2.3`.
Required packages can be installed by `pip install -r requirements.txt`

# Usage
## Single video input
```
python main.py input_video.mp4 --out output_video.mp4
```
<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/repnet/Jongkuk_20190106_195325_SQUAT_10_10.gif" />

## Batch video inputs
```
python main.py /videos/root --out /output/root
```

## Arguments
```
usage: main.py [-h] [--out OUT] [--ckpt CKPT] [--threshold THRESHOLD]
               [--in-threshold IN_THRESHOLD] [--constant-speed]
               [--median-filter] [--no-median-filter] [--fully-periodic]
               [--viz-reps] [--no-viz-reps] [--rot ROT] [--win-size WIN_SIZE]
               [--stride-ratio STRIDE_RATIO]
               path
positional arguments:
  path                  Input video file path or root. If 0 is given, it runs
                        on webcam mode(EXPERIMENTAL)
optional arguments:
  -h, --help            show this help message and exit
  --out OUT             Output video file path or root (default: ./export)
  --ckpt CKPT           Checkpoint weights root. (default: ./weights)
  --threshold THRESHOLD
                        RepNet threshold. (default: 0.2)
  --in-threshold IN_THRESHOLD
                        RepNet within period threshold. (default: 0.5)
  --constant-speed      RepNet constant speed parameter. (default: False)
  --median-filter       RepNet median filter parameter. (default: True)
  --no-median-filter    RepNet median filter parameter. (default: True)
  --fully-periodic      RepNet fully periodic parameter. (default: False)
  --viz-reps            Vizualitation repetition mode (default: True)
  --no-viz-reps         Vizualitation repetition mode (default: True)
  --rot ROT             Rotate videos. (none, cw, ccw, 180) (default: none)
  --win-size WIN_SIZE   Window size for webcam mode (default: 10)
  --stride-ratio STRIDE_RATIO
                        Window stride ratio respect to win-size for webcam
                        mode (default: 0.5)
```

# Failure case
|Good|Bad|
|---|---|
|<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/repnet/Jongkuk_20190106_200857_JUMPING_JACK_10_10.gif" />|<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/repnet/Jongkuk_20190106_200857_LUNGE_10_08.gif" />|
