ByteTrack with Hungarian Matching Algorithm and Enhanced Kalman Filter
Goktug Can Simay, Ali Dogan, Emre Emin Osman

If you do not want to get any errors, we suggest you to open this link and directly run the existing code blocks step by step on Colab: https://colab.research.google.com/drive/1UYf_YsqSb9bBBeWsVuMT9Wj9vLTUIORH

You can also create new notebook for you and try our project or process by running the 1.ipynb file in 1.zip by following the specified steps.  

Please follow the steps detailed below. Run the code blocks in the steps. DO NOT go directly downwards.

1. Step: Checking whether you have GPU support or not.

```shell
!nvidia-smi
```

2. Step: Importing the OS module to perform the necessary operations with the operating system. And assigning the current working directory to the HOME variable.

```shell
import os
HOME = os.getcwd()
print(HOME)
```

3. Step: If you do not have Kaggle API, please skip steps 3, 4, 5, 6 and 7. And do this: Download the road_traffic_clip.mp4 file in 1.zip to your computer. Upload the road_traffic_clip.mp4 file to the "Files" tab, without clicking on any directory, by simply right clicking and clicking "Upload". (If you have the API and still have trouble downloading the dataset, follow this step.)

```shell
!pip install kaggle --upgrade --quiet
```

4. Step: Importing the getpass module used to get the user's information.

```shell
import os
from getpass import getpass
```

5. Step: Getting the Kaggle username and API from the user.

```shell
os.environ['KAGGLE_USERNAME'] = getpass('Enter KAGGLE_USERNAME secret value: ')
os.environ['KAGGLE_KEY'] = getpass('Enter KAGGLE_KEY secret value: ')
```

6. Step: Downloading clips of the DFL Bundesliga Data Shootout competition from Kaggle.

```shell
!kaggle competitions files -c dfl-bundesliga-data-shootout | grep clips | head -10
```

7. Step: 

```shell
import os
def download_clips(home_dir):
    """Bundesliga verilerinden 20 en son klip dosyasını indirir."""

    #Kaggle'dan klip dosyalarının adlarını indir.
    output = os.popen(
        "kaggle competitions files -c dfl-bundesliga-data-shootout | grep clips | head -20 | awk '{print $1}'"
    ).read()
    files = output.splitlines()

    #Her bir klip dosyasını indir.
    for file in files:
        # Dosyanın adını oluştur.
        file_name = file + ".zip"

        #Dosyayı indir.
        os.system(f"kaggle competitions download -c dfl-bundesliga-data-shootout -f {file_name} -d {home_dir}")

if __name__ == "__main__":
    home_dir = os.path.expanduser("~")

    #Klipleri indir.
    download_clips(home_dir)
```

8. Step: Installing YOLO.

```shell
!pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
```

9. Step: Cloning main version of ByteTrack from author's GitHub repo.

```shell
%cd {HOME}
!git clone https://github.com/ifzhang/ByteTrack.git
%cd {HOME}/ByteTrack

# workaround related to https://github.com/roboflow/notebooks/issues/80
!sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt

!pip3 install -q -r requirements.txt
!python3 setup.py -q develop
!pip install -q cython_bbox
!pip install -q onemetric
# workaround related to https://github.com/roboflow/notebooks/issues/112 and https://github.com/roboflow/notebooks/issues/106
!pip install -q loguru lap thop

from IPython import display
display.clear_output()


import sys
sys.path.append(f"{HOME}/ByteTrack")


import yolox
print("yolox.__version__:", yolox.__version__)
```

10. Step: 

```shell
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
```

11. Step: Making object tracking easier by downloading the new supervision library.

```shell
!pip install supervision==0.1.0


from IPython import display
display.clear_output()


import supervision
print("supervision.__version__:", supervision.__version__)
```

12. Step: 

```shell
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
```

13. Step: Calling some functions for tracking.

```shell
from typing import List

import numpy as np


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids
```

14. Step: Fetching a pre-trained model of YOLO.

```shell
MODEL = "yolov8x.pt"
```

15. Step: 

```shell
from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()
```

16. Step: Defining dictionary ID according to the types of vehicles in traffic.

```shell
# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]
```

17. Step: 

```shell
SOURCE_VIDEO_PATH = f"{HOME}/road_traffic_clip.mp4"
```

18. Step: Note the speed, preprocessing, postprocessing time. 

```shell
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)
# acquire first video frame
iterator = iter(generator)
frame = next(iterator)
# model prediction on single frame and conversion to supervision Detections
results = model(frame)
detections = Detections(
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
)
# format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections
]
# annotate and display frame
frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

%matplotlib inline
show_frame_in_notebook(frame, (16, 16))
```

19. Step: Determining own line and moving objects accordingly.

```shell
# settings
LINE_START = Point(250, 250)
LINE_END = Point(400, 200)

TARGET_VIDEO_PATH = f"{HOME}/road_traffic_clip_result.mp4"
```

20. Step: Do what was done above for just one frame and run it for the entire video, this time by putting it in a for loop.

```shell
VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
```

21. Step: 

```shell
from tqdm.notebook import tqdm


# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create LineCounter instance
line_counter = LineCounter(start=LINE_START, end=LINE_END)
# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)
line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

# open target video file
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # loop over video frames
    for frame in tqdm(generator, total=video_info.total_frames):
        # model prediction on single frame and conversion to supervision Detections
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # format custom labels
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        # updating line counter
        line_counter.update(detections=detections)
        # annotate and display frame
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        sink.write_frame(frame)
```

22. Step: Cloning improved version of ByteTrack from Goktug Can Simay's GitHub repo. Our optimized byte_tracker.py and kalman_filter.py will run this time.

```shell
#ByteTrack indir. (IMPROVED)

%cd {HOME}
!git clone https://github.com/simaygoktug/mot.git
%cd {HOME}/mot

# workaround related to https://github.com/roboflow/notebooks/issues/80
!sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt

!pip3 install -q -r requirements.txt
!python3 setup.py -q develop
!pip install -q cython_bbox
!pip install -q onemetric
# workaround related to https://github.com/roboflow/notebooks/issues/112 and https://github.com/roboflow/notebooks/issues/106
!pip install -q loguru lap thop

from IPython import display
display.clear_output()


import sys
sys.path.append(f"{HOME}/mot")


import yolox
print("yolox.__version__:", yolox.__version__)
```

23. Step: 

```shell
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
```

24. Step: Repeat the same procedure for a single frame with improved version. NOTE THE SPEED, PREPROCESSING AND POSTPROCESSING TIME. You can check the gain on speed. 

```shell
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)
# acquire first video frame
iterator = iter(generator)
frame = next(iterator)
# model prediction on single frame and conversion to supervision Detections
results = model(frame)
detections = Detections(
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
)
# format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections
]
# annotate and display frame
frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

%matplotlib inline
show_frame_in_notebook(frame, (16, 16))
```

25. Step: Repeat the same procedure for the whole video with improved version. NOTE THE SPEED, PREPROCESSING AND POSTPROCESSING TIME. You can check the gain on speed. 

```shell
from tqdm.notebook import tqdm


# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create LineCounter instance
line_counter = LineCounter(start=LINE_START, end=LINE_END)
# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)
line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

# open target video file
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # loop over video frames
    for frame in tqdm(generator, total=video_info.total_frames):
        # model prediction on single frame and conversion to supervision Detections
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # format custom labels
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        # updating line counter
        line_counter.update(detections=detections)
        # annotate and display frame
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        sink.write_frame(frame)
```

26. Step: You can download the road_traffic_clip_result.mp4 from "Files" tab to your computer and check the results. 
