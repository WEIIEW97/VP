import os
import cv2
import numpy as np

import json


# extract frames from video
def extract_frames(video_path, frames_dir, n_max=100, overwrite=False):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    else:
        if not overwrite and len(os.listdir(frames_dir)) > 0:
            print(
                "frames already extracted at {}. Set overwrite=True to overwrite".format(
                    frames_dir
                )
            )
            return

    # initialize a VideoCapture object to read video data into a numpy array
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: {}".format(frame_count))

    for i in range(1, frame_count + 1):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(frames_dir, f"{i:04d}.jpg"), frame)
        else:
            print("frame {} could not be read".format(i))
            break
        if i == n_max + 1:
            print("read {} frames".format(n_max))
            break

    cap.release()
    print("frames extracted at {}".format(frames_dir))


def h265_to_frames(
    video_path, output_folder, frame_interval=1, warmup=0, n_max=None, jpeg_quality=95
):
    """
    Extract frames from H.265 video and save as JPG images using OpenCV.

    Args:
        video_path (str): Path to input H.265 video file.
        output_folder (str): Directory to save JPG frames.
        frame_interval (int): Save every Nth frame (1=every frame).
        warmup (int): Skip the first N frames.
        n_max (int): Maximum number of frames to save (None = no limit).
        jpeg_quality (int): JPEG quality (0-100).
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    saved_count = 0

    try:
        while True:
            if n_max is not None and saved_count >= n_max:
                break

            # Skip decoding if frame is not needed
            if frame_count < warmup or frame_count % frame_interval != 0:
                ret = cap.grab()  # Skip frame without decoding
                frame_count += 1
                continue

            ret, frame = cap.retrieve() if frame_count > 0 else cap.read()

            if not ret:
                break

            output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            saved_count += 1
            frame_count += 1

    finally:
        cap.release()

    print(f"Extracted {saved_count} frames from {frame_count} total frames")


def frames_to_video(
    image_path: str,
    video_path: str,
    fps: int = 24,
    fourcc: str = "mp4",
):
    images = [
        img
        for img in os.listdir(image_path)
        if img.endswith(".png") or img.endswith(".jpg")
    ]
    images.sort()

    frame = cv2.imread(os.path.join(image_path, images[0]))
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for image in images:
        frame = cv2.imread(os.path.join(image_path, image))
        out.write(frame)

    out.release()


def load_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def locate_indices(im_names: list):
    """
    files name is like ```frame_000100.jpg```
    """
    im_names = sorted(im_names)
    begin_file = im_names[0]
    second_file = im_names[1]
    end_file = im_names[-1]
    begin_index = int(begin_file.split("_")[-1].split(".")[0])
    second_index = int(second_file.split("_")[-1].split(".")[0])
    sep = second_index - begin_index
    end_index = int(end_file.split("_")[-1].split(".")[0])
    return begin_index, end_index, sep


if __name__ == "__main__":
    video_path = (
        "/home/william/extdisk/data/boximu-rgb/dataFromYF/data0731/zhuizi/20250731_135519_main.h265"
    )
    frames_dir = "/home/william/extdisk/data/boximu-rgb/dataFromYF/data0731/zhuizi/frames"
    h265_to_frames(video_path, frames_dir, frame_interval=10, warmup=100)

    # vis_path1 = "/home/william/extdisk/data/motorEV/19700101_002523/vis"
    # vis_path2 = "/home/william/extdisk/data/motorEV/19700101_002523/vis_smooth"

    # video_path1 = "/home/william/extdisk/data/motorEV/19700101_002523/vis.mp4"
    # video_path2 = "/home/william/extdisk/data/motorEV/19700101_002523/vis_smooth.mp4"
    # frames_to_video(vis_path1, video_path1, fps=24)
    # frames_to_video(vis_path2, video_path2, fps=24)
