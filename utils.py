import os
import cv2
import numpy as np

# extract frames from video
def extract_frames(video_path, frames_dir, n_max=100, overwrite=False):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    else:
        if not overwrite and len(os.listdir(frames_dir)) > 0:
            print('frames already extracted at {}. Set overwrite=True to overwrite'.format(frames_dir))
            return

    # initialize a VideoCapture object to read video data into a numpy array
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames: {}'.format(frame_count))

    for i in range(1, frame_count+1):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(frames_dir, f'{i:04d}.jpg'), frame)
        else:
            print('frame {} could not be read'.format(i))
            break
        if i == n_max + 1:
            print("read {} frames".format(n_max))
            break

    cap.release()
    print("frames extracted at {}".format(frames_dir))

if __name__ == "__main__":
    video_path = '/home/william/extdisk/data/Lane_Detection_Result/ref/19700101_002021-20250213_163412.mp4'
    frames_dir = '/home/william/extdisk/data/Lane_Detection_Result/ref/19700101_002021-20250213_163412'
    extract_frames(video_path, frames_dir, overwrite=True)