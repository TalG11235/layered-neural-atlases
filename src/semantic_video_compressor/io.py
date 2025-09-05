import os, shutil
import cv2

def clear_previous_data():
    dirs_to_clear = [
        "results",
        "atlases",
        "compressed_atlases",
        "checkpoint",
        "compressed_editing_outputs"
    ]

    for dir_name in dirs_to_clear:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)


def separate_video_to_frames(file_path, video_name, frame_rate=10): 
    if os.path.exists(f'data/{video_name}'):
        print(f"Directory data/{video_name} already exists.")
        return

    if not os.path.exists(file_path):
        raise ValueError(f"Video file does not exist: {file_path}")

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {file_path}")

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval
    frame_interval = max(1, int(original_fps / frame_rate))
    
    frame_dir = f"data/{video_name}"
    os.makedirs(frame_dir, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Only save frames at the specified interval
        if frame_idx % frame_interval == 0:
            frame_filename = os.path.join(frame_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_idx += 1

    cap.release()