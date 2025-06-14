import multiprocessing
import os
import json
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter
from ultralytics import YOLO
from utils import select_folders, load_known_people, process_file, extract_metadata, \
    select_folders_auto  # assumed available


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return set(json.load(f))
    return set()


def save_checkpoint(processed_files, checkpoint_path):
    with open(checkpoint_path, 'w') as f:
        json.dump(list(processed_files), f, indent=4)


def process_video(args):
    (video_path, known_encodings, known_names, thresholds, yolo_model_path, output_folder, use_gpu) = args

    faces_found = []
    objects_found = []
    object_counter = Counter()
    metadata = extract_metadata(video_path)

    try:
        yolo_model = YOLO(yolo_model_path)
        if use_gpu:
            yolo_model.to('cuda')

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = min(thresholds.get('max_frames_per_video', 30), frame_count)
        frame_interval = max(1, frame_count // frames_to_process)

        processed_frames = 0
        current_frame = 0

        while processed_frames < frames_to_process and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = []
                face_encodings = []

                for encoding in face_encodings:
                    matches = [np.linalg.norm(encoding - known) for known in known_encodings]
                    if matches and min(matches) < thresholds['face_match_threshold']:
                        matched_index = np.argmin(matches)
                        matched_name = known_names[matched_index]
                        faces_found.append(matched_name)
                    else:
                        faces_found.append("Unknown")

                results = yolo_model.predict(source=frame, conf=thresholds['yolo_conf_threshold'], verbose=False, device='cuda' if use_gpu else 'cpu')
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = yolo_model.model.names.get(cls_id, str(cls_id))
                        object_counter[cls_name] += 1
                        objects_found.append(cls_name)

                processed_frames += 1

            current_frame += 1

        cap.release()

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

    return {
        "file": os.path.basename(video_path),
        "full_path": video_path,
        "faces": faces_found,
        "objects": objects_found,
        "object_counts": dict(object_counter),
        "metadata": metadata,
        "type": "video"
    }


def process_file_wrapper(args):
    return process_file(*args)


def main():
    # Load configuration
    config = load_config('config.yaml')
    thresholds = config['thresholds']
    yolo_model_path = config['yolo_model_path']
    output_folder = config['output_folder']
    use_gpu = config.get('use_gpu', False)
    known_faces_folder = config.get('known_people_folder', 'known_faces')

    # Load known faces
    print("Loading known faces...")
    known_encodings, known_names = load_known_people(known_faces_folder)
    print(f"Loaded {len(known_encodings)} known faces.")

    # Select folders
    root_folders = select_folders_auto()

    print(f"-->Selected folders: {root_folders} <--")

    # Build image tasks
    image_tasks = []
    for folder in root_folders:
        print(f"--> analyzing folder: {folder}")
        for root, dirs, files in os.walk(folder):
            for file in files:
                #print(f'\t--> file: {file}')
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    full_path = os.path.join(root, file)
                    #print(f'\t\t--> path: {full_path}')
                    image_tasks.append((
                        full_path,
                        known_encodings,
                        known_names,
                        thresholds,
                        yolo_model_path,
                        output_folder,
                        use_gpu
                    ))

    print(f"Processing {len(image_tasks)} images...")

    # Parallel processing
    with multiprocessing.Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(process_file_wrapper, image_tasks),
            total=len(image_tasks),
            desc="Images"
        ):
            pass  # Handle results here if needed

    print("Processing completed.")


if __name__ == "__main__":
    main()
