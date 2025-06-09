import os
import json
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count, Manager
from ultralytics import YOLO
from utils import select_folders, load_known_people, process_file, extract_metadata  # assumed available

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
        # Load YOLO model
        yolo_model = YOLO(yolo_model_path)
        if use_gpu:
            yolo_model.to('cuda')  # Force GPU

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
                # Face detection — placeholder (CPU unless you switch to a CUDA-compatible method)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = []
                face_encodings = []

                # Add your face recognition logic here (can switch to GPU with Dlib)

                for encoding in face_encodings:
                    matches = [np.linalg.norm(encoding - known) for known in known_encodings]
                    if matches and min(matches) < thresholds['face_match_threshold']:
                        matched_index = np.argmin(matches)
                        matched_name = known_names[matched_index]
                        faces_found.append(matched_name)
                    else:
                        faces_found.append("Unknown")

                # YOLO object detection — now using GPU
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

def main():
    # Load configuration
    config = load_config("config.yaml")
    known_people_folder = config.get('known_people_folder', 'known_people')
    output_folder = config.get('output_folder', 'output')
    thresholds = {
        'face_match_threshold': config.get('face_match_threshold', 0.6),
        'yolo_conf_threshold': config.get('yolo_conf_threshold', 0.5),
        'max_frames_per_video': config.get('max_frames_per_video', 30)
    }
    max_workers = config.get('max_workers', cpu_count())

    os.makedirs(output_folder, exist_ok=True)

    picked_folders = select_folders()
    if not picked_folders:
        print("No folders selected. Exiting.")
        return

    print(f"Selected folders: {picked_folders}")

    known_encodings, known_names = load_known_people(known_people_folder)
    print(f"Loaded {len(known_encodings)} known faces.")

    yolo_model_path = 'yolov8n.pt'

    checkpoint_path = os.path.join(output_folder, "checkpoint.json")
    processed_files = load_checkpoint(checkpoint_path)

    all_files = []
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    image_extensions = ('.jpg', '.jpeg', '.png')

    for folder in picked_folders:
        print(f'--> analyzing folder: {folder}')
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(image_extensions + video_extensions):
                    all_files.append(os.path.join(root, file))

    image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]

    all_results = []

    with Manager() as manager:
        processed_files_manager = manager.list(processed_files)

        use_gpu = True  # or False, configurable via config.yaml

        video_tasks = [
            (vid, known_encodings, known_names, thresholds, yolo_model_path, output_folder, use_gpu)
            for vid in video_files if os.path.basename(vid) not in processed_files
        ]

        for vid in video_files:
            if os.path.basename(vid) not in processed_files:
                video_tasks.append((vid, known_encodings, known_names, thresholds, yolo_model_path, output_folder))

        # Prepare image tasks
        image_tasks = []
        for img in image_files:
            if os.path.basename(img) not in processed_files:
                image_tasks.append((img, known_encodings, known_names, thresholds, yolo_model_path, output_folder))

        with Pool(processes=max_workers) as pool:
            # Images
            image_results = []
            if image_tasks:
                print(f"Processing {len(image_tasks)} images...")
                for result in tqdm(pool.imap_unordered(lambda args: process_file(*args), image_tasks), total=len(image_tasks), desc="Images"):
                    if result:
                        image_results.append(result)
                        processed_files_manager.append(os.path.basename(result["file"]))

            # Videos
            video_results = []
            if video_tasks:
                print(f"Processing {len(video_tasks)} videos...")
                for result in tqdm(pool.imap_unordered(process_video, video_tasks), total=len(video_tasks), desc="Videos"):
                    if result:
                        video_results.append(result)
                        processed_files_manager.append(os.path.basename(result["file"]))

        all_results.extend(image_results)
        all_results.extend(video_results)

        # Save checkpoint
        save_checkpoint(processed_files_manager, checkpoint_path)

    results_json_path = os.path.join(output_folder, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Analysis complete. Results saved to {results_json_path}")

if __name__ == "__main__":
    main()
