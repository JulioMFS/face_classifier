import os
import cv2
import json
import yaml
import tkinter as tk
from tkinter import filedialog
import face_recognition
from ultralytics import YOLO
from PIL import Image, ExifTags
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_known_people(folder):
    known_encodings = []
    known_names = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file)[0])
    return known_encodings, known_names

def extract_image_metadata(image_path):
    metadata = {}
    try:
        img = Image.open(image_path)
        exif = {
            ExifTags.TAGS.get(k, k): v
            for k, v in img._getexif().items()
        } if img._getexif() else {}
        # GPS
        gps_info = exif.get("GPSInfo")
        if gps_info:
            def convert_to_degrees(value):
                d, m, s = value
                return d[0] / d[1] + (m[0] / m[1]) / 60 + (s[0] / s[1]) / 3600
            lat = convert_to_degrees(gps_info[2]) * (-1 if gps_info[3] == 'S' else 1)
            lon = convert_to_degrees(gps_info[4]) * (-1 if gps_info[5] == 'W' else 1)
            metadata['gps'] = {'lat': lat, 'lon': lon}
        # Timestamp
        timestamp = exif.get("DateTimeOriginal") or exif.get("DateTime")
        if timestamp:
            metadata['timestamp'] = datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S").isoformat()
        # Camera Model
        if 'Model' in exif:
            metadata['camera_model'] = exif['Model']
    except Exception as e:
        print(f"Could not extract metadata from {image_path}: {e}")
    return metadata

def analyze_image(image_path, known_encodings, known_names, yolo_model, thresholds, output_folder):
    print(f"Analyzing image: {image_path}")
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces_found = []
    objects_found = []

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = distances.argmin() if len(distances) > 0 else -1
        name = "Unknown"
        distance = None
        if best_match_index >= 0 and distances[best_match_index] < thresholds['face_match_threshold']:
            name = known_names[best_match_index]
            distance = float(distances[best_match_index])
        faces_found.append({"name": name, "distance": distance})
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    results = yolo_model.predict(image, conf=thresholds['yolo_conf_threshold'])
    object_counter = Counter()
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        names = r.names
        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = [int(i) for i in box]
            label = names[int(cls)]
            object_counter[label] += 1
            objects_found.append({"label": label, "confidence": float(conf)})
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

    metadata = extract_image_metadata(image_path)

    return {
        "file": os.path.basename(image_path),
        "faces": faces_found,
        "objects": objects_found,
        "object_counts": dict(object_counter),
        "metadata": metadata
    }

def analyze_video(video_path, known_encodings, known_names, yolo_model, thresholds, output_folder):
    print(f"Analyzing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    video_results = {
        "file": os.path.basename(video_path),
        "faces": [],
        "objects": [],
        "object_counts": {},
        "metadata": {}
    }

    frame_idx = 0
    object_counter = Counter()
    unique_faces = {}
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= thresholds['max_frames_per_video']:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = distances.argmin() if len(distances) > 0 else -1
            name = "Unknown"
            distance = None
            if best_match_index >= 0 and distances[best_match_index] < thresholds['face_match_threshold']:
                name = known_names[best_match_index]
                distance = float(distances[best_match_index])
            unique_faces[name] = distance
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        results = yolo_model.predict(frame, conf=thresholds['yolo_conf_threshold'])
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            names = r.names
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = [int(i) for i in box]
                label = names[int(cls)]
                object_counter[label] += 1
                video_results["objects"].append({"label": label, "confidence": float(conf)})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Save keyframes
        keyframe_name = os.path.splitext(os.path.basename(video_path))[0] + f"_frame{frame_idx}.jpg"
        cv2.imwrite(os.path.join(output_folder, keyframe_name), frame)
        frame_idx += 1

    cap.release()

    # Video metadata
    video_results["faces"] = [{"name": k, "distance": v} for k, v in unique_faces.items()]
    video_results["object_counts"] = dict(object_counter)
    video_results["metadata"] = {
        "frame_count": frame_idx,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "duration_sec": round(frame_idx / cap.get(cv2.CAP_PROP_FPS), 2) if cap.get(cv2.CAP_PROP_FPS) else None
    }

    return video_results

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    folder_selected = filedialog.askdirectory(title="Select Folder to Analyze")
    return folder_selected

def save_checkpoint(processed_files, checkpoint_path):
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(list(processed_files), f)
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                processed = json.load(f)
                return set(processed)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    return set()

def process_file(file_path, known_encodings, known_names, yolo_model, thresholds, output_folder):
    try:
        print(f"Started processing: {file_path}")
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            result = analyze_image(file_path, known_encodings, known_names, yolo_model, thresholds, output_folder)
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            result = analyze_video(file_path, known_encodings, known_names, yolo_model, thresholds, output_folder)
        else:
            print(f"Unsupported file type: {file_path}")
            return None
        print(f"Finished processing: {file_path}")
        return result
    except Exception as e:
        print(f"Exception processing {file_path}:\n{traceback.format_exc()}")
        return None

def main():
    config = load_config("config.yaml")
    known_people_folder = config.get('known_people_folder', 'known_people')
    output_folder = config.get('output_folder', 'output')
    thresholds = {
        'face_match_threshold': config.get('face_match_threshold', 0.6),
        'yolo_conf_threshold': config.get('yolo_conf_threshold', 0.5),
        'max_frames_per_video': config.get('max_frames_per_video', 30)
    }

    os.makedirs(output_folder, exist_ok=True)

    picked_folder = select_folder()
    if not picked_folder:
        print("No folder selected. Exiting.")
        return

    known_encodings, known_names = load_known_people(known_people_folder)
    print(f"Loaded {len(known_encodings)} known faces.")

    yolo_model = YOLO('yolov8n.pt')

    checkpoint_path = os.path.join(output_folder, "checkpoint.json")
    processed_files = load_checkpoint(checkpoint_path)

    all_files = []
    for root_dir, _, files in os.walk(picked_folder):
        for file in files:
            all_files.append(os.path.join(root_dir, file))

    print(f"Total files found: {len(all_files)}")

    # Filter out already processed files
    files_to_process = [f for f in all_files if os.path.basename(f) not in processed_files]

    print(f"Files to process after checkpoint: {len(files_to_process)}")
    for f in files_to_process:
        print(f"  -> {f}")

    all_results = []

    max_workers = 1  # Set to 1 for debugging; increase for more threads later
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, path, known_encodings, known_names, yolo_model, thresholds, output_folder)
            for path in files_to_process
        ]

        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Exception in future: {e}")
                    result = None

                if result:
                    all_results.append(result)
                    processed_files.add(os.path.basename(result["file"]))
                    save_checkpoint(processed_files, checkpoint_path)

                pbar.update(1)

    results_json_path = os.path.join(output_folder, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("Analysis complete. Results saved to", results_json_path)

if __name__ == "__main__":
    main()
