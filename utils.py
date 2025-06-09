import os
import json
import cv2
import face_recognition

import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PyQt5.QtWidgets import QApplication, QFileDialog, QListView, QTreeView, QAbstractItemView
import sys

def select_folders():
    """
    Open a PyQt5 dialog to select one or more folders (including drives like C:/).
    """
    app = QApplication(sys.argv)
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setDirectory("C:/")  # Default to root drive
    dialog.setWindowTitle("Select one or more folders or drives")
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)

    # Hack to enable multi-folder selection
    list_view = dialog.findChild(QListView, "listView")
    tree_view = dialog.findChild(QTreeView)
    if list_view:
        list_view.setSelectionMode(QAbstractItemView.MultiSelection)
    if tree_view:
        tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

    if dialog.exec_():
        folders = dialog.selectedFiles()
        print("Selected folders:", folders)
        return folders
    else:
        print("No folders selected.")
        sys.exit(1)


def load_known_people(known_folder):
    """Load known face encodings from images in a given folder."""
    known_encodings = []
    known_names = []
    for filename in os.listdir(known_folder):
        filepath = os.path.join(known_folder, filename)
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def extract_metadata(video_path):
    """Extract basic metadata from video using OpenCV."""
    metadata = {}
    cap = cv2.VideoCapture(video_path)
    metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
    metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata['duration'] = metadata['frame_count'] / metadata['fps'] if metadata['fps'] > 0 else 0
    cap.release()
    return metadata

def process_file(file_path, known_encodings, known_names, thresholds, yolo_model_path, output_folder, use_gpu):
    """A wrapper that calls either image or video processing."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        # Call your image processing logic here
        pass  # Placeholder
    elif ext in ['.mp4', '.avi', '.mov']:
        from face_obj_detection import process_video
        return process_video((file_path, known_encodings, known_names, thresholds, yolo_model_path, output_folder, use_gpu))
    else:
        print(f"Unsupported file format: {ext}")
        return None
