import os
import platform
import json
import hashlib
import shutil
import threading
import csv
from pathlib import Path
from datetime import datetime
from tkinter import Tk, Label, Button, Text, Scrollbar, filedialog, END, messagebox, ttk

def is_media_file(filename):
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.3gp', '.mpeg', '.mpg'}
    ext = Path(filename).suffix.lower()
    return ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS

def compute_sha256(filepath, block_size=65536):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(block_size):
            sha.update(chunk)
    return sha.hexdigest()

def get_all_media_files(paths):
    media_files = []
    for root in paths:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if is_media_file(filename):
                    media_files.append(os.path.join(dirpath, filename))
    return media_files

def get_file_info(filepath):
    stat = os.stat(filepath)
    return {
        "path": filepath,
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256": compute_sha256(filepath)
    }

def export_duplicates_to_csv(duplicates, filename="duplicates.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['SHA256 Hash', 'File Paths'])
        for h, paths in duplicates.items():
            writer.writerow([h, "; ".join(paths)])

def quarantine_duplicates(duplicates, log_fn):
    quarantine_folder = os.path.abspath("quarantine")
    os.makedirs(quarantine_folder, exist_ok=True)
    quarantine_map = {}
    moved = 0

    for h, paths in duplicates.items():
        for dup_path in paths[1:]:  # keep the first one
            try:
                dest = Path(quarantine_folder) / Path(dup_path).name
                shutil.move(dup_path, dest)
                quarantine_map[str(dest)] = dup_path
                moved += 1
                log_fn(f"Moved to quarantine: {dup_path}")
            except Exception as e:
                log_fn(f"Failed to move {dup_path}: {e}")

    with open("quarantine_map.json", "w", encoding='utf-8') as f:
        json.dump(quarantine_map, f, indent=2)

    return moved

def restore_quarantined_files(gui):
    try:
        with open("quarantine_map.json", "r", encoding='utf-8') as f:
            quarantine_map = json.load(f)
    except FileNotFoundError:
        gui.log("No quarantine_map.json found.")
        return

    restored = 0
    for current_path, original_path in quarantine_map.items():
        try:
            os.makedirs(os.path.dirname(original_path), exist_ok=True)
            shutil.move(current_path, original_path)
            gui.log(f"Restored: {original_path}")
            restored += 1
        except Exception as e:
            gui.log(f"Failed to restore {original_path}: {e}")

    gui.log(f"Total restored files: {restored}")


def run_scan(gui, folders):
    gui.log(f"Indexing files in: {folders}")
    media_files = get_all_media_files(folders)
    total_files = len(media_files)
    gui.progress_bar['maximum'] = total_files

    gui.log(f"Total media files to scan: {total_files}")
    images, videos, hash_map = [], [], {}

    for i, filepath in enumerate(media_files, 1):
        try:
            info = get_file_info(filepath)
            if Path(filepath).suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}:
                images.append(info)
            else:
                videos.append(info)
            hash_map.setdefault(info['sha256'], []).append(info['path'])
            gui.progress_bar['value'] = i
            gui.master.update_idletasks()
        except Exception as e:
            gui.log(f"Skipped {filepath}: {e}")

    duplicates = {h: p for h, p in hash_map.items() if len(p) > 1}

    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_images": len(images),
            "total_videos": len(videos),
            "total_files": total_files,
            "duplicate_groups": len(duplicates)
        },
        "images": images,
        "videos": videos,
        "duplicates": duplicates
    }

    with open("media_files_with_duplicates.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    export_duplicates_to_csv(duplicates)
    gui.log("\nExported JSON and CSV reports.")

    if duplicates:
        moved_count = quarantine_duplicates(duplicates, gui.log)
        gui.log(f"Moved {moved_count} duplicate files to quarantine/")

    gui.log(f"\nImages: {len(images)}")
    gui.log(f"Videos: {len(videos)}")
    gui.log(f"Duplicate groups: {len(duplicates)}")
    gui.scan_button.config(state='normal')

class MediaScannerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Media File Scanner with Duplicates")
        master.geometry("800x600")

        Label(master, text="Select a folder to scan for images/videos and duplicates:").pack(pady=10)

        self.select_button = Button(master, text="Select Folder and Scan", command=self.select_and_scan)
        self.select_button.pack()

        self.scan_button = Button(master, text="Scan Entire System (⚠️ slow)", command=self.scan_whole_system)
        self.scan_button.pack(pady=5)

        self.restore_button = Button(master, text="Restore Quarantined Files", command=lambda: threading.Thread(target=restore_quarantined_files, args=(self,), daemon=True).start())
        self.restore_button.pack(pady=5)

        self.progress_bar = ttk.Progressbar(master, orient='horizontal', length=700, mode='determinate')
        self.progress_bar.pack(pady=5)

        self.text_area = Text(master, wrap='word', height=20, width=100)
        self.scroll = Scrollbar(master, command=self.text_area.yview)
        self.text_area.config(yscrollcommand=self.scroll.set)
        self.text_area.pack(pady=10)
        self.scroll.pack(side='right', fill='y')

        self.open_button = Button(master, text="Open JSON Result", command=self.open_json)
        self.open_button.pack(pady=5)

    def log(self, message):
        self.text_area.insert(END, message + "\n")
        self.text_area.see(END)
        self.master.update_idletasks()

    def select_and_scan(self):
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            self.text_area.delete(1.0, END)
            self.scan_button.config(state='disabled')
            threading.Thread(target=run_scan, args=(self, [folder]), daemon=True).start()

    def scan_whole_system(self):
        paths = self.get_all_mount_points()
        self.text_area.delete(1.0, END)
        self.scan_button.config(state='disabled')
        threading.Thread(target=run_scan, args=(self, paths), daemon=True).start()

    def get_all_mount_points(self):
        system = platform.system()
        mounts = []
        if system == 'Windows':
            import win32file
            for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                path = f'{d}:\\'
                try:
                    dtype = win32file.GetDriveType(path)
                    if dtype in {win32file.DRIVE_REMOVABLE, win32file.DRIVE_FIXED}:
                        mounts.append(path)
                except: continue
        else:
            try:
                with open('/proc/mounts', 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts[1].startswith(('/media', '/mnt', '/run/media', '/Volumes', '/home')):
                            mounts.append(parts[1])
            except:
                pass
            mounts.append('/')
        return list(set(mounts))

    def open_json(self):
        path = os.path.abspath("media_files_with_duplicates.json")
        if os.path.exists(path):
            if platform.system() == 'Windows':
                os.startfile(path)
            else:
                os.system(f"xdg-open '{path}'")
        else:
            messagebox.showerror("File Not Found", "Run a scan to generate the JSON file first.")

if __name__ == '__main__':
    root = Tk()
    app = MediaScannerGUI(root)
    root.mainloop()
