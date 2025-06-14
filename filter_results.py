import json
import cv2
import os
import shutil

source_folder = 'output'
destination_folder = 'T:\\temporary'  # Double backslashes or raw string
os.makedirs(destination_folder, exist_ok=True)
# Load data from results.json
with open("output/results.json", "r") as f:
    data = json.load(f)

# Iterate over entries and find those with non-empty metadata or faces
for item in data:
    metadata = item.get("metadata", {})
    faces = item.get("faces", [])

    # Check if metadata or faces are non-empty
    if metadata or faces:
        image_name = item.get('file')
        print(f"File: {image_name}")

        # Print face names if faces exist
        if faces:
            face_names = [face.get("name", "Unknown") for face in faces]
            print("Faces:", ", ".join(face_names))
        else:
            print("Faces: None")

        # Print metadata
        if metadata:
            print("Metadata:", metadata)
        else:
            print("Metadata: None")

        print("-" * 40)


        # Set the image path
        output_folder = 'output'
        image_path = os.path.join(output_folder, image_name)

        # Read the image
        image = cv2.imread(image_path)

        # Check if image was loaded
        if image is None:
            print(f"Failed to load image from {image_path}")
        else:
            # Display the image
            # Target width
            target_width = 800

            # Get original dimensions
            original_height, original_width = image.shape[:2]

            # Compute scale ratio and new height
            scale_ratio = target_width / original_width
            new_height = int(original_height * scale_ratio)

            # Resize while keeping aspect ratio
            resized_image = cv2.resize(image, (target_width, new_height))

            # Show resized image
            #cv2.imshow('Resized Image', resized_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            src = os.path.join(source_folder, image_path)
            dst = os.path.join(destination_folder, image_path)
            shutil.copy2(image_path, destination_folder)
            print(f"Copied: {image_path}")