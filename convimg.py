import os
import cv2
import numpy as np

# Set paths
INPUT_FOLDER = "processed_images"  # Folder containing original images
OUTPUT_FOLDER = "processed_test_img"  # Folder to save processed images

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process images
def process_and_save_images():
    for img_name in os.listdir(INPUT_FOLDER):
        img_path = os.path.join(INPUT_FOLDER, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        img_resized = cv2.resize(img, (64, 64))  # Resize to 64x64
        img_resized = img_resized.astype(np.float32) / 127.5 - 1  # Normalize to [-1,1]
        
        output_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(output_path, (img_resized + 1) * 127.5)  # Convert back to [0,255] for saving
        
        print(f"Processed: {img_name}")

if __name__ == "__main__":
    process_and_save_images()
    print("All images processed and saved.")