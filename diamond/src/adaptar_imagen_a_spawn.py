from PIL import Image
import numpy as np
import os
import cv2
import torchvision.transforms.functional as T

low_res_w = 256 // 4  # 64
low_res_h = 240 // 4

def process_image(image_path, output_dir, frame_count=4):
    # Load the image
    image = Image.open(image_path)
    
    # Resize the image to 256x240 (high resolution)
    resized_image = image.resize((256, 240), Image.Resampling.LANCZOS
)
    
    # Save full-resolution frames and generate low-resolution ones
    low_res_frames = []
    high_res_frames = []

    for j in range(frame_count):
        # Save full-resolution frames
        full_res_path = os.path.join(output_dir, f"full_res_{j}.png")
        resized_image.save(full_res_path)
        print(f"Saved full resolution frame {j} at {full_res_path}")
        
        # Create low-resolution frames
        low_res_image = T.resize(resized_image, (low_res_h, low_res_w), interpolation=T.InterpolationMode.BICUBIC)
        low_res_frames.append(np.array(low_res_image))
        high_res_frames.append(np.array(resized_image))

    # Stack low-resolution frames for saving
    low_res_stacked = np.stack(low_res_frames)
    low_res_stacked = np.transpose(low_res_stacked, (0, 3, 1, 2))  # Adjust to match PyTorch format
    
    high_res_stacked = np.stack(high_res_frames)
    high_res_stacked = np.transpose(high_res_stacked, (0, 3, 1, 2))  # Adjust to match PyTorch format
    
    # Save the stacked low-resolution frames
    low_res_output_path = os.path.join(output_dir, "low_res.npy")
    np.save(low_res_output_path, low_res_stacked)
    np.save( os.path.join(output_dir, "high_res.npy"), high_res_stacked)

    print(f"Saved low resolution frames at {low_res_output_path}")

# Example usage
if __name__ == "__main__":
    image_path = "breakout.png"  # Replace with your image path
    output_dir = "src/csgo/spawn"  # Replace with your output directory
    os.makedirs(output_dir, exist_ok=True)
    process_image(image_path, output_dir)