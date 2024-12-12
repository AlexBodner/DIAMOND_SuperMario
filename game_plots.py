import cv2
import os
import matplotlib.pyplot as plt
from typing import List

def plot_high_low_res_frames(high_res_frames: List[str], low_res_frames: List[str], actions: List[str]):
    """
    Create a figure with high-res frames in the first row and corresponding low-res frames in the second row.

    Args:
        high_res_frames (List[str]): List of high-res image file paths.
        low_res_frames (List[str]): List of low-res image file paths.
        actions (List[str]): List of actions corresponding to each timestep.

    Returns:
        None
    """
    if len(high_res_frames) != len(low_res_frames) or len(high_res_frames) != len(actions):
        print("Error: Number of high-res frames, low-res frames, and actions must match.")
        return

    # Load high-res and low-res frames
    high_res_images = []
    low_res_images = []

    for path in high_res_frames:
        if not os.path.exists(path):
            print(f"Error: High-res frame file '{path}' does not exist.")
            return
        high_res_images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    for path in low_res_frames:
        if not os.path.exists(path):
            print(f"Error: Low-res frame file '{path}' does not exist.")
            return
        low_res_images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    # Create the figure
    fig, axes = plt.subplots(2, len(high_res_images), figsize=(15, 8), gridspec_kw={"height_ratios": [1, 1]})

    # Adjust layout
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.2, hspace=0.3)

    # Add row titles between rows
    for i in range(len(high_res_images)):
        # High-res image (first row)
        axes[0, i].imshow(high_res_images[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(actions[i], fontsize=10)

        # Low-res image (second row)
        axes[1, i].imshow(low_res_images[i])
        axes[1, i].axis('off')

    # Add centered row titles between the rows
    fig.text(0.5, 0.95, "Upsampler", va="center", ha="center", fontsize=14, weight="bold")
    fig.text(0.5, 0.5, "Denoiser", va="center", ha="center", fontsize=14, weight="bold")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

frames = [4,5,6,7,8]

# Plot high-res and low-res frames
high_res_frames = ["diamond/frames/frame_0004.png", "diamond/frames/frame_0005.png","diamond/frames/frame_0006.png","diamond/frames/frame_0007.png","diamond/frames/frame_0008.png"]
low_res_frames = ["diamond/frames/frame_low_res_0004.png", "diamond/frames/frame_low_res_0005.png", "diamond/frames/frame_low_res_0006.png","diamond/frames/frame_low_res_0007.png" ,"diamond/frames/frame_low_res_0008.png"]
actions = ["Action 1", "Action 2", "Action 2" , "Action 2" , "Action 2"]
plot_high_low_res_frames(high_res_frames, low_res_frames, actions)