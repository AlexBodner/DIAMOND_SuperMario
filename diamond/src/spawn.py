import argparse
from pathlib import Path
import os
import random
import h5py
import numpy as np
import torchvision.transforms.functional as T
from PIL import Image
import cv2

low_res_w = 256//4 #64
low_res_h = 240//4 

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"full_res_directory",
		type=Path,
		help="Specify your full_res directory.",
	)
	
	parser.add_argument(
		"model_directory",
		type=Path,
		help="Specify ",
	)
	return parser.parse_args()

def main():
	args = parse_args()
	
	full_res_directory = args.full_res_directory.absolute()
	model_directory = args.model_directory.absolute()
	
	spawn_dir = model_directory / "csgo/spawn"
	existing_spawns = len(os.listdir(spawn_dir))
	os.makedirs(spawn_dir / str(existing_spawns), exist_ok=True)
	full_res = os.path.join(full_res_directory, random.choice(os.listdir(full_res_directory)))
	h5file = h5py.File(full_res, 'r')	
	i = random.randint(0, 799)

	data_x_frames = []
	data_y_frames = []
	next_act_frames = []
	low_res_frames = []
	
	for j in range(4):
		frame_x = f'frame_{i}_x' # frame of observation
		frame_y = f'frame_{i}_y' # frame of action
		
		# Check if the datasets exist in the file
		if frame_x in h5file and frame_y in h5file:
			# Append each frame to the lists
			data_x = h5file[frame_x][:]
			data_y = h5file[frame_y][:]
			data_x = cv2.cvtColor(data_x, cv2.COLOR_BGR2RGB)
			
			data_x_frames.append(data_x)
			data_y_frames.append(data_y)
			
			# Resize the data_x frame to low resolution and append to low_res_frames
			image = Image.fromarray(data_x)
			image.save(spawn_dir / f"{existing_spawns}/full_res_{j}.png")
			print(f"{j}: {data_y}")
			resized_image = T.resize(image, (low_res_h, low_res_w), interpolation=T.InterpolationMode.BICUBIC)
			low_res_frames.append(np.array(resized_image))
		else:
			print(f"One or both of {frame_x} or {frame_y} do not exist in the file.")
		i += 1
	for _ in range(200):
		next_act = f'frame_{i}_y'
		if next_act in h5file:
			next_act_data = h5file[next_act][:]
			next_act_frames.append(next_act_data)

	data_x_stacked = np.stack(data_x_frames)
	data_y_stacked = np.stack(data_y_frames)
	next_act_stacked = np.stack(next_act_frames)
	low_res_stacked = np.stack(low_res_frames)
	
	low_res_stacked = np.transpose(low_res_stacked, (0, 3, 1, 2))
	data_x_stacked = np.transpose(data_x_stacked, (0, 3, 1, 2))
	
	
	print(f"Saving act.npy of size {data_y_stacked.shape}")
	np.save(spawn_dir / f"{existing_spawns}/act.npy", data_y_stacked)
	print(f"Saving full_res.npy of size {data_x_stacked.shape}")
	np.save(spawn_dir / f"{existing_spawns}/full_res.npy", data_x_stacked)
	print(f"Saving next_act.npy of size {next_act_stacked.shape}")
	np.save(spawn_dir / f"{existing_spawns}/next_act.npy", next_act_stacked)
	print(f"Saving low_res.npy of size {low_res_stacked.shape}")
	np.save(spawn_dir / f"{existing_spawns}/low_res.npy", low_res_stacked)
	
	h5file.close()
	


if __name__ == "__main__":
	main()