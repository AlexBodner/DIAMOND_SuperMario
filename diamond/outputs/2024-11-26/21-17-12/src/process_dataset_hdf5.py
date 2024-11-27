"""
IMPORTANT:
This script prepares data downloaded from the OneDrive link provided on the repo that introduced the dataset: https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/
=> Any issue related to the download of this data should be reported on the dataset repo linked above (NOT on DIAMOND's repo)

This script should be called with exactly 2 positional arguments:

- <tar_dir>: folder containing the .tar files from `dataset_dm_scraped_dust2_tars` folder on the OneDrive
- <out_dir>: a new dir (should not exist already), the script will untar and process data there
"""

import argparse
from functools import partial
from pathlib import Path
from multiprocessing import Pool
import shutil
import subprocess

import torch
import torchvision.transforms.functional as T
from tqdm import tqdm

from data.dataset import Dataset, CSGOHdf5Dataset
from data.episode import Episode
from data.segment import SegmentId

import os

low_res_w = 256//4 #64
low_res_h = 240//4 #60

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"in_dir",
		type=Path,
		help="Specify a directory containing a list of .hdf5 files.",
	)
	parser.add_argument(
		"out_dir",
		type=Path,
		help="Specify a directory where the dataset will be processed and saved.",
	)
	return parser.parse_args()

def main():
	args = parse_args()
	
	in_dir = args.in_dir.absolute()
	out_dir = args.out_dir.absolute()

	with Path("test_split.txt").open("r") as f:
		test_files = f.read().split("\n")

	full_res_dir = out_dir / "full_res"
	low_res_dir = out_dir / "low_res"
	
	os.makedirs(full_res_dir, exist_ok=True)
	#
	# Create low-res data
	#
	#cont = 0
	for file_name in os.listdir(in_dir):
		file_path = os.path.join(in_dir, file_name)
		print(file_path)
		
		if os.path.isfile(file_path):
			destination_path = os.path.join(full_res_dir, file_name)
			
			# Check if the file already exists in the destination directory
			if os.path.exists(destination_path):
				print(f"File {file_name} already exists in {full_res_dir}. Skipping...")
				continue
			
			# Copy the file if it doesn't already exist
			shutil.copy(file_path, full_res_dir)
		# cont += 1
		# if cont >= 4:
			# break
	
	csgo_dataset = CSGOHdf5Dataset(full_res_dir)

	train_dataset = Dataset(low_res_dir / "train", None)
	test_dataset = Dataset(low_res_dir / "test", None)
	original_size = None
	num_actions = 0

	for i in tqdm(csgo_dataset._filenames, desc="Creating low_res"):
		episode = Episode(
			**{
				k: v
				for k, v in csgo_dataset[SegmentId(i, 0, 1000)].__dict__.items()
				if k not in ("mask_padding", "id")
			}
		)
		
		if not original_size:
			original_size = episode.obs.size()
		print("shape",episode.obs.shape)
		episode.obs = T.resize(
			episode.obs, (low_res_h, low_res_w), interpolation=T.InterpolationMode.BICUBIC
		)
		num_actions = episode.act.size()
		filename = csgo_dataset._filenames[i]
		file_id = f"{filename.parent.stem}/{filename.name}"
		episode.info = {"original_file_id": file_id}
		dataset = test_dataset if filename.name in test_files else train_dataset
		dataset.add_episode(episode)

	train_dataset.save_to_default_path()
	test_dataset.save_to_default_path()

	print(
		f"Split train/test data ({train_dataset.num_episodes}/{test_dataset.num_episodes} episodes)\n"
	)

	print("You can now edit `config/env/csgo.yaml` and set:")
	print(f"path_data_low_res: {low_res_dir}")
	print(f"path_data_full_res: {full_res_dir}")
	print(f"Your FULL RES SIZE is {original_size} (format: [value, channel, HEIGHT, WIDTH] you only care about last 2), and your low_res size is W: {low_res_w} H:{low_res_h}")
	print("in config/env/csgo.yaml put your FULL RES SIZE [width, height];" + 
			"in config/agent/csgo.yaml set the upsampler.upsampling_factor to the factor to upsample from low_res to full_res.")
	print(f"Also your number of actions is {num_actions}")


if __name__ == "__main__":
	main()
