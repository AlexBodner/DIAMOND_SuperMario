<<<<<<< HEAD
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
import random

low_res_w = 64
low_res_h = 48


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"tar_dir",
		type=Path,
		help="folder containing the .tar files.",
	)
	parser.add_argument(
		"out_dir",
		type=Path,
		help="a new directory (should not exist already), the script will untar and process data there",
	)

	parser.add_argument("--fix", action="store_true", help="Use this to edit an existing dataset.")
	return parser.parse_args()


def process_tar(path_tar: Path, out_dir: Path, remove_tar: bool) -> None:
	d = out_dir
	d.mkdir(exist_ok=True, parents=True)
	shutil.move(path_tar, d)
	subprocess.run(f"cd {d} && tar -xzvf {path_tar.name}", shell=True)
	new_path_tar = d / path_tar.name
	if remove_tar:
		new_path_tar.unlink()
	else:
		shutil.move(new_path_tar, path_tar.parent)

import os
import shutil

def move_and_rename_files(parent_folder):
    # Get a list of all subdirectories in the parent folder
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            # Iterate over all files in the subfolder
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                
                # Check if it's a file
                if os.path.isfile(file_path):
                    # Create the new file name
                    new_file_name = f"{subfolder}_{file_name}"
                    new_file_path = os.path.join(parent_folder, new_file_name)
                    
                    # Move and rename the file
                    shutil.move(file_path, new_file_path)
                    print(f"Moved: {file_path} -> {new_file_path}")
            
            # Optionally, remove the now-empty subfolder
            os.rmdir(subfolder_path)
            print(f"Removed empty folder: {subfolder_path}")



def main():
	args = parse_args()

	tar_dir = args.tar_dir.absolute()
	out_dir = args.out_dir.absolute()

	if not tar_dir.exists():
		print(
			"Wrong usage: the tar directory should exist (and contain the downloaded .tar files)"
		)
		return

	if not args.fix and out_dir.exists():
		print(f"Wrong usage: the output directory should not exist ({args.out_dir})")
		return



	full_res_dir = out_dir / "full_res"
	low_res_dir = out_dir / "low_res"

	if not args.fix:
		tar_files = [
			x for x in tar_dir.iterdir() if str(x).endswith(".tar.gz")
		]
		n = len(tar_files)


		str_files = "\n".join(map(str, tar_files))
		print(f"Ready to untar {n} tar files:\n{str_files}")

		remove_tar = (
			input("Remove .tar files once they are processed? [y|N] ").lower() == "y"
		)

		# Untar CSGO files
		f = partial(process_tar, out_dir=full_res_dir, remove_tar=remove_tar)
		with Pool(n) as p:
			p.map(f, tar_files)

		print(f"{n} .tar files unpacked in {full_res_dir}")

		move_and_rename_files(full_res_dir)

	with Path("test_split.txt").open("w") as f:
		for datasetfile in os.listdir(full_res_dir):
			is_test = random.randint(1, 12) == 1
			if is_test:
				print(f"selected {datasetfile} as a test split.")
				f.write(datasetfile + "\n")

	#
	# Create low-res data
	#

	csgo_dataset = CSGOHdf5Dataset(full_res_dir)

	train_dataset = Dataset(low_res_dir / "train", None)
	test_dataset = Dataset(low_res_dir / "test", None)


	with Path("test_split.txt").open("r") as f:
		test_files = f.read().split("\n")

	for i in tqdm(csgo_dataset._filenames, desc="Creating low_res"):
		episode = Episode(
			**{
				k: v
				for k, v in csgo_dataset[SegmentId(i, 0, 1000)].__dict__.items()
				if k not in ("mask_padding", "id")
			}
		)
		episode.obs = T.resize(
			episode.obs, (low_res_h, low_res_w), interpolation=T.InterpolationMode.BICUBIC
		)
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


if __name__ == "__main__":
=======
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
import random

low_res_w = 64
low_res_h = 48


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"tar_dir",
		type=Path,
		help="folder containing the .tar files.",
	)
	parser.add_argument(
		"out_dir",
		type=Path,
		help="a new directory (should not exist already), the script will untar and process data there",
	)

	parser.add_argument("--fix", action="store_true", help="Use this to edit an existing dataset.")
	return parser.parse_args()


def process_tar(path_tar: Path, out_dir: Path, remove_tar: bool) -> None:
	d = out_dir
	d.mkdir(exist_ok=True, parents=True)
	shutil.move(path_tar, d)
	subprocess.run(f"cd {d} && tar -xzvf {path_tar.name}", shell=True)
	new_path_tar = d / path_tar.name
	if remove_tar:
		new_path_tar.unlink()
	else:
		shutil.move(new_path_tar, path_tar.parent)

import os
import shutil

def move_and_rename_files(parent_folder):
    # Get a list of all subdirectories in the parent folder
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            # Iterate over all files in the subfolder
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                
                # Check if it's a file
                if os.path.isfile(file_path):
                    # Create the new file name
                    new_file_name = f"{subfolder}_{file_name}"
                    new_file_path = os.path.join(parent_folder, new_file_name)
                    
                    # Move and rename the file
                    shutil.move(file_path, new_file_path)
                    print(f"Moved: {file_path} -> {new_file_path}")
            
            # Optionally, remove the now-empty subfolder
            os.rmdir(subfolder_path)
            print(f"Removed empty folder: {subfolder_path}")



def main():
	args = parse_args()

	tar_dir = args.tar_dir.absolute()
	out_dir = args.out_dir.absolute()

	if not tar_dir.exists():
		print(
			"Wrong usage: the tar directory should exist (and contain the downloaded .tar files)"
		)
		return

	if not args.fix and out_dir.exists():
		print(f"Wrong usage: the output directory should not exist ({args.out_dir})")
		return



	full_res_dir = out_dir / "full_res"
	low_res_dir = out_dir / "low_res"

	if not args.fix:
		tar_files = [
			x for x in tar_dir.iterdir() if str(x).endswith(".tar.gz")
		]
		n = len(tar_files)


		str_files = "\n".join(map(str, tar_files))
		print(f"Ready to untar {n} tar files:\n{str_files}")

		remove_tar = (
			input("Remove .tar files once they are processed? [y|N] ").lower() == "y"
		)

		# Untar CSGO files
		f = partial(process_tar, out_dir=full_res_dir, remove_tar=remove_tar)
		with Pool(n) as p:
			p.map(f, tar_files)

		print(f"{n} .tar files unpacked in {full_res_dir}")

		move_and_rename_files(full_res_dir)

	with Path("test_split.txt").open("w") as f:
		for datasetfile in os.listdir(full_res_dir):
			is_test = random.randint(1, 12) == 1
			if is_test:
				print(f"selected {datasetfile} as a test split.")
				f.write(datasetfile + "\n")

	#
	# Create low-res data
	#

	csgo_dataset = CSGOHdf5Dataset(full_res_dir)

	train_dataset = Dataset(low_res_dir / "train", None)
	test_dataset = Dataset(low_res_dir / "test", None)


	with Path("test_split.txt").open("r") as f:
		test_files = f.read().split("\n")

	for i in tqdm(csgo_dataset._filenames, desc="Creating low_res"):
		episode = Episode(
			**{
				k: v
				for k, v in csgo_dataset[SegmentId(i, 0, 1000)].__dict__.items()
				if k not in ("mask_padding", "id")
			}
		)
		episode.obs = T.resize(
			episode.obs, (low_res_h, low_res_w), interpolation=T.InterpolationMode.BICUBIC
		)
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


if __name__ == "__main__":
>>>>>>> origin/master
	main()