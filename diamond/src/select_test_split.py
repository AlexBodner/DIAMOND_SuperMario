<<<<<<< HEAD
import os
import random
import argparse

def select_random_files(folder_path, output_file, percentage=10):
    """
    Selects a percentage of files randomly from the given folder and writes their names to a file.

    Args:
        folder_path (str): Path to the folder containing the files.
        output_file (str): Path to the output file where selected file names will be written.
        percentage (int): Percentage of files to select.
    """
    # Get the list of files in the folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Calculate the number of files to select
    num_files_to_select = max(1, int(len(all_files) * percentage / 100))
    
    # Randomly select files
    selected_files = random.sample(all_files, num_files_to_select)
    
    # Write the selected file names to the output file
    with open(output_file, 'w') as f:
        for file_name in selected_files:
            f.write(file_name + '\n')
    
    print(f"{len(selected_files)} files written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select random files from a folder and write their names to a file.")
    parser.add_argument("folder_path", help="Path to the folder containing files.")
    parser.add_argument("output_file", help="Path to the output file.")
    parser.add_argument("--percentage", type=int, default=10, help="Percentage of files to select (default: 10%).")

    args = parser.parse_args()

    select_random_files(args.folder_path, args.output_file, args.percentage)
=======
import os
import random
import argparse

def select_random_files(folder_path, output_file, percentage=10):
    """
    Selects a percentage of files randomly from the given folder and writes their names to a file.

    Args:
        folder_path (str): Path to the folder containing the files.
        output_file (str): Path to the output file where selected file names will be written.
        percentage (int): Percentage of files to select.
    """
    # Get the list of files in the folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Calculate the number of files to select
    num_files_to_select = max(1, int(len(all_files) * percentage / 100))
    
    # Randomly select files
    selected_files = random.sample(all_files, num_files_to_select)
    
    # Write the selected file names to the output file
    with open(output_file, 'w') as f:
        for file_name in selected_files:
            f.write(file_name + '\n')
    
    print(f"{len(selected_files)} files written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select random files from a folder and write their names to a file.")
    parser.add_argument("folder_path", help="Path to the folder containing files.")
    parser.add_argument("output_file", help="Path to the output file.")
    parser.add_argument("--percentage", type=int, default=10, help="Percentage of files to select (default: 10%).")

    args = parser.parse_args()

    select_random_files(args.folder_path, args.output_file, args.percentage)
>>>>>>> origin/master
