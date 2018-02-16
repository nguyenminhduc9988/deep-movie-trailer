

import sys
import os
import re

def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)


if __name__=="__main__":
	root_folder = sys.argv[1]
	mkdir(sys.argv[2])
	all_files = [f for f in os.listdir(root_folder) if os.path.isfile(os.path.join(root_folder, f))]
	for file in all_files:
		folder_name = file.split(".")[0]
		folder_path = os.path.join(sys.argv[2], folder_name)
		mkdir(folder_path)
		print(os.path.join(root_folder, file))
		os.rename(os.path.join(root_folder, file), os.path.join(folder_path, file))