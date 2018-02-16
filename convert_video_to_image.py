
import sys
import os
import re
from os.path import join, isfile, isdir, abspath
from subprocess import call

def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)


if __name__=="__main__":
	root_folder = sys.argv[1]
	all_folder = sorted([f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))])
	for folder in all_folder:
		all_video_files = [f for f in os.listdir(join(root_folder, folder)) if os.path.isfile(os.path.join(join(root_folder, folder), f))]
		for file in all_video_files:
			file_path = os.path.join(join(root_folder, folder), file)
			print "ffmpeg -i " + file_path + " -vf fps=" + sys.argv[2] + " " + os.path.join(join(root_folder, folder) + "/%05d.jpg")
			call(["ffmpeg", "-i", file_path, "-vf", "fps=" + sys.argv[2], os.path.join(join(root_folder, folder) + "/%05d.jpg")])