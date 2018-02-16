# usage: python extract_sound.py /home/nguyenmd/workspace/data/LIRIS/data_img /home/nguyenmd/workspace/data/LIRIS/data_sound 
import os
import sys
from os import listdir
from os.path import isfile, join, isdir, splitext
from subprocess import call

def readDir(path):
	return sorted([join(path, f) for f in listdir(path) if isdir(join(path, f))])

def readVideoFiles(path):
	return sorted([join(path, f) for f in listdir(path) if isfile(join(path, f)) and splitext(f)[1] in (".mp4",)])

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def extract_sound(folder_path, output_path):
	dirs = readDir(folder_path)
	for subdir in dirs:
		folder_name = subdir.split("/")[-1]
		video_files = readVideoFiles(subdir)
		mkdir(os.path.join(output_path, folder_name))
		for video_file in video_files:
			outfile = os.path.join(output_path, folder_name, video_file.split("/")[-1][:-4] + ".mp3")
			# print(["/home/nguyenmd/workspace/libs/FFmpeg/ffmpeg", "-i", video_file, "-f", "mp3", "-ab", "192000", "-vn", outfile])
			call(["/home/nguyenmd/workspace/libs/FFmpeg/ffmpeg", "-i", video_file, "-f", "mp3", "-ab", "441000", "-vn", outfile])



if __name__=="__main__":
	extract_sound(sys.argv[1], sys.argv[2])