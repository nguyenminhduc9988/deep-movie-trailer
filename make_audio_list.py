# usage: python make_audio_list.py /home/nguyenmd/workspace/data/LIRIS/data_sound ./audio.list
import os
import sys
from os import listdir
from os.path import isfile, join, isdir, splitext
from subprocess import call

def readDir(path):
	return sorted([join(path, f) for f in listdir(path) if isdir(join(path, f))])

def readAudioFiles(path):
	return sorted([join(path, f) for f in listdir(path) if isfile(join(path, f)) and splitext(f)[1] in (".mp3",)])

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def make_list(sound_path, output_file):
	dirs = readDir(sound_path)
	with open(output_file, "w") as out:
		for subdir in dirs:
			mfiles = readAudioFiles(subdir) 
			for mfile in mfiles:
				out.write(mfile)
				out.write("\n")



if __name__=="__main__":
	make_list(sys.argv[1], sys.argv[2])