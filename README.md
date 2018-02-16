# deep-movie-trailer
- Resources: C3D, facenet, soundnet, LIRIS, RNN
- Goal: predict Valance, Arousal from LIRIS.

# Pipeline/Baseline:
	- Feature extraction with C3D, facenet, soundnet


# LIRIS dataset:
	- The LIRIS dataset can be found here: http://liris-accede.ec-lyon.fr/files/database-download/download.php
	- Account: eurecom
	- Password: Giadinh@2


# How to use the code:
	## Audio
	- To get sound out of a video: python extract_sound.py `<video_path> <output_path>`
	- To generate a list of all audio files: python make_audio_list.py `<audio_path> <output_filename>`
	- To extract audio feature: go to either
		+ https://gitlab.eurecom.fr/nguyenmd/vgg-audio
		+ https://gitlab.eurecom.fr/nguyenmd/soundnet-tensorflow
	
	## Video
	- To sample the video and output a set of images: python convert_video_to_image.py `<video_folder>` `<fps>`
	- To extract C3D feature goto:
		+ https://gitlab.eurecom.fr/nguyenmd/c3d-tensorflow

	## Fusion
	- Modify path and parameters in train.py then run python train.py