import os
from os import listdir
from os.path import isfile, join, isdir, splitext
import sys
import tensorflow as tf
import csv
import pandas
import numpy.random as rand
import numpy as np
import models 
from sklearn.utils import shuffle
import datetime
import sklearn.preprocessing as skp


def read_dir(path):
	return sorted([join(path, f) for f in listdir(path) if isdir(join(path, f))])

def get_data_file_name(feature_dirs):
	return [d.strip('\n').split('/')[-1] for d in feature_dirs]

def read_feature_folder(f_path):
	return sorted([join(f_path, f) for f in listdir(f_path) if isfile(join(f_path, f)) and f[-4:] == '.npy'])

def read_csv(csv_path):
	#	dict {filename: valence, arousal, valence var, arousal var}
	#
	#
	result = {}
	with open(csv_path, 'r') as op:
		csv_data = csv.reader(op, delimiter='\t')
		next(csv_data)
		for row in csv_data:
			result[row[1][:-4]] = row[4:]
	return result

def read_npy(file_path):
	return np.load(file_path)

def concat_array(a, b):
	return np.concatenate((a, b))

def get_feature(c3d_path, sound_path):
	c3d_feature = read_npy(c3d_path).reshape(-1)
	# sound_feature = read_npy(sound_path).reshape(-1)
	sound_feature = read_csv_comma(sound_path)
	return concat_array(c3d_feature, sound_feature)

def read_all_features(data_dict):
	keys = data_dict.keys()
	features = {}
	for key in sorted(keys):
		feature_dirs = data_dict[key][4]
		features[key] = np.zeros((8096, len(feature_dirs)))
		cnt = 0
		audio_feature = read_npy(data_dict[key][5]).reshape(-1)
		for f in feature_dirs[0: len(feature_dirs) / 20]:
			features[key][:, cnt] = concat_array(read_npy(f).reshape(-1), audio_feature)
			cnt += 1
		print(key)
	return features


def build_batch(data_dict, data_feature, batch_size=50):
	keys = data_dict.keys()
	chosen_idx = rand.randint(0, high=len(keys), size=batch_size)
	result_feature = np.zeros((batch_size, 8096))
	result_label = np.zeros((batch_size, 2))
	result_augmented_data = np.zeros((batch_size, 2))
	cnt = 0
	for idx_num in chosen_idx:
		idx = keys[idx_num]
		c3d_features_path = data_dict[idx][4]
		chosen_feature = rand.randint(0, len(c3d_features_path) - 1)
		temp = get_feature(c3d_features_path[chosen_feature], data_dict[idx][5])
		result_feature[cnt, 0:temp.shape[0]] = temp
		result_label[cnt, :] = data_dict[idx][0:2]
		result_augmented_data[cnt, :] = data_dict[idx][2:4]
		cnt += 1
	return result_feature, result_label, result_augmented_data

def get_number_training_sample(data_dict):
	return np.sum([len(data_dict[key][4]) for key in data_dict.keys()])

def read_csv_comma(path):
	with open(path, 'r') as op:
		a = csv.reader(op)
		arr = np.array([x for x in a]).astype(np.float32).reshape(-1)
		return arr / np.linalg.norm(arr)


if __name__=='__main__':
# load data
	C3D_FEATURE_DIR = '/home/nguyenmd/workspace/data/LIRIS/feature_c3d'
	SOUND_FEATURE_DIR = '/home/nguyenmd/workspace/data/LIRIS/sound_feature'
	VALENCE_AROUSAL_CSV = '/home/nguyenmd/workspace/data/LIRIS/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt'
	VGG_PATH = '/home/nguyenmd/Downloads/audioset/features'
	# Training params
	BATCH_SIZE = 100
	LEARNING_RATE = 1.0
	TRAINING_EPOCHS = 1000000

	all_feature_dirs = read_dir(C3D_FEATURE_DIR)
	all_data_file_name = get_data_file_name(all_feature_dirs)
	all_feature_arousal_valence = read_csv(VALENCE_AROUSAL_CSV)
	# aggregate all data for faster lookup
	for key in all_feature_arousal_valence.keys():
		c3d_features = read_feature_folder(join(C3D_FEATURE_DIR, key))
		# sound_feature = join(SOUND_FEATURE_DIR, key) + '.npy'
		sound_feature = join(VGG_PATH, key) + '.csv'
		all_feature_arousal_valence[key].append(c3d_features)
		all_feature_arousal_valence[key].append(sound_feature)
	# ###
	data_dict = all_feature_arousal_valence

	n_training_sample = get_number_training_sample(data_dict)
	# print("Reading features")
	# all_features = read_all_features(data_dict)
	# print("Read all features")
	# model definition
	x = tf.placeholder(tf.float32, [None, 8096], name='InputData')
	# 0-9 digits recognition,  10 classes
	y = tf.placeholder(tf.float32, [None, 2], name='LabelData')

	model = models.fusion_model(x)
	cross_entropy = tf.norm(model - y, ord='euclidean')
	loss_operation = tf.reduce_mean(cross_entropy)

	# Optimization 
	optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
	training_operation = optimizer.minimize(loss_operation)

	init = tf.global_variables_initializer()
	# ######
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	config.allow_soft_placement=True
    # sess_config.gpu_options.allow_growth = True
	ITER = 50
	saver = tf.train.Saver()
	
	save_file = './models/model_epoch_171_16_02_18_01_48.ckpt'

	with tf.Session(config=config) as sess:
		sess.run(init)
		# summary_writer = tf.summary.FileWriter('./logdir', graph=tf.get_default_graph())
		print("Training")
		# saver.restore(sess, save_file)
		for i in range(TRAINING_EPOCHS):
			total_loss = 0 
			for j in range(ITER):
				features, labels, _ = build_batch(data_dict, BATCH_SIZE)
				loss, _ = sess.run([loss_operation, training_operation], feed_dict={x: features, y: labels})
				total_loss += loss
				print("Loss in %d epoch, batch number %d: %f" % (i, j, loss))
			saver.save(sess, './models/model_epoch_' + str(i) + '_' + datetime.datetime.now().strftime('%d_%m_%y_%H_%M') + '.ckpt')
			print("Avg loss for epoch %d: %f" % (i, total_loss / ITER))