import sys
import os

import csv
from shutil import copyfile




 

IDS_FILE = '/home/joris/tf_projects/segmentation/code/image-fingerprinting-master/name_pairs_png.csv'

IMAGE_DIR = '/home/joris/RESULTS/tensorflow/fully_supervised/'





for filename in os.listdir(IMAGE_DIR):
	with open(IDS_FILE, 'rt') as f:
	    reader = csv.reader(f, delimiter=',')
	    for row in reader:
	        if filename == row[1]:

	        	print row
	        	new_name = str(row[0])
	        	print row[0]
	        	os.rename((IMAGE_DIR + filename), (IMAGE_DIR + new_name))





'''
ids_file = open(IDS_FILE)
ids = [line[:-1] for line in ids_file]

for in_idx, in_ in enumerate(ids):
	image_name = '_' + str(in_idx) + extention
	new_name = in_[0] + extention


	for filename in os.listdir(IMAGE_DIR):
		print filename
		print image_name
		if filename == image_name:
			os.rename(filename, new_name)


	
	

	

for in_idx, in_ in enumerate(ids):
	image = IMAGE_DIR + in_ + extention
	image_copy = DESTINATION + in_ + extention
	
	if os.path.exists(image):
		copyfile(image, image_copy)
	else:
		print in_idx, in_, 'image not in folder'

'''