import sys
import os
from os.path import isfile, join, splitext
import csv
from shutil import copyfile


DATA_DIR = '/home/joris/tf_projects/Data/VOCdevkit/VOC2012/' 

IDS_FILE_DIR = DATA_DIR + 'ImageSets/Segmentation/'
IDS_FILE = 'val.txt'

IMAGE_DIR = DATA_DIR + 'JPEGImages/'

DESTINATION = '/home/joris/RESULTS/JPEGImages/'
extention = '.jpg'



# 1
'''
IDS_FILE = 'train.txt'
IMAGE_DIR = DATA_DIR + 'JPEGImages/'
DESTINATION = DATA_DIR + 'SelectedImages/Train/Images/'
extention = '.jpg'

# 2
IDS_FILE = 'train.txt'
IMAGE_DIR = DATA_DIR + 'SegmentationClass/'
DESTINATION = DATA_DIR + 'SelectedImages/Train/GroundTruth/'
extention = '.png'

# 3 
IDS_FILE = 'val.txt'
IMAGE_DIR = DATA_DIR + 'JPEGImages/'
DESTINATION = DATA_DIR + 'SelectedImages/Val/Images/'
extention = '.jpg'

# 4 
IDS_FILE = 'val.txt'
IMAGE_DIR = DATA_DIR + 'SegmentationClass/'
DESTINATION = DATA_DIR + 'SelectedImages/Val/GroundTruth/'
extention = '.png'
'''

ids_file = open(IDS_FILE_DIR + IDS_FILE)
ids = [line[:-1] for line in ids_file]


for in_idx, in_ in enumerate(ids):
	image = IMAGE_DIR + in_ + extention
	image_copy = DESTINATION + in_ + extention
	
	if os.path.exists(image):
		copyfile(image, image_copy)
	else:
		print in_idx, in_, 'image not in folder'

