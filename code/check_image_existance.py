import sys
import os
from os.path import isfile, join, splitext
import csv


debugfile = open('/home/joris/RESULTS/check_image_existance_pascal_2012t.csv', 'w')
writer = csv.writer(debugfile)

DATA_DIR = '/home/joris/RESULTS/' 
IDS_FILE = DATA_DIR + 'NetworkOutput/val.txt'
IMAGE_DIR = DATA_DIR + 'NetworkOutput/ccnn/voc_2012_val_results/Images/'


ids_file = open(IDS_FILE)
ids = [line[:-1] for line in ids_file]


for in_idx, in_ in enumerate(ids):
	if os.path.exists(IMAGE_DIR + in_ + '.jpg'):
		continue
	else:
		debugline = 'Image does not exist: DIR = %s%s.jpg \n' %(IMAGE_DIR, in_)
		debugfile.write(debugline)

