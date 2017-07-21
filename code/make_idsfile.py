import sys
import os
from os.path import isfile, join, splitext

def main(argv):
	
	IMG_DIR = '/home/joris/tf_projects/Data/VOCdevkit/VOC2012/SegmentationClassSelected/'
	OUT_FILE = '/home/joris/tf_projects/Data/VOCdevkit/VOC2012/ImageSets/Segmentation/planes_train.txt'


	with open(OUT_FILE, "a") as txt_file:
		for f in os.listdir(IMG_DIR):
			img_prefix = splitext(f)[0]
			if isfile(join(IMG_DIR, f)): 
				txt_file.write(img_prefix + '\n')	

if __name__ == "__main__":
    main(sys.argv)
