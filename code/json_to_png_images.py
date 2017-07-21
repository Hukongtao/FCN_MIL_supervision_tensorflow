import json
import csv
from skimage.io import imread, imsave
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET



DOCKER_DIR = '/home/joris/dockers'

pascal_images = DOCKER_DIR + '/WTP-docker/data/VOC2012/JPEGImages' 

json_infile_name = '/home/joris/dockers/WTP-docker/data/WTP/pascal2012_trainval_main.json'
json_data = json.loads(open(json_infile_name, 'r').read())


debugfile = open('/home/joris/dockers/WTP-docker/data/WTP/points_to_masks_debug.csv', 'w')
writer = csv.writer(debugfile)

save_path = '/home/joris/dockers/WTP-docker/data/TEST/' 


c = 0 

# walk over the json data 
for k in json_data:
	c+=1
	# print(c)
	if c == 10:
		continue
	v = json_data[k]
	if not len(v):
		del v
	else:

		if os.path.exists(os.path.join(pascal_images, k + '.jpg')):

			# open image with name of k	
			image = imread(os.path.join(pascal_images, k + '.jpg'))

			# Get dimensions of the image
			image_hight = image.shape[0]
			image_width = image.shape[1]

			if image_hight > 10 and image_width > 10:
				# make nparray in image size with only 0
				size = (image_hight, image_width)
				annotated_image = np.zeros(size, dtype=np.uint8)

				#itterate over json dictionairy
				for i in range (0, len(v)):

					line = [int(v[i]["x"]), int(v[i]["y"]), int(v[i]["rank"]), int(v[i]["cls"])]

					point_hight = line[1] 
					point_width = line[0] 
						
					if point_hight >= image_hight or point_width >= image_width:
						debugline = 'OUT OF BOUNDS: name = %s, line = %s, hight = %s width = %s \n' %(k, line, image_hight, image_width)
						debugfile.write(debugline)
					else:
						# add points
						annotated_image[point_hight, point_width] = line[3]
			
					
				# mask black for visualisation
				'''mask = annotated_image < 1
										annotated_image[mask] = 255'''

				# Save the annotated image			
				name = ''.join([save_path, k, '.png'])
				imsave(name, annotated_image)
				print 'image saved'
			else:
				debugline = 'IMAGE TO SMALL: name = %s, line = %s, hight = %s width = %s \n' %(k, line, image_hight, image_width)
				debugfile.write(debugline)
		else: 
			debugline = 'NO JPEG IMAGE: name = %s, line = %s, hight = %s width = %s \n' %(k, line, image_hight, image_width)
			debugfile.write(debugline)