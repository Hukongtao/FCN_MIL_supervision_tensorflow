# USAGE
# python search.py --dataset images --shelve db.shelve --query images/84eba74d-38ae-4bf6-b8bd-79ffa1dad23a.jpg

# import the necessary packages
from PIL import Image
import imagehash
import argparse
import shelve
import csv


def hamdist(str1, str2):
	"""Count the # of differences between equal length strings str1 and str2"""
	diffs = 0
	for ch1, ch2 in zip(str1, str2):
   		if ch1 != ch2:
   			diffs += 1
   	return diffs

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--shelve", required = True,
	help = "output shelve database")

args = vars(ap.parse_args())

DATASET = '/home/joris/RESULTS/tensorflow/gt_num/'

# open the shelve database
db = shelve.open(args["shelve"])

for i in range(1449):
	name = ''.join(['_', str(i),'.jpg'])


	# load the query image, compute the difference image hash, and
	# and grab the images from the database that have the same hash
	# value
	query = Image.open(DATASET + name)
	h = str(imagehash.dhash(query))

	filenames = db[h]

	# set hamdist == 0 for exact similar images, raise for less similairity (NOTE: might find to much!)
	if len(filenames) == 1:
		for line in db:
			if hamdist(line, h) == 1 or hamdist(line, h) == 2 or hamdist(line, h) == 3 or hamdist(line, h) == 4 or hamdist(line, h) == 5:
				filenames = db[h], db[line]

				#x = db[line]
				#filenames.append(x)
			

	print "Found %d images" % (len(filenames))
	
	if len(filenames) == 2:
		with open('name_pairs.csv', 'a') as csvfile:
			writer = csv.writer(csvfile)
			if len(filenames[0]) > len(filenames[1]):
				writer.writerow([filenames[0], filenames[1]])
			else:
				writer.writerow([filenames[1], filenames[0]])

	if len(filenames) == 1:
		with open('names_not_found.csv', 'a') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow([filenames[0]])



	# loop over the images
	#for filename in filenames:
		#print filename
		#image = Image.open(DATASET + "/" + filename)
		#image.show()

# close the shelve database
db.close()





