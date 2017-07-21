import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from PIL import Image
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

DIR = '/home/joris/tf_projects/segmentation/'

sys.path.append("/home/joris/workspace/models/slim")
sys.path.append("/home/joris/tf_projects/segmentation")

#Input Paths
vgg_checkpoint_path = DIR + 'checkpoints/vgg_16.ckpt'
tfrecord_file = DIR + 'tfrecords/mode1/pascal_augmented_train.tfrecords'

#Output Paths
#save_path = DIR + 'checkpoints/model_fcn32s_weak_final/'
#save_name = 'model_fcn32s_weak_final'

save_path = DIR + 'checkpoints/fcn32s_pascalVOC/'
save_name = 'fcn32s_pascalVOC'

log_path = DIR + 'logs/fcn32s_pascalVOC'





# import 
from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.models.fcn_32s import FCN_32s, extract_vgg_16_mapping_without_fc8
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from tf_image_segmentation.utils.training import get_valid_logits_and_labels
from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color, flip_randomly_left_right_image_with_annotation, scale_randomly_image_with_annotation_with_fixed_size_output)


slim = tf.contrib.slim
image_train_size = [384, 384]
number_of_classes = 21
pascal_voc_lut = pascal_segmentation_lut()
class_labels = pascal_voc_lut.keys()
learning_rate = 0.000001
num_epochs = 10


# Read the tfrecord_file containing the filename queue, and define number of epochs
filename_queue = tf.train.string_input_producer([tfrecord_file], num_epochs=num_epochs)

# Walk trough the filenames queue and create image/annotation tensor from raw binary representations. 
image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Flip image/annotation tuple with .5 probability (WHY?)
image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)

# Randomly adjust brightness, saturation, hue and contrast. Used for inception model training in TF-Slim
'''image = distort_randomly_image_color(image)'''

# Randomly scale the image/annotation tensors
resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation, image_train_size)

# remove all dimensions of size 1 from annotation tensors 
resized_annotation = tf.squeeze(resized_annotation)

# Create queue (of 1) to train on 
image_batch, annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                             batch_size=1,
                                             capacity=3000,
                                             num_threads=2,
                                             min_after_dequeue=1000)

# FCN-32 Model defenition ()
upsampled_logits_batch, vgg_16_variables_mapping = FCN_32s(image_batch_tensor=image_batch,
                                                           number_of_classes=number_of_classes,
                                                           is_training=True)


valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                     logits_batch_tensor=upsampled_logits_batch,
                                                                                    class_labels=class_labels)




# Full supervison:

# Softmax cross entropy between logits and lables
cross_entropies = tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels_batch_tensor, logits=valid_logits_batch_tensor)


# Normalize the cross entropy -- the number of elements is different during each step due to mask out regions
cross_entropy_sum = tf.reduce_mean(cross_entropies)

pred = tf.argmax(upsampled_logits_batch, axis=3)

probabilities = tf.nn.softmax(upsampled_logits_batch)


# Loss for point-level supervision

'''cross_entropy_sum_points = cross_entropy_sum + cross_entropies'''


# loss for Point + objectness




with tf.variable_scope("adam_vars"):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_sum)


# Variable's initialization functions
vgg_16_without_fc8_variables_mapping = extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping)


init_fn = slim.assign_from_checkpoint_fn(model_path=vgg_checkpoint_path,
                                         var_list=vgg_16_without_fc8_variables_mapping)

global_vars_init_op = tf.global_variables_initializer()


tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

merged_summary_op = tf.summary.merge_all()

summary_string_writer = tf.summary.FileWriter(log_path)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_path):
     os.makedirs(log_path)
    
#The op for initializing the variables.
local_vars_init_op = tf.local_variables_initializer()

combined_op = tf.group(local_vars_init_op, global_vars_init_op)

# We need this to save only model variables and omit
# optimization-related and other variables.
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)


with tf.Session()  as sess:
    
    sess.run(combined_op)
    init_fn(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    

    # 10 epochs
    # number of items in training set
    number_of_items = 11127
    save_interval = 5000

    for i in xrange(number_of_items * num_epochs):
    
        cross_entropy, summary_string, _ = sess.run([ cross_entropy_sum,
                                                      merged_summary_op,
                                                      train_step ])
  
        summary_string_writer.add_summary(summary_string, number_of_items * num_epochs + i)

        print("step :" + str(i) + " Loss: " + str(cross_entropy))

        if i % save_interval == 0: #i % 11127 == 0 
            save_state = saver.save(sess, save_path + save_name + '_' + str(i)  + '.ckpt')
            print("Model saved in file: %s" % save_state)
        
    coord.request_stop()
    coord.join(threads)
    
    save_state = saver.save(sess, save_path)
    print("Model saved in file: %s" % save_state)
    
summary_string_writer.close()



