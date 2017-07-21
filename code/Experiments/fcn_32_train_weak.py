import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from PIL import Image
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


sys.path.append("/home/joris/workspace/models/slim")
sys.path.append("/home/joris/tf_projects/segmentation")

DIR = '/home/joris/tf_projects/segmentation/'

#Input Paths
vgg_checkpoint_path = DIR + 'checkpoints/vgg_16.ckpt'
tfrecord_file = DIR + 'tfrecords/mode1/pascal_augmented_train.tfrecords'

#Output Paths
#save_path = DIR + 'checkpoints/model_fcn32s_weak_final/'
#save_name = 'model_fcn32s_weak_final'

save_path = DIR + 'checkpoints/fcn32s_weak_mode1/'
save_name = 'fcn32s_weak_mode1'

log_path = DIR + 'logs/fcn32s_weak_mode1'

# import 
from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.models.fcn_32s import FCN_32s, extract_vgg_16_mapping_without_fc8
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from tf_image_segmentation.utils.training_weak import get_valid_logits_and_labels
from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color, flip_randomly_left_right_image_with_annotation, scale_randomly_image_with_annotation_with_fixed_size_output)


slim = tf.contrib.slim
image_train_size = [384, 384]
number_of_classes = 21
pascal_voc_lut = pascal_segmentation_lut()
class_labels = pascal_voc_lut.keys()
learning_rate = 0.00000001
num_epochs = 10

# Read the tfrecord_file containing the filename queue, and define number of epochs
filename_queue = tf.train.string_input_producer([tfrecord_file], num_epochs=num_epochs)

# Walk trough the filenames queue and create image/annotation tensor from raw binary representations. 
# image =  Tensor("Reshape:0", shape=(?, ?, 3), dtype=uint8) 
# annotation =  Tensor("Reshape_1:0", shape=(?, ?, 1), dtype=uint8) 
image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Flip image/annotation tuple with .5 probability 
image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)

# Randomly adjust brightness, saturation, hue and contrast. Used for inception model training in TF-Slim
'''image = distort_randomly_image_color(image)'''

# Randomly scale the image/annotation tensors
# resized_image =  Tensor("Squeeze_2:0", shape=(384, 384, 3), dtype=uint8) 
# resized_annotation =  Tensor("sub_12:0", shape=(384, 384, 1), dtype=int32) 
resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation, image_train_size)

# remove all dimensions of size 1 from annotation tensors 
# resized_annotation =  Tensor("Squeeze_4:0", shape=(384, 384), dtype=int32)
resized_annotation = tf.squeeze(resized_annotation)

# Create queue (of 1) to train on 
# image batch =  Tensor("shuffle_batch:0", shape=(1, 384, 384, 3), dtype=uint8)
# annotation_batch =  Tensor("shuffle_batch:1", shape=(1, 384, 384), dtype=int32) 
image_batch, annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                        batch_size=1,
                                                        capacity=3000,
                                                        num_threads=2,
                                                        min_after_dequeue=1000)

# FCN-32 Model defenition ()
# upsampled_logits_batch = Tensor("fcn_32s/conv2d_transpose:0", shape=(?, ?, ?, ?), dtype=float32) 
# logits = shape=(1, 12, 12, 21)
upsampled_logits_batch, vgg_16_variables_mapping, logits_batch = FCN_32s(image_batch_tensor=image_batch,
                                                                                number_of_classes=number_of_classes,
                                                                                is_training=True)

# set the shape of the logits so unstacking works
#upsampled_logits_batch.set_shape([1, image_train_size[0], image_train_size[1], number_of_classes])

#print 'upsampled_logits_batch = ', upsampled_logits_batch, '\n'


# Get all classes and probabilities (only for known pixels, undefined pixels are neglected )
# valid_labels_batch_tensor = Tensor("GatherNd:0", shape=(?, 21), dtype=float32) 
# valid_logits_batch_tensor = Tensor("GatherNd_1:0", shape=(?, ?), dtype=float32) 
valid_labels_batch_tensor, valid_logits_batch_tensor, number_of_classes_in_annotation = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                                                    logits_batch_tensor=logits_batch,
                                                                                                                    class_labels=class_labels)

# Softmax cross entropy between logits and labels
# RETURNS: 1-D Tensor of length batch_size with the softmax cross entropy loss (of the same type as logits) .
# The cross_entropies = Tensor("Reshape_6:0", shape=(?,), dtype=float32) 
cross_entropies = tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels_batch_tensor, 
                                                          logits=valid_logits_batch_tensor)

# Normalize the cross entropy - the number of elements is different during each step due to mask out regions
# The cross_entropy_sum = Tensor("Mean:0", shape=(), dtype=float32) 
cross_entropy_sum = tf.reduce_mean(cross_entropies)


# 1/number of classes in anntotation = factor

factor = tf.cast(tf.divide(1, number_of_classes_in_annotation), tf.float32)

# 1/|L| * softmax cross entropy 
final_loss = tf.multiply(factor, cross_entropy_sum)

# size_of(pred): (batch_size=1) * image_width*image_height  (each pixel has one predicted label)
# pred = Tensor("ArgMax_1:0", shape=(?, ?, ?), dtype=int64)
pred = tf.argmax(upsampled_logits_batch, axis=3)

# size_of(probabilities): (batch_size=1) * image_width*image_height*num_classes  (each pixel has num_classes probabilities)
# probabilities = Tensor("Reshape_8:0", shape=(?, ?, ?, ?), dtype=float32)
probabilities = tf.nn.softmax(upsampled_logits_batch)


with tf.variable_scope("adam_vars"):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(final_loss)


# Variable's initialization functions
vgg_16_without_fc8_variables_mapping = extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping)


init_fn = slim.assign_from_checkpoint_fn(model_path=vgg_checkpoint_path,
                                         var_list=vgg_16_without_fc8_variables_mapping)

global_vars_init_op = tf.global_variables_initializer()


annotation_batch = tf.cast(annotation_batch, tf.float32)
annotation_softmax = tf.nn.softmax(annotation_batch)
output_softmax = tf.nn.softmax(upsampled_logits_batch)


tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)
tf.summary.histogram('Probability expected', annotation_softmax)
tf.summary.histogram('Probability inferred', output_softmax)

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
saver = tf.train.Saver(model_variables, max_to_keep=100)


with tf.Session()  as sess:
    
    sess.run(combined_op)
    init_fn(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    

    # 10 epochs
    # number of items in training set
    number_of_items = 11127
    save_interval = 2500

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

