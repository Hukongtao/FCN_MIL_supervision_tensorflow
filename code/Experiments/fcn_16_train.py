import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from PIL import Image
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


sys.path.append("/home/joris/workspace/models-fully_conv_vgg/slim/")
sys.path.append("/home/joris/tf_projects/tf_whats_the_point/")

log_folder = 'logs/fcn16s'

fcn_32s_checkpoint_path = 'checkpoints/model_fcn32s_final/model_fcn32s_final.ckpt'

save_path = 'checkpoints/model_fcn16s_final/model_fcn16s_final.ckpt'

tfrecord_filename = 'tf_image_segmentation/pascal_augmented_val.tfrecords'

from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.models.fcn_16s import FCN_16s
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from tf_image_segmentation.utils.training import get_valid_logits_and_labels
from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)


slim = tf.contrib.slim
image_train_size = [384, 384]
number_of_classes = 21
pascal_voc_lut = pascal_segmentation_lut()
class_labels = pascal_voc_lut.keys()

filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=10)

image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Various data augmentation stages
image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)

# image = distort_randomly_image_color(image)

resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation, image_train_size)


resized_annotation = tf.squeeze(resized_annotation)

image_batch, annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                             batch_size=1,
                                             capacity=3000,
                                             num_threads=2,
                                             min_after_dequeue=1000)

upsampled_logits_batch, fcn_32s_variables_mapping = FCN_16s(image_batch_tensor=image_batch,
                                                           number_of_classes=number_of_classes,
                                                           is_training=True)


valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                     logits_batch_tensor=upsampled_logits_batch,
                                                                                    class_labels=class_labels)



cross_entropies = tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels_batch_tensor,
                                                          logits=valid_logits_batch_tensor)

#cross_entropy_sum = tf.reduce_sum(cross_entropies)

cross_entropy_sum = tf.reduce_mean(cross_entropies)

pred = tf.argmax(upsampled_logits_batch, axis=3)

probabilities = tf.nn.softmax(upsampled_logits_batch)


with tf.variable_scope("adam_vars"):
    train_step = tf.train.AdamOptimizer(learning_rate=0.00000001).minimize(cross_entropy_sum)


#adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])

# Variable's initialization functions
init_fn = slim.assign_from_checkpoint_fn(model_path=fcn_32s_checkpoint_path,
                                         var_list=fcn_32s_variables_mapping)

global_vars_init_op = tf.global_variables_initializer()

tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

merged_summary_op = tf.summary.merge_all()

summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
     os.makedirs(log_folder)

#optimization_variables_initializer = tf.GLOBAL_VARIABLES_initializer(adam_optimizer_variables)
    
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
    

    # Let's read off 3 batches just for example
    for i in xrange(11127 * 10):
    
        cross_entropy, summary_string, _ = sess.run([ cross_entropy_sum,
                                                      merged_summary_op,
                                                      train_step ])

        summary_string_writer.add_summary(summary_string, 11127 * 10 + i)
        
        print("step :" + str(i) + " Loss: " + str(cross_entropy))
        
        if i % 11127 == 0:
            save_state = saver.save(sess, save_path)
            print("Model saved in file: %s" % save_state)
        
    coord.request_stop()
    coord.join(threads)
    
    save_state = saver.save(sess, save_path)
    print("Model saved in file: %s" % save_state)
    
summary_string_writer.close()
