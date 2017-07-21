import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from PIL import Image
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

save_path = '/home/joris/RESULTS/tensorflow/fully_supervised/' 

sys.path.append("/home/joris/workspace/models/slim")
sys.path.append("/home/joris/tf_projects/segmentation")

load_path = '/home/joris/tf_projects/segmentation/checkpoints/fcn32s_full_mode1/fcn32s_pascalVOC_105000.ckpt'

tfrecord_filename = '/home/joris/tf_projects/segmentation/tfrecords/mode1/pascal_augmented_val.tfrecords'
#tfrecord_filename = '/home/joris/tf_projects/segmentation/tfrecords/pascal_voc/pascal_augmented_val.tfrecords'

PASCAL_CLASSES =  [ 'background', 
                    'aeroplane',   'bicycle', 'bird',  'boat',      'bottle',
                    'bus',         'car',     'cat',   'chair',     'cow',
                    'diningtable', 'dog',     'horse', 'motorbike', 'person', 
                    'pottedplant', 'sheep',   'sofa',  'train',     'tvmonitor']


from tf_image_segmentation.models.fcn_32s import FCN_32s
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.visualization import visualize_segmentation_adaptive
from tf_image_segmentation.utils.export import get_export_image

slim = tf.contrib.slim
pascal_voc_lut = pascal_segmentation_lut()
number_of_classes = 21

filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=1)

image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image, axis=0)
annotation_batch_tensor = tf.expand_dims(annotation, axis=0)

# Be careful: after adaptation, network returns final labels and not logits
FCN_32s = adapt_network_for_any_size_input(FCN_32s, 32)


pred, fcn_32s_variables_mapping, logits = FCN_32s(image_batch_tensor=image_batch_tensor,
                                          number_of_classes=number_of_classes,
                                          is_training=False)

# Take away the masked out values from evaluation
weights = tf.to_float( tf.not_equal(annotation_batch_tensor, 255))

# Define the accuracy metric: Mean Intersection Over Union
miou, update_op = slim.metrics.streaming_mean_iou(predictions=pred,
                                                   labels=annotation_batch_tensor,
                                                   num_classes=number_of_classes,
                                                   weights=weights)



# The op for initializing the variables.
initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(initializer)

    saver.restore(sess, load_path)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    with open('/home/joris/RESULTS/maxscoringpixel.csv', 'wr') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
                        'id',
                        'class',
                        'max_pixel_hight',
                        'max_pixel_width',
                        ])
    # There are 904 images in restricted validation dataset
    # There are 1449 images in pascalVOC validation dataset (also used for ccnn)
    for i in range(1449):
        
        count = 0 
        image_np, annotation_np, pred_np, tmp = sess.run([image, annotation, pred, update_op])

        hight = pred_np.shape[1]
        width = pred_np.shape[2]

        print 'the size of this image is:', hight, width
        logits_resized = tf.image.resize_bilinear(
                                                images=logits,
                                                size=[hight,width],
                                                align_corners=None,
                                                name=None
                                                )

        new_logits = sess.run(logits_resized)

        #print new_logits.size
        logits_reshaped = tf.squeeze(new_logits)

        class_stack = tf.unstack(logits_reshaped,
                        num=None,
                        axis=2,
                        name='unstack'
                        )  
        
        for c in class_stack:
            max_scoring_hight_list = tf.argmax(
                                c,
                                axis=0,
                                name=None,
                                dimension=None
                                )
            max_scoring_hight_list = sess.run(max_scoring_hight_list)

            max_scoring_value_hight_list = tf.reduce_max(
                                c,
                                axis=0,
                                keep_dims=False,
                                name=None,
                                reduction_indices=None
                                )  

            max_scoring_value_hight_list = sess.run(max_scoring_value_hight_list)
            max_scoring_width = sess.run(tf.argmax(max_scoring_value_hight_list)) 
            max_scoring_hight = max_scoring_hight_list[max_scoring_width]


            object_class = PASCAL_CLASSES[count]
            count += 1

            print object_class
            print 'hight =', max_scoring_hight
            print 'width =', max_scoring_width  

            with open('/home/joris/RESULTS/maxscoringpixel.csv', 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                        i,
                        object_class,
                        max_scoring_hight,
                        max_scoring_width
                        ])
            if max_scoring_hight > hight:
                print 'ERRORRRR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            if max_scoring_width > width:
                print 'ERRORRRR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

        # Display the segmentation result
        if i >= 0 and i < 1:   
            upsampled_predictions = pred_np.squeeze()
            visualize_segmentation_adaptive(upsampled_predictions, pascal_voc_lut)

        # Export resulting images
        if i >= 0 and i < 1449:
 
            upsampled_predictions = pred_np.squeeze()
            resulting_segmentation_image = get_export_image(upsampled_predictions)

            # # Save the annotated image      
            # name = ''.join([save_path, '_', str(i),'.png'])
            # imsave(name, resulting_segmentation_image)
            # print 'image', i, 'saved'

            # Save the original image      
            #name = ''.join([save_path, '_', str(i),'.jpg'])
            #imsave(name, image_np)
            #print 'image', i, 'saved'

            # Display save images
            #plt.imshow(resulting_segmentation_image)
            #plt.show()

            #plt.imshow(image_np)
            #plt.show()

        
    coord.request_stop()
    coord.join(threads)
    
    res = sess.run(miou)
    
    print("Pascal VOC 2012 Restricted (RV-VOC12) Mean IU: " + str(res))

