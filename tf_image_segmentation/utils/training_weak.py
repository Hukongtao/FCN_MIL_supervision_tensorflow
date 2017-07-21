import tensorflow as tf


def get_labels_from_annotation(annotation_tensor, class_labels):
 
    valid_entries_class_labels = class_labels[:-1]
    
    # Stack the binary masks for each class
    labels_2d = map(lambda x: tf.equal(annotation_tensor, x),
                    valid_entries_class_labels)

    # Perform the merging of all of the binary masks into one matrix
    labels_2d_stacked = tf.stack(labels_2d, axis=2)
    
    # Convert tf.bool to tf.float
    # Later on in the labels and logits will be used
    # in tf.softmax_cross_entropy_with_logits() function
    # where they have to be of the float type.
    labels_2d_stacked_float = tf.to_float(labels_2d_stacked)

    return labels_2d_stacked_float



def get_number_of_classes_in_annotation(labels_2d_stacked_float, class_labels):

    labels_2d_flat = tf.reshape(labels_2d_stacked_float, [-1])

    classes_in_annotaion, id_classes_in_annotaion = tf.unique(labels_2d_flat)

    number_of_classes_in_annotation = tf.size(classes_in_annotaion)

    return number_of_classes_in_annotation



def get_labels_from_annotation_batch(annotation_batch_tensor, class_labels):

    batch_labels = tf.map_fn(fn=lambda x: get_labels_from_annotation(annotation_tensor=x, class_labels=class_labels),
                             elems=annotation_batch_tensor,
                             dtype=tf.float32)
    
    return batch_labels


def get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor, class_labels):

    mask_out_class_label = class_labels[-1]
 
    valid_labels_mask = tf.not_equal(annotation_batch_tensor,
                                        mask_out_class_label)
    
    valid_labels_indices = tf.where(valid_labels_mask)
    
    return tf.to_int32(valid_labels_indices)



def create_derived_annotation_batch_tensor(logits_batch_tensor):

    image_train_size = [12,12]
 

    softmax_scores = tf.nn.softmax(logits_batch_tensor)

    softmax_scores_reshaped = tf.squeeze(softmax_scores)

    class_stack = tf.unstack(softmax_scores_reshaped,
                        num=None,
                        axis=2,
                        name='unstack'
                        )

    for c in class_stack:
        
        max_scoring_x_list = []
        max_scoring_value_x_list = []
        
        # get the max scoring pixel location out of the hight dimension (per label class)
        # max_scoring_class =  Tensor("ArgMax:0", shape=(384,), dtype=int64)
        max_scoring_x = tf.argmax(
                            c,
                            axis=0,
                            name=None,
                            dimension=None
                            )

        max_scoring_value_x = tf.reduce_max(
                            c,
                            axis=0,
                            keep_dims=False,
                            name=None,
                            reduction_indices=None
                            )  

        max_scoring_x_list.append(max_scoring_x)
        max_scoring_value_x_list.append(max_scoring_value_x)

    count = 0 
    derived_annotation_batch_tensor = tf.zeros([1, image_train_size[0], image_train_size[1]], dtype=tf.int32)


    for i in max_scoring_value_x_list:

        count += 1
        max_scoring_y = tf.argmax(
                                i,
                                axis=0,
                                name=None,
                                dimension=None
                                )

        max_scoring_x = tf.slice(max_scoring_x, [max_scoring_y], [1])

        sess = tf.Session()

        x = max_scoring_x[0]
        y = max_scoring_y   

        gt_tensor_layer = tf.SparseTensor(indices=[[x, y]], values=[count], dense_shape=[image_train_size[0], image_train_size[1]])

        derived_annotation_batch_tensor = derived_annotation_batch_tensor + tf.sparse_tensor_to_dense(gt_tensor_layer)

    return derived_annotation_batch_tensor


def get_valid_logits_and_labels(annotation_batch_tensor,
                                logits_batch_tensor,
                                class_labels):
 
    labels_batch_tensor = get_labels_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor,
                                                           class_labels=class_labels)
 
    number_of_classes = get_number_of_classes_in_annotation(labels_2d_stacked_float=labels_batch_tensor, 
                                                            class_labels=class_labels)   
    #valid_batch_indices = get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor,
                                                                          #class_labels=class_labels)

    
    #valid_labels_batch_tensor = tf.gather_nd(params=labels_batch_tensor, indices=valid_batch_indices)
    
    #valid_logits_batch_tensor = tf.gather_nd(params=logits_batch_tensor, indices=valid_batch_indices)
    

    derived_annotation_batch_tensor = create_derived_annotation_batch_tensor(logits_batch_tensor=logits_batch_tensor)




    labels_batch_tensor = get_labels_from_annotation_batch(annotation_batch_tensor=derived_annotation_batch_tensor,
                                                           class_labels=class_labels)     

    valid_batch_indices = get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor=derived_annotation_batch_tensor,
                                                                          class_labels=class_labels)



    valid_labels_batch_tensor_gt = tf.gather_nd(params=labels_batch_tensor, indices=valid_batch_indices)

    valid_logits_batch_tensor_gt = tf.gather_nd(params=logits_batch_tensor, indices=valid_batch_indices)









    return valid_labels_batch_tensor_gt, valid_logits_batch_tensor_gt, number_of_classes




