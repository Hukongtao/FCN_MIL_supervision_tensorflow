
import numpy as np
from matplotlib import pyplot as plt


def get_export_image(predictions):

    # unique_classes, relabeled_image = np.unique(predictions,
    #                                             return_inverse=True)

    # relabeled_image = relabeled_image.reshape(predictions.shape)

    # print relabeled_image

    hight = predictions.shape[0]
    width = predictions.shape[1]
    
    rgb_lut = [(0, 0, 0), 
               (128, 0, 0),   (0, 128, 0),     (128, 128, 0), (0, 0, 128),    (128, 0, 128), 
               (0, 128, 128), (128, 128, 128), (64, 0, 0),    (192, 0, 0),    (64, 128, 0), 
               (192, 128, 0), (64, 0, 128),    (192, 0, 128), (64, 128, 128), (192, 128, 128), 
               (0, 64, 0),    (128, 64, 0),    (0, 192, 0),   (128, 192, 0),  (0, 64, 128)]


    result_image = np.zeros((hight,width), dtype=(float,3))

    for w in range(width):
        for h in range(hight):
            # print h, w, relabeled_image[h,w]
            rgb = rgb_lut[predictions[h,w]]
            # print rgb 
            result_image[h,w] = (float(rgb[0]) / 256.0, float(rgb[1]) / 256.0, float(rgb[2]) / 256.0)

    return result_image



    # check if order in .val file is the same 
