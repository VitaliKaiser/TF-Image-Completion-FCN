import os
import sys
import random
import cv2
import numpy as np
import tensorflow as tf

RESIZE_AXIS_SIZE_MIN = 256.0
RESIZE_AXIS_SIZE_MAX = 384.0
HOLE_SIZE_MIN = 96
HOLE_SIZE_MAX = 128
CROP_SIZE = 256
MEAN_PIXEL_SAMPLES = 1000

class MaskedImageDataset:
    """ Class for converting a bunch of images to a image completion dataset
    Args:
        path (str): Path to the images.

    Attributes:
        path (str): Path to the images.
        files (list of str): a list of files in 'path'
    """

    def __init__(self, path, random=False):

        self.path = path
        ### list all files only ones - listdir is pretty slow for a lot of files
        self.files = [os.path.join(dp, f) for dp, dn, fn in os.walk(self.path) for f in fn]
        #self.files = os.listdir(self.path)
        self.randomness = random

    def iterate(self):

        if self.randomness:
            random.shuffle(self.files)

        mean_pixel = self.get_mean_pixel_value(MEAN_PIXEL_SAMPLES)

        for file_name in self.files:

            try:
                image = cv2.imread(file_name) / 255.0 

                # resize
                random_rescale_axe = np.random.uniform(low=RESIZE_AXIS_SIZE_MIN, high=RESIZE_AXIS_SIZE_MAX)
                smallest_axis = np.argmin(image.shape[0:2])
                scaling_factor = random_rescale_axe / image.shape[smallest_axis]
                image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
                image = image[...,::-1] # bgr 2 rgb

                # get random crop
                surplus = np.subtract(image.shape[0:2], CROP_SIZE).astype(np.float32)
                offset_x = int(np.random.uniform(high=surplus[0]))
                offset_y = int(np.random.uniform(high=surplus[1]))
                image = image[offset_x:offset_x+CROP_SIZE, offset_y:offset_y+CROP_SIZE]

                if(image.shape[0] != CROP_SIZE or image.shape[1] != CROP_SIZE):
                    print('image does not fit to crop size. Skipping {}...'.format(file_name))
                    continue

                # create random mask
                hole_width = int(np.random.uniform(low=HOLE_SIZE_MIN, high=HOLE_SIZE_MAX))
                hole_height = int(np.random.uniform(low=HOLE_SIZE_MIN, high=HOLE_SIZE_MAX))
                hole_offset_x = int(np.random.uniform(high=CROP_SIZE-hole_width))
                hole_offset_y = int(np.random.uniform(high=CROP_SIZE-hole_height))
                mask = np.ones((CROP_SIZE,CROP_SIZE))
                mask[hole_offset_x:hole_offset_x+hole_width, hole_offset_y:hole_offset_y+hole_height] = 0.0            
                mask = np.expand_dims(mask, axis=2)

                # apply mask to crop
                masked_image = mask * image
                invert_mask = mask*-1.0 + 1.0
                masked_image += mean_pixel * invert_mask

                # add mask as channel to input and make it RGB
                input_image = np.concatenate((masked_image, mask), axis=2)

                if(mask.shape[0] != CROP_SIZE or mask.shape[1] != CROP_SIZE):
                    print('mask does not fit crop size. Skipping {}...'.format(file_name))
                    continue

                if(input_image.shape[0] != CROP_SIZE or input_image.shape[1] != CROP_SIZE):
                    print('input does not fit crop size. Skipping {}...'.format(file_name))
                    continue

            except:
                print("Unexpected error during image preprocessing:", sys.exc_info())
                continue

            yield input_image, invert_mask, image


    def get_mean_pixel_value(self, samples=1000):
        
        mean_pixels = np.zeros((samples, 3))
        for i in range(samples):
            file_name = random.choice(self.files)
            image = cv2.imread(file_name)
            mean_pixels = image.mean(axis=(0,1)) / 255.0

        return mean_pixels.mean(axis=0)


    def get_tf_dataset(self):
        dataset = tf.data.Dataset.from_generator(generator=self.iterate, output_types=(tf.float32, tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([CROP_SIZE, CROP_SIZE, 4]),
                                                        tf.TensorShape([CROP_SIZE, CROP_SIZE, 1]),
                                                        tf.TensorShape([CROP_SIZE, CROP_SIZE, 3])
                                                                        ))

        return dataset
        


dataset = MaskedImageDataset('ic_fcn/data')

for x in dataset.iterate():
    a=0

