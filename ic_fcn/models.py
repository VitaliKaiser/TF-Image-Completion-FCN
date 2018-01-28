import tensorflow as tf


def build_local_discriminator(input_features):

    last_layer = input_features
    with tf.name_scope("discriminator_local"):
        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2))
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=128,
            kernel_size=[5, 5],
            strides=(2, 2))
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[5, 5],
            strides=(2, 2))
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2))
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2))
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        flat = tf.contrib.layers.flatten(last_layer)
        logits = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=1024)
        
    return logits


def build_global_discriminator(input_features):

    last_layer = input_features
    with tf.name_scope("discriminator_global"):
        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2))

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=128,
            kernel_size=[5, 5],
            strides=(2, 2))

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[5, 5],
            strides=(2, 2))

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2))

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2))

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2))
        
        flat = tf.contrib.layers.flatten(last_layer)
        logits = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=1024)

    return logits


def build_completion_fcn(input_features):

    last_layer = input_features
    with tf.name_scope("completion_fcn"):
        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=64,
            kernel_size=[5, 5],
            strides=(1, 1),
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
            padding='same')

        last_layer = tf.layers.batch_normalization(inputs=last_layer)
        ###################################################
        
        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=128,
            kernel_size=[3, 3], ###
            strides=(2, 2),
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=128,
            kernel_size=[3, 3],
            strides=(1, 1), ###
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)
        ###################################################
        
        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[3, 3],
            strides=(2, 2), ####
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1), ###
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(1, 1), 
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(2, 2), ###
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(4, 4), ###
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(8, 8), ###
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(16, 16), ###
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(1, 1), ###
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)
        ###################################################

        last_layer = tf.layers.conv2d_transpose(inputs=last_layer, ###
            filters=128,
            kernel_size=[4, 4], ###
            strides=(2, 2), ###
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer, ###
            filters=256,
            kernel_size=[3, 3], ###
            strides=(1, 1), ###
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)
        ###################################################
        
        last_layer = tf.layers.conv2d_transpose(inputs=last_layer, ###
            filters=64, ###
            kernel_size=[4, 4], ###
            strides=(2, 2), ###
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer, ###
            filters=32, ###
            kernel_size=[3, 3], ###
            strides=(1, 1), ###
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
            padding='same')
        last_layer = tf.layers.batch_normalization(inputs=last_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer,
            filters=3, ###
            kernel_size=[3, 3],
            strides=(1, 1), 
            dilation_rate=(1, 1),
            activation=None, ###
            padding='same')

        

    return last_layer