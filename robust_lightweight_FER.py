import numpy
from scipy.spatial.qhull import QhullError
from scipy import spatial
spatial.QhullError = QhullError
from imgaug import augmenters as iaa
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
np.bool = np.bool_
from thop import profile
import torch
import torchvision
from retinaface import RetinaFace
from PIL import Image

class RetinaFaceExtractor:
    def __init__(self, image_path, align=True):
        self.image_path = image_path
        self.align = align

    def extract(self):
        return RetinaFace.extract_faces(img_path=self.image_path, align=self.align)

    def convert_to_tensor(self, extracted):
        if len(extracted) > 0:
            tensor = torchvision.transforms.ToTensor()(extracted[0].copy())
            tensor = torchvision.transforms.Resize((224,224), torchvision.transforms.functional.InterpolationMode.BILINEAR)(tensor)
            return torchvision.transforms.ToPILImage()(tensor)
        else:
            tensor = torchvision.transforms.ToTensor()(Image.open(self.image_path))
            tensor = torchvision.transforms.Resize((224,224), torchvision.transforms.functional.InterpolationMode.BILINEAR)(tensor)
            return torchvision.transforms.ToPILImage()(tensor)

class RandAug:
    def __init__(self, n: tuple, m: tuple, data: np.array):
        self.n = n
        self.m = m
        self.data = data.astype('uint8')

    def transform(self):
        rand_aug = iaa.RandAugment(n=self.n, m=self.m)
        return rand_aug(images=self.data).astype('float32')

class tripletattentionmechanism(tf.keras.layers.Layer):
    def __init__(self):
        super(tripletattentionmechanism, self).__init__()
        self.permute_lyr_wh = tf.keras.layers.Permute((1, 2, 3))
        self.permute_lyr_w = tf.keras.layers.Permute((1,3,2))
        self.permute_lyr_w_r = tf.keras.layers.Permute((1,3,2))
        self.permute_lyr_h = tf.keras.layers.Permute((3, 1, 2))
        self.permute_lyr_h_r = tf.keras.layers.Permute((3, 2, 1))
        self.concat_lyr_wh = tf.keras.layers.Concatenate(axis=-1)
        self.conv2d_lyr_wh = tf.keras.layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same', data_format='channels_last')
        self.batchnorm_wh = tf.keras.layers.BatchNormalization()
        self.sigmoid_wh = tf.keras.layers.Activation('sigmoid')
        self.skipconnect_wh = tf.keras.layers.Multiply()
        self.concat_lyr_w = tf.keras.layers.Concatenate(axis=-1)
        self.conv2d_lyr_w = tf.keras.layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same', data_format='channels_last')
        self.batchnorm_w = tf.keras.layers.BatchNormalization()
        self.sigmoid_w = tf.keras.layers.Activation('sigmoid')
        self.skipconnect_w = tf.keras.layers.Multiply()
        self.concat_lyr_h = tf.keras.layers.Concatenate(axis=-1)
        self.conv2d_lyr_h = tf.keras.layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same', data_format='channels_last')
        self.batchnorm_h = tf.keras.layers.BatchNormalization()
        self.sigmoid_h = tf.keras.layers.Activation('sigmoid')
        self.skipconnect_h = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Average()

    def call(self, inputs):
        x_1_1 = self.permute_lyr_wh(inputs)
        x1 = self.concat_lyr_wh([tf.expand_dims(tf.math.reduce_max(x_1_1, axis=-1), axis=-1),
                             tf.expand_dims(tf.math.reduce_mean(x_1_1, axis=-1), axis=-1)])
        x1 = self.conv2d_lyr_wh(x1)
        x1 = self.batchnorm_wh(x1)
        x1 = self.sigmoid_wh(x1)
        x1_result = self.skipconnect_wh([x1, x_1_1])

        x_2_1 = self.permute_lyr_w(inputs)
        x2 = self.concat_lyr_w([tf.expand_dims(tf.math.reduce_max(x_2_1, axis=-1), axis=-1),
                             tf.expand_dims(tf.math.reduce_mean(x_2_1, axis=-1), axis=-1)])
        x2 = self.conv2d_lyr_w(x2)
        x2 = self.batchnorm_w(x2)
        x2 = self.sigmoid_w(x2)
        x2 = self.skipconnect_w([x2, x_2_1])
        x2_result = self.permute_lyr_w_r(x2)

        x_3_1 = self.permute_lyr_h(inputs)
        x3 = self.concat_lyr_h([tf.expand_dims(tf.math.reduce_max(x_3_1, axis=-1), axis=-1),
                             tf.expand_dims(tf.math.reduce_mean(x_3_1, axis=-1), axis=-1)])
        x3 = self.conv2d_lyr_h(x3)
        x3 = self.batchnorm_h(x3)
        x3 = self.sigmoid_h(x3)
        x3 = self.skipconnect_h([x3, x_3_1])
        x3_result = self.permute_lyr_h_r(x3)

        return self.add([x1_result, x2_result, x3_result])

class fused_conv_blk(tf.keras.layers.Layer):
    def __init__(self, filters, strides, attention=False):
        super(fused_conv_blk, self).__init__()
        self.attention = attention
        self.filters = filters
        self.strides = strides
        self.conv2d_1 = layers.Conv2D(filters=self.filters, kernel_size=3, strides=self.strides, padding='same', use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')
        self.triplet_att = tripletattentionmechanism()
        self.conv2d_2 = layers.Conv2D(filters=self.filters, kernel_size=self.strides, strides=self.strides, padding='same', use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.add = layers.Add()

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.batchnorm1(x)
        x = self.relu1(x)

        if self.attention==True:
            x = self.triplet_att(x)

        x2 = self.conv2d_2(x)
        x2 = self.batchnorm2(x2)

        if self.strides !=2:
            return self.add([inputs, x2])
        else:
            return x2

class GhostNet_Module(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation=True):
        super(GhostNet_Module, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.conv2d = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=1, padding='same', use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')
        self.depthwise = layers.DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.strides, padding='same', use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.batchnorm1(x)
        x = self.relu1(x)

        x_2 = self.depthwise(x)
        x_2 = self.batchnorm2(x_2)

        if self.activation:
            x_2 = self.relu2(x_2)

        return self.add([x, x_2])

class GhostNet_BottleneckBlk(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation=True):
        super(GhostNet_BottleneckBlk, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.ghostnet_module_1 = GhostNet_Module(self.filters, self.kernel_size, strides=1, activation=True)
        self.ghostnet_module_2 = GhostNet_Module(self.filters, self.kernel_size, strides=1, activation=False)
        self.triplet_att = tripletattentionmechanism()
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.depthwise1 = tf.keras.layers.DepthwiseConv2D(kernel_size=1, strides=1, padding='same', use_bias=False)
        self.depthwise2 = layers.DepthwiseConv2D(kernel_size=self.kernel_size, strides=2, padding='same', use_bias=False)
        self.conv2d = layers.Conv2D(filters=self.filters, kernel_size=1, strides=1, padding='same', use_bias=False)
        self.add1 = layers.Add()
        self.add2 = layers.Add()

    def call(self, inputs):
        x1 = self.ghostnet_module_1(inputs)
        x1 = self.batchnorm1(x1)
        x1 = self.relu(x1)

        x1 = self.triplet_att(x1)

        x_2 = self.ghostnet_module_2(x1)
        x_2 = self.batchnorm2(x_2)

        b0, h0, w0, c0 = x_2.shape

        if self.strides>1:
            if c0!=self.filters*2:
                x = self.depthwise1(inputs)
                x = self.conv2d(x)

            x_3 = self.add1([x, x_2])
            return self.depthwise2(x_3)
        else:
            return self.add2([inputs, x_2])
        return x_3

class robust_lightweight_FER(tf.keras.Model):
    def __init__(self, in_shape, x_train):
        super(robust_lightweight_FER, self).__init__()
        self.input_ly = tf.keras.layers.Input(in_shape)
        self.normalizer = tf.keras.layers.Normalization(axis=None)
        self.normalizer.adapt(x_train)
        self.conv2d = layers.Conv2D(filters=8, kernel_size=5, strides=2, padding='same', use_bias=False, name='normal_conv_1')
        self.batchnorm = layers.BatchNormalization(name='normal_bn_1')
        self.relu = layers.Activation('relu', name='normal_relu')
        self.fused_blk1 = fused_conv_blk(filters=8, strides=1)
        self.fused_blk2 = fused_conv_blk(filters=8, strides=1)
        self.fused_blk3 = fused_conv_blk(filters=8, strides=1, attention=True)
        self.fused_blk4 = fused_conv_blk(filters=16, strides=2, attention=True)
        self.fused_blk5 = fused_conv_blk(filters=16, strides=1)
        self.fused_blk6 = fused_conv_blk(filters=16, strides=1, attention=True)
        self.ghostnet_blk1 = GhostNet_BottleneckBlk(filters=16, kernel_size=3, strides=2)
        self.ghostnet_blk2 = GhostNet_BottleneckBlk(filters=16, kernel_size=3, strides=1)
        self.ghostnet_blk3 = GhostNet_BottleneckBlk(filters=16, kernel_size=3, strides=1)
        self.ghostnet_blk4 = GhostNet_BottleneckBlk(filters=16, kernel_size=3, strides=1)
        self.ghostnet_blk5 = GhostNet_BottleneckBlk(filters=32, kernel_size=3, strides=2)
        self.ghostnet_blk6 = GhostNet_BottleneckBlk(filters=32, kernel_size=3, strides=1)
        self.ghostnet_blk7 = GhostNet_BottleneckBlk(filters=32, kernel_size=3, strides=1)
        self.ghostnet_blk8 = GhostNet_BottleneckBlk(filters=64, kernel_size=3, strides=2)
        self.ghostnet_blk9 = GhostNet_BottleneckBlk(filters=128, kernel_size=3, strides=2)
        self.ghostnet_blk10 = GhostNet_BottleneckBlk(filters=240, kernel_size=3, strides=2)
        self.global_average_pooling = layers.GlobalAveragePooling2D()
        self.conv2d_1 = layers.Dense(240, name='fc1')
        self.conv2d_2 = layers.Dense(7, name='classifier')

    def call(self, x):
        x = self.normalizer(x)
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.fused_blk1(x)
        x = self.fused_blk2(x)
        x = self.fused_blk3(x)
        x = self.fused_blk4(x)
        x = self.fused_blk5(x)
        x = self.fused_blk6(x)
        x = self.ghostnet_blk1(x)
        x = self.ghostnet_blk2(x)
        x = self.ghostnet_blk3(x)
        x = self.ghostnet_blk4(x)
        x = self.ghostnet_blk5(x)
        x = self.ghostnet_blk6(x)
        x = self.ghostnet_blk7(x)
        x = self.ghostnet_blk8(x)
        x = self.ghostnet_blk9(x)
        x = self.ghostnet_blk10(x)
        x = self.global_average_pooling(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        return x
