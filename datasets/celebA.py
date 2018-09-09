import os
import tensorflow as tf

from NetBluePrint.util_ops.imageDirectoryLoader import imageDirectoryLoader
from NetBluePrint.core.dataset import dataset

class celebA(dataset):
    def __init__(self, batchsize = 64, resize_dim=[96,96], central_crop =False , random_crop =False):
        super(celebA, self).__init__(batchsize)
        with tf.name_scope("celebA"):
            path = os.path.split(os.path.abspath(__file__))[0] + "/celebA/"

            #print(path)
            self.d_loader = imageDirectoryLoader(batchsize, path, subdir=False)

            s = self.d_loader.getShape()

            self.x_dim = s[0]
            self.y_dim = s[1]
            self.depth = s[2]
            self.batch = self.d_loader.getBatch()
            if central_crop:
                cropper = lambda x: tf.image.central_crop(x, 0.7)
                self.batch = tf.map_fn(cropper, self.batch)
            elif random_crop:
                cropper = lambda x: tf.random_crop(x, [130, 130, 3])
                self.batch = tf.map_fn(cropper, self.batch)
            if resize_dim != None:
                self.batch = tf.image.resize_bilinear(self.batch, resize_dim)
                self.x_dim = resize_dim[0]
                self.y_dim = resize_dim[1]

            self.batch = tf.reshape(self.batch, [-1, self.x_dim, self.y_dim, 3])

            self.data_dict["image"] = self.batch
            #self.coord = self.d_loader.coord