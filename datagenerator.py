# Data generator
# TODO:
# * distroted.npy regenerate

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from utils import one_hot
import time


# train: [:, 794]
# validation, test: images / labels

def Generator(aug_type, mnist):
    if aug_type == "none":
        datagen = NormalGenerator(mnist)
    elif aug_type == "affine":
        datagen = AffineGenerator(mnist)
    elif aug_type == "align":
        datagen = AlignGenerator()
    elif aug_type == "distortion":
        datagen = DistortionGenerator()
    else:
        assert False, "augmentation type error [{}]".format(aug_type)

    return datagen

class NormalGenerator():
    def __init__(self, mnist):
        self.mnist = mnist
        self.train = np.concatenate([self.mnist.train.images, self.mnist.train.labels], axis=1)
        self.N = self.train.shape[0]

    def generate(self, batch_size=64):
        np.random.shuffle(self.train)
        for i in range(0, self.N, batch_size):
            batch = self.train[i:i+batch_size]
            yield batch[:, :784], batch[:, 784:]

# no overhead comparision with normal gen
class AffineGenerator():
    def __init__(self, mnist):
        from keras.preprocessing.image import ImageDataGenerator
        
        self.mnist = mnist
        self.datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
        self.train_x = np.reshape(self.mnist.train.images, [-1, 28, 28, 1])
        self.train_y = self.mnist.train.labels

    def generate(self, batch_size=64):
        cnt = 0
        batch_n = self.train_x.shape[0] // batch_size
        for x, y in self.datagen.flow(self.train_x, self.train_y, batch_size=batch_size):
            ret_x = x.reshape(-1, 784)
            yield ret_x, y

            cnt += 1
            if cnt == batch_n:
                break

# painful loading time... [380s]
class AlignGenerator():
    def __init__(self, filename="alignmnist.npz"):
        loading_start = time.time()
        print "1. file load & reshape ...",
        align = np.load(filename)
        train_x = align['x'].reshape(-1, 784)
        train_y = one_hot(align['y'].astype(int))
        print("[{:.1f}s]".format(time.time()-loading_start))

        print "2. normalize ...",
        train_x = train_x / 255.0
        print("[{:.1f}s]".format(time.time()-loading_start))
        
        print("3. concat ...")
        self.train = np.concatenate([train_x, train_y], axis=1)
        print("=== loading done! === [{:.1f}s]".format(time.time()-loading_start))
        self.epoch = -1

    # no generation overhead
    def generate(self, batch_size=64):
        # 60000 * 76, first 1 epoch is original
        self.epoch += 1
        if self.epoch == 76:
            self.epoch = 0

        N = 60000
        st_pos = self.epoch * N
        ed_pos = (self.epoch+1) * N
        data = self.train[st_pos:ed_pos] # data for each epoch
        np.random.shuffle(data)
        for i in range(0, N, batch_size):
            batch = data[i:i+batch_size]
            yield batch[:, :784], batch[:, 784:]

# class DistortionGenerator():
#     def __init__(self, filename="distorted.npz")
#         Generator.__init__(self)
