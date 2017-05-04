# coding: utf-8

import tensorflow as tf

"""
ResNet
http://florianmuellerklein.github.io/wRN_vs_pRN/

* Original resnet [1]
> * Bottleneck resnet [1]
* Pre-activation resnet [2]
* Wide resnet [3]

[1] Deep Residual Learning for Image Recognition
[2] Identity Mappings in Deep Residual Networks: BN/ReLU position change
[3] Wide Residual Networks: pRN + increased filter (k-times) + (dropout)
"""


def preproc(x):
    # x = x*2 - 1.0
    # per-example mean subtraction (http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing)
    mean = tf.reduce_mean(x, axis=1, keep_dims=True)
    return x - mean

class ResNet:
    def residual_block(self, x, output_channel, downsampling=False, name='res_block'):
        input_channel = int(x.shape[-1]) # get # of input channels

        if downsampling:
            stride = 2
        else:
            stride = 1

        with tf.variable_scope(name):
            with tf.variable_scope('conv1_in_block'):
                h1 = tf.layers.conv2d(x, output_channel, [3,3], strides=stride, padding='SAME', use_bias=False,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
                h1 = tf.layers.batch_normalization(h1, training=self.training)
                h1 = tf.nn.relu(h1)
            
            with tf.variable_scope('conv2_in_block'):
                h2 = tf.layers.conv2d(h1, output_channel, [3,3], strides=1, padding='SAME', use_bias=False,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
                h2 = tf.layers.batch_normalization(h2, training=self.training)
            
            # option A - zero padding for extra dimension => no need extra params
            # if downsampling:
            #     pooled_x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='valid')
            #     padded_x = tf.pad(pooled_x, [[0,0], [0,0], [0,0], [input_channel // 2, input_channel // 2]])
            # else:
            #     padded_x = x

            # option B - projection with 1x1 strided conv
            if downsampling:
                x = tf.layers.conv2d(x, output_channel, [1,1], strides=stride, padding='SAME', 
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))

            return tf.nn.relu(h2 + x)

    """
    In original resnet, 6n+2 layers for CIFAR-10.
    each feature map sizes: {32, 16, 8}
    # of filters: {16, 32, 64}
    """
    def build_net(self, x_img, layer_n):
        net = x_img

        # conv0: conv-bn-relu [-1, 28, 28, 16]
        # conv1: [-1, 28, 28, 16] * n
        # conv2: [-1, 14, 14, 32] * n
        # conv3: [-1,  7,  7, 64] * n
        # global average pooling
        # dense

        with tf.variable_scope("conv0"):
            net = tf.layers.conv2d(net, 16, [3,3], strides=1, padding='SAME', use_bias=False,
                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
            net = tf.layers.batch_normalization(net, training=self.training)
            net = tf.nn.relu(net)

        with tf.variable_scope("conv1"):
            for i in range(layer_n):
                net = self.residual_block(net, 16, name="resblock{}".format(i+1))
                assert net.shape[1:] == [28, 28, 16]

        with tf.variable_scope("conv2"):
            for i in range(layer_n):
                net = self.residual_block(net, 32, downsampling=(i==0), name="resblock{}".format(i+1))
                assert net.shape[1:] == [14, 14, 32]

        with tf.variable_scope("conv3"):
            for i in range(layer_n):
                net = self.residual_block(net, 64, downsampling=(i==0), name="resblock{}".format(i+1))
                assert net.shape[1:] == [7, 7, 64]

        with tf.variable_scope("fc"):
            net = tf.reduce_mean(net, [1,2]) # global average pooling
            assert net.shape[1:] == [64]

            logits = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed), name="logits")

        return logits

    def __init__(self, name='resnet', lr=0.001, layer_n=3, SEED=777):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, 784], name='X')
            self.y = tf.placeholder(tf.float32, [None, 10], name='y')
            self.training = tf.placeholder(tf.bool, name='training')
            self.seed = SEED

            x = preproc(self.X)
            x_img = tf.reshape(x, [-1, 28, 28, 1])

            logits = self.build_net(x_img, layer_n=layer_n)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y), name="loss")
#             self.loss = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=self.y))
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, name="train_op")
            
            self.pred = tf.argmax(logits, axis=1, name="prediction")
            self.prob = tf.nn.softmax(logits, name="softmax")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.y, axis=1)), tf.float32), name="accuracy")

            # summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.scalar("acc", self.accuracy),
            ])
