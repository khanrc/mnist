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

class WideResNet:
    # pRN & wRN
    # wRN is just (x4) increased filter pRN with decreased depth + dropout
    # in this case, we have to consider first block (no need BN & ReLU)
    def wide_residual_block(self, x, output_channel, dropout_rate=0.3, first_block=False, downsampling=False, name='wRN_block'):
        if downsampling:
            stride = 2
        else:
            stride = 1

        net = x

        with tf.variable_scope(name):
            with tf.variable_scope('conv1_in_block'):
                if first_block == False:
                    net = tf.layers.batch_normalization(net, training=self.training)
                    net = tf.nn.relu(net)
                net = tf.layers.conv2d(net, output_channel, [3,3], strides=stride, padding='SAME', use_bias=False,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
            
            if dropout_rate > 0.0:
                net = tf.layers.dropout(net, rate=dropout_rate, training=self.training, seed=self.seed, name='dropout_in_block')

            with tf.variable_scope('conv2_in_block'):
                net = tf.layers.batch_normalization(net, training=self.training)
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(net, output_channel, [3,3], strides=1, padding='SAME', use_bias=False,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))

            if downsampling:
                x = tf.layers.conv2d(x, output_channel, [1,1], strides=stride, padding='SAME', 
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
            # TODO: check wide resnet for first layer dimension matching
            # elif first_block: # first_block is not downsampling
            #     x = tf.layers.conv2d(x, output_channel, [1,1], strides=1, padding='SAME', 
            #         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))

            return net + x

    def build_wide_resnet(self, x_img, dropout_rate=0.3, layer_n=2):
        net = x_img

        # conv0: conv-bn-relu [-1, 28, 28, 16]
        # conv1: [-1, 28, 28, 64] * n
        # conv2: [-1, 14, 14, 128] * n
        # conv3: [-1,  7,  7, 256] * n
        # global average pooling
        # dense
        # widening factor = 4

        with tf.variable_scope("conv0"):
            # 원래 이게 64가 아니고 16인데, x dim matching 을 어케 해주는지 몰라서 일단 그냥 64로 맞춰놈
            net = tf.layers.conv2d(net, 64, [3,3], strides=1, padding='SAME', use_bias=False,
                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
            net = tf.layers.batch_normalization(net, training=self.training)
            net = tf.nn.relu(net)

        with tf.variable_scope("conv1"):
            for i in range(layer_n):
                net = self.wide_residual_block(net, 64, dropout_rate=dropout_rate, first_block=(i==0), name="resblock{}".format(i+1))
                assert net.shape[1:] == [28, 28, 64]

        with tf.variable_scope("conv2"):
            for i in range(layer_n):
                net = self.wide_residual_block(net, 128, dropout_rate=dropout_rate, downsampling=(i==0), name="resblock{}".format(i+1))
                assert net.shape[1:] == [14, 14, 128]

        with tf.variable_scope("conv3"):
            for i in range(layer_n):
                net = self.wide_residual_block(net, 256, dropout_rate=dropout_rate, downsampling=(i==0), name="resblock{}".format(i+1))
                assert net.shape[1:] == [7, 7, 256]

        with tf.variable_scope("fc"):
            net = tf.layers.batch_normalization(net, training=self.training)
            net = tf.nn.relu(net)
            net = tf.reduce_mean(net, [1,2]) # global average pooling
            assert net.shape[1:] == [256]

            logits = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed), name="logits")

        return logits

    def __init__(self, name='wide_resnet', lr=0.001, dropout_rate=0.3, layer_n=2, SEED=777):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, 784], name='X')
            self.y = tf.placeholder(tf.float32, [None, 10], name='y')
            self.training = tf.placeholder(tf.bool, name='training')
            self.seed = SEED

            x = preproc(self.X)
            x_img = tf.reshape(x, [-1, 28, 28, 1])

            logits = self.build_wide_resnet(x_img, dropout_rate=dropout_rate, layer_n=layer_n)

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
