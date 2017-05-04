import tensorflow as tf

"""
Google Inception models
https://norman3.github.io/papers/docs/google_inception.html

Inception V1: 
    * parallel conv module
    * 5x5
    * auxiliary classifier
Inception V2: 
    * conv factorization
        * 5x5 => 3x3 *2
        * 3x3 => 1x3 * 3x1 (asymetric conv factorization)
Inception V3:
    * RMSProp, BN, Label smoothing
Inception V4: 
    * various modules are applied
Inception-ResNet:
    * add residual connection to inception module
"""

def preproc(x):
    # x = x*2 - 1.0
    # per-example mean subtraction (http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing)
    mean = tf.reduce_mean(x, axis=1, keep_dims=True)
    return x - mean

# chk: variable_scope name duplication
def conv_bn_activ_dropout(x, n_filters, kernel_size, strides, dropout_rate, training, seed, padding='SAME', activ_fn=tf.nn.relu, name="conv_bn_activ_dropout"):
    # with tf.variable_scope(name):
    net = tf.layers.conv2d(x, n_filters, kernel_size, strides=strides, padding=padding, use_bias=False,
                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=seed))
    net = tf.layers.batch_normalization(net, training=training)
    net = activ_fn(net)
    if dropout_rate > 0.0: # 0.0 dropout rate means no dropout
       net = tf.layers.dropout(net, rate=dropout_rate, training=training, seed=seed)

    return net

def conv_bn_activ(x, n_filters, kernel_size, strides, training, seed, padding='SAME', activ_fn=tf.nn.relu, name="conv_bn_activ"):
    return conv_bn_activ_dropout(x, n_filters, kernel_size, strides, 0.0, training, seed, padding, activ_fn, name)


# every c-dim /= 4
# valid => same (for mnist)
class InceptionLight:
    def conv_bn_activ(self, x, n_filters, kernel_size, strides=1, padding='SAME', activ_fn=tf.nn.relu, name="conv_bn_activ"):
        return conv_bn_activ(x, n_filters, kernel_size, strides, training=self.training, seed=self.seed, padding=padding, activ_fn=activ_fn, name=name)
    # |output channels| = |input channels| (inception block)
    # each inception block has fixed # of channels
    def inception_block_a(self, x, name='inception_a'):
        # num of channels: 96 = 24*4
        with tf.variable_scope(name):
            # with tf.variable_scope("branch1"):
            b1 = tf.layers.average_pooling2d(x, [3,3], 1, padding='SAME')
            b1 = self.conv_bn_activ(b1, 24, [1,1])

            # with tf.variable_scope("branch2"):
            b2 = self.conv_bn_activ(x, 24, [1,1])

            # with tf.variable_scope("branch3"):
            b3 = self.conv_bn_activ(x, 16, [1,1])
            b3 = self.conv_bn_activ(b3, 24, [3,3])

            # with tf.variable_scope("branch4"):
            b4 = self.conv_bn_activ(x, 16, [1,1])
            b4 = self.conv_bn_activ(b4, 24, [3,3])
            b4 = self.conv_bn_activ(b4, 24, [3,3])

            concat = tf.concat([b1, b2, b3, b4], axis=-1) # Q. -1 axis works well?
            return concat

    def inception_block_b(self, x, name='inception_b'):
        # num of channels: 256 = 32 + 96 + 64 + 64
        with tf.variable_scope(name):
            b1 = tf.layers.average_pooling2d(x, [3,3], 1, padding='SAME')
            b1 = self.conv_bn_activ(b1, 32, [1,1])

            b2 = self.conv_bn_activ(x, 96, [1,1])

            b3 = self.conv_bn_activ(x, 48, [1,1])
            b3 = self.conv_bn_activ(b3, 56, [1,7])
            b3 = self.conv_bn_activ(b3, 64, [7,1])

            b4 = self.conv_bn_activ(x, 48, [1,1])
            b4 = self.conv_bn_activ(b4, 48, [1,7])
            b4 = self.conv_bn_activ(b4, 48, [7,1])
            b4 = self.conv_bn_activ(b4, 64, [1,7])
            b4 = self.conv_bn_activ(b4, 64, [7,1])

            return tf.concat([b1, b2, b3, b4], axis=-1)

    def inception_block_c(self, x, name='inception_c'):
        # num of channels: 384 = 64*6
        with tf.variable_scope(name):
            b1 = tf.layers.average_pooling2d(x, [3,3], 1, padding='SAME')
            b1 = self.conv_bn_activ(b1, 64, [1,1])

            b2 = self.conv_bn_activ(x, 64, [1,1])

            b3 = self.conv_bn_activ(x, 96, [1,1])
            b3_1 = self.conv_bn_activ(b3, 64, [1,3])
            b3_2 = self.conv_bn_activ(b3, 64, [3,1])

            b4 = self.conv_bn_activ(x, 96, [1,1])
            b4 = self.conv_bn_activ(b4, 112, [1,3])
            b4 = self.conv_bn_activ(b4, 128, [3,1])
            b4_1 = self.conv_bn_activ(b4, 64, [3,1])
            b4_2 = self.conv_bn_activ(b4, 64, [1,3])

            return tf.concat([b1, b2, b3_1, b3_2, b4_1, b4_2], axis=-1)

    # reduction block do downsampling & change ndim of channel
    def reduction_block_a(self, x, name='reduction_a'):
        # 96 => 256 (= 96 + 64 + 96)
        # SAME : 28 > 14 > 7 > 4
        # VALID: 28 > 13 > 6 > 3
        with tf.variable_scope(name):
            b1 = tf.layers.max_pooling2d(x, [3,3], 2, padding='SAME') # 96
            
            b2 = self.conv_bn_activ(x, 96, [3,3], strides=2)
            
            b3 = self.conv_bn_activ(x, 48, [1,1])
            b3 = self.conv_bn_activ(b3, 56, [3,3])
            b3 = self.conv_bn_activ(b3, 64, [3,3], strides=2)

            return tf.concat([b1, b2, b3], axis=-1)

    def reduction_block_b(self, x, name='reduction_b'):
        # 256 => 384 (= 256 + 48 + 80)
        with tf.variable_scope(name):
            b1 = tf.layers.max_pooling2d(x, [3,3], 2, padding='SAME') # 256
            
            b2 = self.conv_bn_activ(x, 48, [1,1])
            b2 = self.conv_bn_activ(b2, 48, [3,3], strides=2)
            
            b3 = self.conv_bn_activ(x, 64, [1,1])
            b3 = self.conv_bn_activ(b3, 64, [1,7])
            b3 = self.conv_bn_activ(b3, 80, [7,1])
            b3 = self.conv_bn_activ(b3, 80, [3,3], strides=2)

            return tf.concat([b1, b2, b3], axis=-1)

    def build_inception(self, x):
        # 28 x 28 x 1
        # [28, 28, 1] => [28, 28, 96]
        with tf.variable_scope('pre_inception'):
            b1 = self.conv_bn_activ(x, 32, [1,1])
            b1 = self.conv_bn_activ(b1, 48, [3,3])

            b2 = self.conv_bn_activ(x, 32, [1,1])
            b2 = self.conv_bn_activ(b2, 32, [1,7])
            b2 = self.conv_bn_activ(b2, 32, [7,1])
            b2 = self.conv_bn_activ(b2, 48, [3,3])

            net = tf.concat([b1, b2], axis=-1)
            assert net.shape[1:] == [28, 28, 96]

        # inception A
        # [28, 28, 96]
        with tf.variable_scope("inception-A"):
            for i in range(2):
                net = self.inception_block_a(net, name="inception-block-a{}".format(i))
            assert net.shape[1:] == [28, 28, 96]

        # reduction A
        # [28, 28, 96] => [14, 14, 256]
        with tf.variable_scope("reduction-A"):
            net = self.reduction_block_a(net)
            assert net.shape[1:] == [14, 14, 256]

        # inception B
        with tf.variable_scope("inception-B"):
            for i in range(3):
                net = self.inception_block_b(net, name="inception-block-b{}".format(i))
            assert net.shape[1:] == [14, 14, 256]

        # reduction B
        # [14, 14, 256] => [7, 7, 384]
        with tf.variable_scope("reduction-B"):
            net = self.reduction_block_b(net)
            assert net.shape[1:] == [7, 7, 384]

        # inception C
        with tf.variable_scope("inception-C"):
            for i in range(1):
                net = self.inception_block_c(net, name="inception-block-c{}".format(i))
            assert net.shape[1:] == [7, 7, 384]

        # GAP + dense
        with tf.variable_scope("fc"):
            net = tf.reduce_mean(net, [1,2]) # GAP
            assert net.shape[1:] == [384]
            net = tf.layers.dropout(net, rate=0.2, training=self.training, seed=self.seed)
            logits = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed), name="logits")

        return logits


    def __init__(self, name='inception_light', lr=0.001, SEED=777):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, 784], name='X')
            self.y = tf.placeholder(tf.float32, [None, 10], name='y')
            self.training = tf.placeholder(tf.bool, name='training')
            self.seed = SEED

            x = preproc(self.X)
            x_img = tf.reshape(x, [-1, 28, 28, 1])

            logits = self.build_inception(x_img)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y), name="loss")

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
