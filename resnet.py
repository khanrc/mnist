import tensorflow as tf

# TODO:
# preproc should be moved to the data management module
# => this preproc does not have dependency to training set
def preproc(x):
    # x = x*2 - 1.0
    # per-example mean subtraction (http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing)
    mean = tf.reduce_mean(x, axis=1, keep_dims=True)
    return x - mean

class ResNet:
	# SEED to self.SEED
	# https://github.com/wenxinxu/resnet-in-tensorflow
	# TODO:
	# * check first layer
	# * add increase_dim
	def residual_module(self, name='res_module' x, n_filters, kernel_size):
		with tf.variable_scope(name):
			h1 = tf.layers.batch_normalization(x, training=self.training)
			h1 = tf.nn.relu(h1)
			h1 = tf.layers.conv2d(h1, n_filters, kernel_size, strides=1, padding='SAME', use_bias=False,
								  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=self.SEED))
			
			h2 = tf.layers.batch_normalization(h1, training=self.training)
			h2 = tf.nn.relu(h2)
			h2 = tf.layers.conv2d(h2, n_filters, kernel_size, strides=1, padding='SAME', use_bias=False,
								  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=self.SEED))
			
		return h2 + x

    def __init__(self, name='ResNet', lr=0.001, SEED=777):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, 784], name='X')
            self.y = tf.placeholder(tf.float32, [None, 10], name='y')
            self.training = tf.placeholder(tf.bool, name='training')
            self.SEED = SEED

            x = preproc(self.X)
            x_img = tf.reshape(x, [-1, 28, 28, 1])

            # hidden layers
            net = x_img
            n_filters = 64
            for i in range(3):
                net = tf.layers.conv2d(net, n_filters, [3,3], strides=1, padding='SAME', use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=self.SEED))
                net = tf.layers.batch_normalization(net, training=self.training)
                net = tf.nn.relu(net)
                net = tf.layers.dropout(net, rate=0.3, training=self.training, seed=self.SEED)

                net = tf.layers.conv2d(net, n_filters, [3,3], strides=1, padding='SAME', use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=self.SEED))
                net = tf.layers.batch_normalization(net, training=self.training)
                net = tf.nn.relu(net)
                net = tf.layers.dropout(net, rate=0.3, training=self.training, seed=self.SEED)
 
                if i == 2: # for last layer - add 1x1 convolution
                    net = tf.layers.conv2d(net, n_filters, [1,1], strides=1, padding='SAME', use_bias=False,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=self.SEED))
                    net = tf.layers.batch_normalization(net, training=self.training)
                    net = tf.nn.relu(net)
                    net = tf.layers.dropout(net, rate=0.3, training=self.training, seed=self.SEED)
                
# strided pooling: all convnet, DCGAN ...
# [5,5] => [3,3] ?
# STRIVING FOR SIMPLICITY, THE ALL CONVOLUTIONAL NET: http://arxiv.org/pdf/1412.6806v3.pdf
                net = tf.layers.conv2d(net, n_filters, [5,5], strides=2, padding='SAME', use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=self.SEED))
#                 net = tf.layers.max_pooling2d(net, pool_size=[2,2], strides=2)
                net = tf.layers.batch_normalization(net, training=self.training)
                net = tf.nn.relu(net)
                net = tf.layers.dropout(net, rate=0.3, training=self.training, seed=self.SEED)

                n_filters *= 2
            
            # x: [28, 28, 1]
            # h1: [14, 14, 64]
            # h2: [7, 7, 128]
            # h3: [4, 4, 256]
            # 4096 -> 1024 -> 10
            
            net = tf.contrib.layers.flatten(net)
#             net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
#             net = tf.layers.dropout(net, rate=0.5, training=self.training)
            logits = tf.layers.dense(net, 10, weights_initializer=tf.contrib.layers.xavier_initializer(seed=self.SEED))
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
#             self.loss = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=self.y))
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)    
            
            self.pred = tf.argmax(logits, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.y, axis=1)), tf.float32))
