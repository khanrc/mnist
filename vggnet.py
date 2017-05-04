import tensorflow as tf


def preproc(x):
    # x = x*2 - 1.0
    # per-example mean subtraction (http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing)
    mean = tf.reduce_mean(x, axis=1, keep_dims=True)
    return x - mean

def conv_bn_activ_dropout(name, x, n_filters, kernel_size, strides, dropout_rate, training, seed, padding='SAME', activ_fn=tf.nn.relu):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(x, n_filters, kernel_size, strides=strides, padding=padding, use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        net = tf.layers.batch_normalization(net, training=training)
        net = activ_fn(net)
        if dropout_rate > 0.0: # 0.0 dropout rate means no dropout
           net = tf.layers.dropout(net, rate=dropout_rate, training=training, seed=seed)

    return net

class VGGNet:
    def build_vggnet(self, x_img, dropout_rate, SEED):
        # hidden layers
        net = x_img
        n_filters = 64
        for i in range(3):
            net = conv_bn_activ_dropout(name="3x3conv{}-{}".format(i+1, 1), x=net, n_filters=n_filters, kernel_size=[3,3], strides=1, 
                dropout_rate=dropout_rate, training=self.training, seed=SEED)

            net = conv_bn_activ_dropout(name="3x3conv{}-{}".format(i+1, 2), x=net, n_filters=n_filters, kernel_size=[3,3], strides=1, 
                dropout_rate=dropout_rate, training=self.training, seed=SEED)

            if i == 2: # for last layer - add 1x1 convolution
                net = conv_bn_activ_dropout(name="1x1conv", x=net, n_filters=n_filters, kernel_size=[1,1], strides=1, 
                    dropout_rate=dropout_rate, training=self.training, seed=SEED)
            
            # 5x5 strided pooling
            n_filters *= 2
            net = conv_bn_activ_dropout(name="5x5stridepool{}".format(i+1), x=net, n_filters=n_filters, kernel_size=[5,5], strides=2, 
                dropout_rate=dropout_rate, training=self.training, seed=SEED)

        # x: [28, 28, 1]
        # h1: [14, 14, 64]
        # h2: [7, 7, 128]
        # h3: [4, 4, 256]
        # 4096 -> 10
        
        net = tf.contrib.layers.flatten(net)
#             net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
#             net = tf.layers.dropout(net, rate=0.5, training=self.training)
        logits = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=SEED), name="logits")

        return logits

    def build_vggnet2(self, x_img, dropout_rate, SEED):
        # hidden layers
        net = x_img
        n_filters = 64
        for i in range(3):
            net = conv_bn_activ_dropout(name="3x3conv{}-{}".format(i+1, 1), x=net, n_filters=n_filters, kernel_size=[3,3], strides=1, 
                dropout_rate=dropout_rate, training=self.training, seed=SEED)

            net = conv_bn_activ_dropout(name="3x3conv{}-{}".format(i+1, 2), x=net, n_filters=n_filters, kernel_size=[3,3], strides=1, 
                dropout_rate=dropout_rate, training=self.training, seed=SEED)

            if i == 2: # for last layer - add 1x1 convolution
                net = conv_bn_activ_dropout(name="1x1conv", x=net, n_filters=n_filters, kernel_size=[1,1], strides=1, 
                    dropout_rate=dropout_rate, training=self.training, seed=SEED)
            
            # strided pooling + maxpooling
            # InceptionV2 style: Rethinking the inception architecture
            # http://laonple.blog.me/220716782369
            net1 = conv_bn_activ_dropout(name="3x3stridepool{}".format(i+1), x=net, n_filters=n_filters, kernel_size=[3,3], strides=2, 
                dropout_rate=0.0, training=self.training, seed=SEED)
            net2 = tf.layers.max_pooling2d(net, [2, 2], 2, padding='SAME', name='2x2maxpool{}'.format(i+1))
            net = tf.concat([net1, net2], axis=3, name="concat{}".format(i+1)) ## add to channel
            net = tf.layers.dropout(net, rate=dropout_rate, training=self.training, seed=SEED)

            n_filters *= 2
        
        net = tf.contrib.layers.flatten(net)
#             net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
#             net = tf.layers.dropout(net, rate=0.5, training=self.training)
        logits = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=SEED), name="logits")

        return logits

    def __init__(self, name, lr=0.001, SEED=777):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, 784], name='X')
            self.y = tf.placeholder(tf.float32, [None, 10], name='y')
            self.training = tf.placeholder(tf.bool, name='training')
            
            x = preproc(self.X)
            x_img = tf.reshape(x, [-1, 28, 28, 1])
            if name == "vggnet":
                logits = self.build_vggnet(x_img, dropout_rate=0.3, SEED=SEED)
            elif name == "vggnet2":
                logits = self.build_vggnet2(x_img, dropout_rate=0.3, SEED=SEED)
            else:
                print("wrong name!!")
                exit()
            
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

