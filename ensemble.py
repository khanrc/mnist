import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import vggnet
from solver import Solver
import os

SEED = 777
tf.set_random_seed(SEED)
np.random.seed(SEED)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
N = mnist.train.num_examples

sess = tf.Session()
CHECKPOINT_DIR = "./checkpoint"
model_paths = ["VGG2-400", "VGG-300", "VGG-748"]

def one_hot_vectorize(indices, depth=10):
    idx = np.array(indices)
    ret = np.zeros([idx.shape[0], depth])
    ret[np.arange(idx.shape[0]), idx] = 1
    return ret

sess.run(tf.global_variables_initializer())

y_sum = np.zeros_like(mnist.test.labels)

# Walking every paths below CHECKPOINT_DIR
# for path in os.walk(CHECKPOINT_DIR):
#     checkpoint = tf.train.get_checkpoint_state(path[0])
#     if checkpoint:

# selective ensemble by model_paths
for subp in model_paths:
    path = os.path.join(CHECKPOINT_DIR, subp)
    checkpoint = tf.train.get_checkpoint_state(path)
    if checkpoint:
        if subp.startswith("VGG-"):
            model = vggnet.VGGNet(name="vggnet")
        elif subp.startswith("VGG2-"):
            model = vggnet.VGGNet(name="vggnet2")
        else:
            print("what model?")
            break
        #model = vggnet.VGGNet(SEED=SEED)
        solver = Solver(sess, model)
        saver = tf.train.Saver()

        for v in checkpoint.all_model_checkpoint_paths:
            if v.endswith("test"):
                continue

            saver.restore(sess, v)
            # test_loss, test_acc = solver.evaluate(mnist.test.images, mnist.test.labels)
            
            pred, acc = sess.run([model.pred, model.accuracy], 
                {model.X: mnist.test.images, model.y: mnist.test.labels, model.training: False})
            print("{}: {:.2%}".format(v, acc))
            
            # how to ensemble?
            # majority voting
            y_sum += one_hot_vectorize(pred)

corrects_sum = np.sum(np.equal(np.argmax(y_sum, 1), np.argmax(mnist.test.labels, 1)))
print("ensemble: {:.2%}".format(corrects_sum / float(y_sum.shape[0])))

