CHECKPOINT_DIR = "./checkpoint"
model_paths = ["RES32", "VGG1000", "VGG-xavier-64"]
# model_paths = ["RES20", "RES32", "VGG1000-0.0001", "VGG1000-0.0005", "VGG1000"]
GPU_num = 0
batch_size = 1000

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_num)

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from utils import *

SEED = 777
tf.set_random_seed(SEED)
np.random.seed(SEED)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
N = mnist.test.num_examples

y_sum = np.zeros_like(mnist.test.labels)
prob_sum = np.zeros_like(mnist.test.labels)

# selective ensemble by model_paths
for subp in model_paths:
    path = os.path.join(CHECKPOINT_DIR, subp)
    checkpoint = tf.train.get_checkpoint_state(path)
    if checkpoint:
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ensemble_weight = 1.0

            if subp.startswith("VGG"):
                model = get_model(name="vggnet")
                # ensemble_weight = 2.0
            elif subp.startswith("RES"):
                layer_n = int(subp[3:])
                layer_n = (layer_n-2)/6
                model = get_model(name="resnet", resnet_layer_n=layer_n)
            else:
                print("what model?")
                break
            
            saver = tf.train.Saver()
            
            for v in checkpoint.all_model_checkpoint_paths:
                if not v.endswith("valid"):
                    continue

                saver.restore(sess, v) 
                
                total_acc = 0
                pred = np.zeros([N], dtype=int)
                prob = np.zeros([N, 10], dtype=float)

                for i in range(0, N, batch_size):
                    x_batch = mnist.test.images[i:i+batch_size]
                    y_batch = mnist.test.labels[i:i+batch_size]
                    step_acc, step_pred, step_prob = sess.run([model.accuracy, model.pred, model.prob], {model.X: x_batch, model.y: y_batch, model.training: False})
                    pred[i:i+batch_size] = step_pred
                    prob[i:i+batch_size] = step_prob * ensemble_weight
                    total_acc += step_acc * x_batch.shape[0]

                total_acc /= N
                print("{}: {:.2%}".format(v, total_acc))
                
                # how to ensemble?
                # majority voting
                y_sum += one_hot(pred) * ensemble_weight
                prob_sum += prob

                
voting_acc = np.average(np.equal(np.argmax(y_sum, 1), np.argmax(mnist.test.labels, 1)))
prob_avg_acc = np.average(np.equal(np.argmax(prob_sum, 1), np.argmax(mnist.test.labels, 1)))
print("majority voting ensemble: {:.2%}".format(voting_acc))
#print("probability averaging ensemble: {:.2%}".format(prob_avg_acc))


