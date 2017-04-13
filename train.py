import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import vggnet
from solver import Solver
import time
import datetime
import os

# Params
# TODO:
# move to argparse... https://github.com/carpedm20/BEGAN-tensorflow/blob/master/config.py
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_runs", 1, "# of runs for tensorboard summaries (default: 1)")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# tf.reset_default_graph()
SEED = 777
tf.set_random_seed(SEED)
np.random.seed(SEED)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
batch_size = 128
epoch_n = FLAGS.num_epochs
learning_rate = 0.001
N = mnist.train.num_examples

SUMMARY_DIR = "./tmp/run{}".format(FLAGS.num_runs)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

sess = tf.Session()
model = vggnet.VGGNet(lr=learning_rate, SEED=SEED)
solver = Solver(sess, model)

train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', sess.graph)
valid_writer = tf.summary.FileWriter(SUMMARY_DIR + '/valid')
test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test')

max_train_acc = 0
max_valid_acc = 0
max_test_acc = 0

start_time = time.time()

# train
sess.run(tf.global_variables_initializer())

for epoch in range(epoch_n):
    for _ in range(N // batch_size):
        batches = mnist.train.next_batch(batch_size)
        _, train_loss = solver.train(batches[0], batches[1])

    train_loss, train_acc = solver.evaluate(mnist.train.images, mnist.train.labels, 1000, writer=train_writer, step=epoch+1)
    valid_loss, valid_acc = solver.evaluate(mnist.validation.images, mnist.validation.labels, 1000, writer=valid_writer, step=epoch+1)
    test_loss, test_acc = solver.evaluate(mnist.test.images, mnist.test.labels, 1000, writer=test_writer, step=epoch+1)
    line = "[{}/{}] train: {:.4f}, {:.3%} / valid: {:.4f}, {:.2%} / test: {:.4f}, {:.2%}". \
    format(epoch+1, epoch_n, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc)
    print(line)

    if train_acc > max_train_acc:
        max_train_acc = train_acc
        train_line = line
    if valid_acc > max_valid_acc:
        max_valid_acc = valid_acc
        valid_line = line
    if test_acc > max_test_acc:
        max_test_acc = test_acc
        test_line = line

print("[train max] {}".format(train_line))
print("[valid max] {}".format(valid_line))
print("[ test max] {}".format(test_line))
elapsed_time = time.time() - start_time
formatted = datetime.timedelta(seconds=int(elapsed_time))
print("=== training time elapsed: {}s ===".format(formatted))

# wrong answers check
wrongs, predicted = solver.wrong_indices(mnist.test.images, mnist.test.labels, 1000)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot(images, labels, predicted):
    l = len(images)
    height = (l+9) // 10
    fig = plt.figure(figsize=(10,height+1))
    gs = gridspec.GridSpec(height,10)
    gs.update(wspace=0.05, hspace=0.05)
    corrects = np.argmax(labels, axis=1) # one-hot => single

    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_title("{} ({})".format(corrects[i], predicted[i]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(image.reshape(28, 28), cmap='Greys')

    return fig

# plt.imshow(mnist.test.images[0].reshape(28, 28), cmap='Greys')
# plt.show()

fig = plot(mnist.test.images[wrongs], mnist.test.labels[wrongs], predicted[wrongs])
fig.savefig('wrong_answers.png')

