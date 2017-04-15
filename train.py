# when import tensorflow, both GPU memory is allocated (in 2-GPU system)
# therefore, we have to use 'CUDA_VISIBLE_DEVICES' before import tensorflow
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import vggnet
from solver import Solver
import time, datetime
import os, glob, shutil

# Params
# TODO:
# move to argparse... https://github.com/carpedm20/BEGAN-tensorflow/blob/master/config.py
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_runs", 0, "# of runs for tensorboard summaries (default: 0)")
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
batch_size = FLAGS.batch_size
epoch_n = FLAGS.num_epochs
learning_rate = 0.001
N = mnist.train.num_examples

SUMMARY_DIR = "./tmp/run{}".format(FLAGS.num_runs)
CHECKPOINT_DIR = "./checkpoint/run{}".format(FLAGS.num_runs)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# reset for coressponding RUN #
for f in glob.glob(SUMMARY_DIR+"/*"):
    shutil.rmtree(f)


sess = tf.Session()
model = vggnet.VGGNet(lr=learning_rate, SEED=SEED)
solver = Solver(sess, model)

# writers
train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', sess.graph)
valid_writer = tf.summary.FileWriter(SUMMARY_DIR + '/valid')
test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test')

# saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)

        # restore test
        train_loss, train_acc = solver.evaluate(mnist.train.images, mnist.train.labels, 1000, writer=train_writer, step=epoch+1)
        valid_loss, valid_acc = solver.evaluate(mnist.validation.images, mnist.validation.labels, 1000, writer=valid_writer, step=epoch+1)
        test_loss, test_acc = solver.evaluate(mnist.test.images, mnist.test.labels, 1000, writer=test_writer, step=epoch+1)
        line = "[{}/{}] train: {:.4f}, {:.3%} / valid: {:.4f}, {:.2%} / test: {:.4f}, {:.2%}". \
        format(epoch+1, epoch_n, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc)
        print(line)

        exit()
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")


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
        saver.save(sess, CHECKPOINT_DIR + "/model-train")
    if valid_acc > max_valid_acc:
        max_valid_acc = valid_acc
        valid_line = line
        saver.save(sess, CHECKPOINT_DIR + "/model-valid")
    if test_acc > max_test_acc:
        max_test_acc = test_acc
        test_line = line
        saver.save(sess, CHECKPOINT_DIR + "/model-test")

print("[train max] {}".format(train_line))
print("[valid max] {}".format(valid_line))
print("[ test max] {}".format(test_line))
elapsed_time = time.time() - start_time
formatted = datetime.timedelta(seconds=int(elapsed_time))
print("=== training time elapsed: {}s ===".format(formatted))


# validation

# import validation
# validation.wrong_answer_check(solver, mnist.test.images, mnist.test.labels, filename="wrong_answers.png")
