import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import vggnet
from solver import Solver
import time, datetime
import os, glob, shutil
from keras.preprocessing.image import ImageDataGenerator
from mnist_helpers import elastic_transform_wrapper

# from argparse import ArgumentParser

# def build_parser():
#   parser = ArgumentParser()
#   parser.add_argument('--num-epochs', dest="num_epochs", help="Number of training epochs (default: 100)")

# Params
# TODO:
# move to argparse... https://github.com/carpedm20/BEGAN-tensorflow/blob/master/config.py
# add GPU control 
tf.flags.DEFINE_integer("num_epochs", 150, "Number of training epochs (default: 150)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_runs", 0, "# of runs for tensorboard summaries (default: 0)")
tf.flags.DEFINE_string("save_dir", "tmp", "checkpoint subdirectory (default: tmp)")
tf.flags.DEFINE_integer("gpu_num", 0, "CUDA visible device (default: 0)")
tf.flags.DEFINE_string("model_name", "vggnet", "vggnet / vggnet2 (default: vggnet)")
tf.flags.DEFINE_integer("elastic_distortion", 0, "Use elastic distortion (default: 0)")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
if FLAGS.save_dir == "tmp":
    print("========== save_dir is tmp!!! ==========")
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu_num)

# tf.reset_default_graph()
SEED = 777
tf.set_random_seed(SEED)
np.random.seed(SEED)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
batch_size = FLAGS.batch_size
epoch_n = FLAGS.num_epochs
learning_rate = 0.001
N = mnist.train.num_examples

# right position..?
if FLAGS.model_name == "vggnet":
    model = vggnet.VGGNet(name="vggnet", lr=learning_rate, SEED=SEED)
elif FLAGS.model_name == "vggnet2":
    model = vggnet.VGGNet(name="vggnet2", lr=learning_rate, SEED=SEED)
else:
    print("Wrong model name!")
    exit()

SUMMARY_DIR = "./tmp/gpu{}".format(FLAGS.gpu_num)
CHECKPOINT_DIR = "./checkpoint/{}".format(FLAGS.save_dir)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# reset for coressponding RUN #
for f in glob.glob(SUMMARY_DIR+"/*"):
    shutil.rmtree(f)


def train():
    sess = tf.Session()
    # model = vggnet.VGGNet(lr=learning_rate, SEED=SEED)
    solver = Solver(sess, model)

    # writers
    train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(SUMMARY_DIR + '/valid')
    test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test')

    # saver
    saver = tf.train.Saver()
    # checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     try:
    #         saver.restore(sess, checkpoint.model_checkpoint_path)
    #         print("Successfully loaded:", checkpoint.model_checkpoint_path)

    #         # restore test
    #         train_loss, train_acc = solver.evaluate(mnist.train.images, mnist.train.labels, 1000, writer=train_writer, step=epoch+1)
    #         valid_loss, valid_acc = solver.evaluate(mnist.validation.images, mnist.validation.labels, 1000, writer=valid_writer, step=epoch+1)
    #         test_loss, test_acc = solver.evaluate(mnist.test.images, mnist.test.labels, 1000, writer=test_writer, step=epoch+1)
    #         line = "[{}/{}] train: {:.4f}, {:.3%} / valid: {:.4f}, {:.2%} / test: {:.4f}, {:.2%}". \
    #         format(epoch+1, epoch_n, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc)
    #         print(line)

    #         exit()
    #     except:
    #         print("Error on loading old network weights")
    # else:
    #     print("Could not find old network weights")


    max_train_acc = 0
    max_valid_acc = 0

    start_time = time.time()

    # train
    sess.run(tf.global_variables_initializer())
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

    train_x = np.reshape(mnist.train.images, [-1, 28, 28, 1])
    train_y = mnist.train.labels
    datagen_N = train_x.shape[0] // batch_size

    # for epoch in range(epoch_n):
    # datagen of Keras has no end
    # so, if we want to use epoch concept, the datagen loop have to be disconnected forcibly for each epoch.
    epoch = 0
    cnt = 0
    print("start train...")
    for x, y in datagen.flow(train_x, train_y, batch_size=batch_size, seed=SEED):
        if FLAGS.elastic_distortion == 1:
            x = elastic_transform_wrapper(x, negated=True) # kernel_dim=15, alpha=5.5, sigma=35, negated=True : soft distortion
        # TODO: fix reshaping
        x = np.reshape(x, [-1, 784])
        _, train_loss = solver.train(x, y)
        
        cnt += 1
        if cnt == datagen_N:
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

            cnt = 0
            epoch += 1

            if epoch == epoch_n:
                break

    saver.save(sess, CHECKPOINT_DIR + "/model-final")

    print("[train max] {}".format(train_line))
    print("[valid max] {}".format(valid_line))
    print("[  final  ] {}".format(line))
    elapsed_time = time.time() - start_time
    formatted = datetime.timedelta(seconds=int(elapsed_time))
    print("=== training time elapsed: {}s ===".format(formatted))


# validation

# import validation
# validation.wrong_answer_check(solver, mnist.test.images, mnist.test.labels, filename="wrong_answers.png")

if __name__ == '__main__':
    train()
