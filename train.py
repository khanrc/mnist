import os
from argparse import ArgumentParser

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', default=150, help="Number of training epochs (default: 150)", type=int)
    parser.add_argument('--batch_size', default=128, help="Batch size (default: 128)", type=int)
    parser.add_argument('--learning_rate', default=0.001, help="Learning rate for ADAM (default: 0.001)", type=float)
    parser.add_argument('--save_dir', default='tmp', help="checkpoint & summaries save dir name (default: tmp)")
    parser.add_argument('--gpu_num', default=0, help="CUDA visible device (default: 0)")
    parser.add_argument("--model_name", help="vggnet / vggnet2 / resnet / wide_resnet / inception", required=True)
    parser.add_argument("--augmentation_type", default="affine", help="none / affine / align (default: affine)")
    parser.add_argument("--resnet_layer_n", default=3, help="6n+2: {3, 5, 7, 9 ... 18} (default: 3)", type=int)
    parser.add_argument("--ignore_exist_model", default=False, help="Overwrite new model to exist model (default: false)", type=bool)
    parser.add_argument("--gpu_memory_fraction", default=0.3, help="If this value is 0.0, allow_growth option is on (default: 0.3)", type=float)

    return parser

parser = build_parser()
FLAGS = parser.parse_args()
print("\nParameters:")
for attr, value in sorted(vars(FLAGS).items()):
    print("{}={}".format(attr.upper(), value))
print("")
if FLAGS.save_dir == "tmp":
    print("========== save_dir is tmp!!! ==========")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu_num)


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import vggnet, resnet, wide_resnet, inception_light
from solver import Solver
import time, datetime
import os, glob, shutil
import datagenerator
from utils import *


# tf.reset_default_graph()
SEED = 777
tf.set_random_seed(SEED)
np.random.seed(SEED)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
batch_size = FLAGS.batch_size
epoch_n = FLAGS.num_epochs
learning_rate = 0.001
N = mnist.train.num_examples

model = get_model(name=FLAGS.model_name, learning_rate=FLAGS.learning_rate, SEED=SEED, resnet_layer_n=FLAGS.resnet_layer_n)

SUMMARY_DIR = "./tmp/{}".format(FLAGS.save_dir)
CHECKPOINT_DIR = "./checkpoint/{}".format(FLAGS.save_dir)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


def train():
    config = tf.ConfigProto()
    if FLAGS.gpu_memory_fraction > 0.0:
        if FLAGS.gpu_memory_fraction < 1.0:
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    else:
        config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    solver = Solver(sess, model)

    # saver
    saver = tf.train.Saver()
    if FLAGS.save_dir != "tmp":
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            print("model exist")
            if FLAGS.ignore_exist_model == False:
            	exit()
            # try:
            #     saver.restore(sess, checkpoint.model_checkpoint_path)
            #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
            #     exit()
            # except:
            #     print("Error on loading old network weights")
        # else:
        #     print("Could not find old network weights")
    
    # reset summaries for our GPU
    for f in glob.glob(SUMMARY_DIR+"/*"):
        shutil.rmtree(f)

    # writers
    train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(SUMMARY_DIR + '/valid')
    test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test')

    datagen = datagenerator.Generator(FLAGS.augmentation_type, mnist)

    max_train_acc = 0
    max_valid_acc = 0

    # train
    sess.run(tf.global_variables_initializer())

    start_time = time.time()
    print("========== training start ==========")

    for epoch in range(epoch_n):
        # for _ in range(N // batch_size):
        for x, y in datagen.generate(batch_size=batch_size):
            _, train_loss = solver.train(x, y)

        train_loss, train_acc = solver.evaluate(mnist.train.images, mnist.train.labels, batch_size, writer=train_writer, step=epoch+1)
        valid_loss, valid_acc = solver.evaluate(mnist.validation.images, mnist.validation.labels, batch_size, writer=valid_writer, step=epoch+1)
        test_loss, test_acc = solver.evaluate(mnist.test.images, mnist.test.labels, batch_size, writer=test_writer, step=epoch+1)
        line = "[{}/{}] train: {:.4f}, {:.3%} / valid: {:.4f}, {:.2%} / test: {:.4f}, {:.2%}". \
        format(epoch+1, epoch_n, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc)
        print line,
        elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
        print("[{}]".format(elapsed_time))

        if train_acc > max_train_acc:
            max_train_acc = train_acc
            train_line = line
            saver.save(sess, CHECKPOINT_DIR + "/model-train")
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            valid_line = line
            saver.save(sess, CHECKPOINT_DIR + "/model-valid")

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
