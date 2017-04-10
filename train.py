import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import models
from solver import Solver

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
SEED = 777

tf.reset_default_graph()

sess = tf.Session()

basic_cnn = models.VGG(lr=0.001, SEED=SEED)
solver = Solver(sess, basic_cnn)

tf.set_random_seed(SEED)
np.random.seed(SEED)

sess.run(tf.global_variables_initializer())

batch_size = 64
epoch_n = 120
N = mnist.train.num_examples

max_train_acc = 0
max_valid_acc = 0
max_test_acc = 0

# train
for epoch in range(epoch_n):
    for _ in range(N // batch_size):
        batches = mnist.train.next_batch(batch_size)
        _, train_loss = solver.train(batches[0], batches[1])
#         sess.run(solver, {X: batches[0], y: batches[1]})

    train_loss, train_acc = solver.evaluate(mnist.train.images, mnist.train.labels, 1000)
    valid_loss, valid_acc = solver.evaluate(mnist.validation.images, mnist.validation.labels, 1000)
    test_loss, test_acc = solver.evaluate(mnist.test.images, mnist.test.labels, 1000)
    line = "[{:0>2d}/{}] train: {:.4f}, {:.3%} / valid: {:.4f}, {:.2%} / test: {:.4f}, {:.2%}". \
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

