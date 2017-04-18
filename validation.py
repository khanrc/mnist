import tensorflow as tf
import vggnet
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np


class Validation:
    def __init__(self, sess, model, X, y):
        self.sess = sess
        self.model = model
        self.X = X
        self.y = y

    
    def plot(self, images, labels, predicted):
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


    # return: wrong_indices, predicted, 
    def wrong_indices(self, batch_size=None):
        N = self.X.shape[0]
        y_c = np.argmax(self.y, axis=1)
        pred_stack = np.empty(0, dtype='int32')
        if batch_size == None:
            batch_size = N
        for i in range(0, N, batch_size):
            X_batch = self.X[i:i + batch_size]
            y_batch = self.y[i:i + batch_size]

            feed = {
                self.model.X: X_batch,
                self.model.y: y_batch,
                self.model.training: False
            }
            
            pred = self.sess.run(self.model.pred, feed_dict=feed)
            pred_stack = np.hstack([pred_stack, pred])

        assert pred_stack.shape == y_c.shape
        return np.argwhere(np.equal(pred_stack, y_c) == False).T[0], pred_stack


    def wrong_answers_check(self, filename="wrong_answers.png"):
        # wrong answers check
        wrongs, predicted = self.wrong_indices(1000)

        fig = self.plot(self.X[wrongs], self.y[wrongs], predicted[wrongs])
        fig.savefig(filename)
        acc = 1 - len(wrongs) / float(len(predicted))

        return acc


    def check_from_checkpoint(self, model_path, CHECKPOINT_DIR="./checkpoint"):
        path = os.path.join(CHECKPOINT_DIR, model_path)
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint:
            saver = tf.train.Saver()

            for v in checkpoint.all_model_checkpoint_paths:
                if v.endswith("test"):
                    continue

                saver.restore(sess, v)
                acc = self.wrong_answers_check(filename="{}-wa.png".format(os.path.basename(v)))
                print("{}: {:.2%}".format(v, acc))


if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    model = vggnet.VGGNet(name="vggnet")
    validation = Validation(sess=sess, model=model, X=mnist.test.images, y=mnist.test.labels)
    validation.check_from_checkpoint("VGGAUG500")
    # check_from_checkpoint("VGGAUG500")
