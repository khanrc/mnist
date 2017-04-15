# import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# refactoring to class...?

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


def wrong_answers_check(solver, test_images, test_labels, filename="wrong_answers.png"):
    # wrong answers check
    wrongs, predicted = solver.wrong_indices(test_images, test_labels, 1000)

    fig = plot(test_images[wrongs], test_labels[wrongs], predicted[wrongs])
    fig.savefig(filename)
