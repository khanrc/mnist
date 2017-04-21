import numpy as np
import vggnet, resnet, wide_resnet, inception_light


def one_hot(dense, ndim=10):
    N = dense.shape[0]
    ret = np.zeros([N, ndim])
    ret[np.arange(N), dense] = 1
    return ret

def get_model(name, learning_rate=0.001, SEED=777, resnet_layer_n=3):
    # right position..?
    if name == "vggnet":
        model = vggnet.VGGNet(name="vggnet", lr=learning_rate, SEED=SEED)
    elif name == "vggnet2":
        model = vggnet.VGGNet(name="vggnet2", lr=learning_rate, SEED=SEED)
    elif name == "resnet":
        model = resnet.ResNet(name="resnet", lr=learning_rate, layer_n=resnet_layer_n, SEED=SEED)
    elif name == "wide_resnet":
        model = wide_resnet.WideResNet(name="wide_resnet", lr=learning_rate, layer_n=resnet_layer_n, SEED=SEED)
    elif name == "inception":
        model = inception_light.InceptionLight(name="inception_light", lr=learning_rate, SEED=SEED)
    else:
        assert False, "wrong model name"

    return model