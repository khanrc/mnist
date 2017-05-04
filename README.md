# MNIST!

The aim of this project is to learn mnist classifier with very high accuracy (above 99.8% !)

based on python 2.7, tensorflow 1.1.0.

## Models

* VGG like
	* VGG + all conv
	* VGG + inception V2 intuition
* ResNet like
	* Original ResNet (for CIFAR-10)
	* Pre-activation ResNet
	* Wide ResNet
* Inception like
	* Inception V4 for MNIST
	* Lightweight Inception V4 for MNIST

## Results

* VGG + all conv + batch size 64: 99.74%
* VGG + all conv + batch size 128: 99.68% 
* resnet-32: 99.68%
* **majority voting (ensemble)**: 99.76%

### Other models

* resnet-20: 99.68%
* INCEPTION lightweight: 99.65%
* wide resnet-14: 99.68%

## Usage

### Reproduce 99.76%

Note: `ensemble.py` indicates gpu=0

```
python ensemble.py
```

### train
```
$ python train.py --help
usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                [--learning_rate LEARNING_RATE] [--save_dir SAVE_DIR]
                [--gpu_num GPU_NUM] --model_name MODEL_NAME
                [--augmentation_type AUGMENTATION_TYPE]
                [--resnet_layer_n RESNET_LAYER_N]
                [--ignore_exist_model IGNORE_EXIST_MODEL]
                [--gpu_memory_fraction GPU_MEMORY_FRACTION]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 150)
  --batch_size BATCH_SIZE
                        Batch size (default: 128)
  --learning_rate LEARNING_RATE
                        Learning rate for ADAM (default: 0.001)
  --save_dir SAVE_DIR   checkpoint & summaries save dir name (default: tmp)
  --gpu_num GPU_NUM     CUDA visible device (default: 0)
  --model_name MODEL_NAME
                        vggnet / vggnet2 / resnet / wide_resnet / inception
  --augmentation_type AUGMENTATION_TYPE
                        none / affine / align (default: affine)
  --resnet_layer_n RESNET_LAYER_N
                        6n+2: {3, 5, 7, 9 ... 18} (default: 3)
  --ignore_exist_model IGNORE_EXIST_MODEL
                        Overwrite new model to exist model (default: false)
  --gpu_memory_fraction GPU_MEMORY_FRACTION
                        If this value is 0.0, allow_growth option is on
                        (default: 0.3)
```

## Learn to learn

* training NN tips & tricks
	* http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
	* http://www.topbots.com/14-design-patterns-improve-convolutional-neural-network-cnn-architecture/
* How to do better ensemble?
* Does elastic distortion works well?
* Hyperparameters
	* learning rate
		* decaying
	* batch size
* initialization
	* xavier vs. msra
	* https://github.com/hwalsuklee/tensorflow-mnist-MLP-batch_normalization-weight_initializers
* Which optimization method and params works better?
	* learning to learn by GD by GD
* Which activation function works better?
* Pre-processings
* Other models
	* SqueezeNet
	* FractalNet
	* ResNetXt
		* https://blog.waya.ai/deep-residual-learning-9610bb62c355
	* etc ...
