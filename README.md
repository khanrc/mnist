# MNIST!

The aim of this project is to learn mnist classifier with very high accuracy (above 99.8% !)

based on python 2.7, tensorflow 0.12.

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

* VGG + all conv: 99.73%
* resnet-20: 99.63%
* resnet-32: 99.60%
* **majority voting (ensemble)**: 99.77%

Inception V4 and wide resnet are not yet tested.

## Learn to learn

* How to do better ensemble?
* Does elastic distortion works well?
* Hyperparameters
	* learning rate
		* decaying
	* batch size
* initialization
	* xavier vs. msra
* Which optimization method and params works better?
* Which activation function works better?
* Pre-processings
* Other models
	* BN-NIN
	* SqueezeNet
	* HighwayNet
	* FractalNet
	* etc ...