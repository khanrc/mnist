# MNIST!

The aim of this project is to learn mnist classifier with very high accuracy
(above 99.7%)

## TODO

* [0] fix argparse - CUDA_VISIBLE_DEVICES
* fix reproducibility
* [1] add data manager module
    * TFRecord check
    * integrate fragmented train.py ... T.T
* [1] add data augmentation
    * elastic distortion (do do do)
        * simultaneou distortion is very expensive
        * affine transform is cheap
        * therefore, pre-distort only
    * augmented set
        * AlignMNIST
            * maybe better
            * http://www2.compute.dtu.dk/~sohau/augmentations/
        * InfiMNIST
        * https://github.com/CalculatedContent/big_deep_simple_mlp
* [2] implement other models
    * [2] resnet
    * [3] inception
    * maxout
    * nin
    * etc...

## DONE
* [done] add tensorboard (graph, summaries ...)
    * data smoothing in tensorboard can be changed to assign value directly
* [done] add command-line parameters to run (epochs ...)
    * maybe move to argparse
* [no] fix data preprocessing
    * I was confused. but we can test other preprocessing methods - width normalization, ...
    * chk
* [done] add validation module (wrong results check ...)
* [done] add ensemble
* [done] refact vggnet2 with vggnet
    * chk
* [done] make last epoch save point: 150
* [1] add data augmentation
    * [done] affine transition (rotate, shift, scale ...)
    * [done] elastic distortion
    * [no] GAN
* [done] add argparser with GPU command line control module
