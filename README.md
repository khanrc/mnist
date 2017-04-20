# MNIST!

The aim of this project is to learn mnist classifier with very high accuracy (above 99.8% !)

## TODO

* [1] modify train loss/acc calc
    * move to training step
    * summaries record method change into global step
* change data distortion, align augmentation epoch type
* [no] fix reproducibility
    * looks impossible
* [1] implement other models
    * [1] resnet
    * [1] inception
    * [2] maxout
    * [2] nin
    * etc...

## DONE
* [done] add data augmentation
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
* [done] add data manager module
    * TFRecord check
    * integrate fragmented train.py ... T.T
* [done] fix argparse - CUDA_VISIBLE_DEVICES
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
