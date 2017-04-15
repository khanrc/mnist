# MNIST!

The aim of this project is to learn mnist classifier with very high accuracy
(above 99.7%)

## TODO

* [1] fix reproducibility
* [2] add data manager module
* [2] add data augmentation
    * affine transition (rotate, shift, scale ...)
        * keras image generator chk
    * augmented set
        * AlignMNIST
            * maybe better
            * http://www2.compute.dtu.dk/~sohau/augmentations/
        * InfiMNIST
        * https://github.com/CalculatedContent/big_deep_simple_mlp
    * elastic distortion
        * mnist-helper
            * https://github.com/vsvinayak/mnist-helper
    * GAN
* add argparser with GPU command line control module
* implement other models
    * resnet
    * inception
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
* [done] add validation module (wrong results check ...)
* [done] add ensemble
