# MNIST!

The aim of this project is to learn mnist classifier with very high accuracy
(above 99.7%)

## TODO

* add tensorboard (graph, summaries ...)
* add command-line parameters to run (epochs ...)
* add data manager module
* add validation module (wrong results check ...)
* fix data preprocessing
    * use train data statistics
* add data augmentation
    * affine transition (rotate, shift, scale ...)
        * keras image generator chk
    * augmented set
        * AlignMNIST
            * maybe better
            * http://www2.compute.dtu.dk/~sohau/augmentations/
        * InfiMNIST
        * https://github.com/CalculatedContent/big_deep_simple_mlp
    * elastic distortion
    * GAN
* add ensemble
* fix reproducibility
* implement other models
    * maxout
    * nin
    * etc...
