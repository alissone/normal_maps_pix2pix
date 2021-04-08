# Converting images into normal maps with pix2pix CNNs

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)



This repository contains a simple case study on generating normal maps from color images without depth information (or any other information whatsoever).
The results are reasonable as a proof of concept that a CNN is able to store a certain level of representation of a tridimentional space inside its latent space.

The actual training and testing is inside [main.ipynb](https://github.com/alissone/normal_maps_pix2pix/blob/main/main.ipynb) file. It uses a modified version of U-Net to work as pix2pix network, trained with the NYU v2 dataset, available at https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html.

![Result image](https://github.com/alissone/normal_maps_pix2pix/blob/main/result_sample.png?raw=true)

It's possible to see that the predictions C and F are very similar to the ground truth B and E, generated through the model from A and D images, respectively
