# Converting images into normal maps with pix2pix CNNs

This repository contains a simple case study on generating normal maps from color images without depth information (or any other information whatsoever).
The results are reasonable as a proof of concept that a CNN is able to store a certain level of representation of a tridimentional space inside its latent space.

The actual training and testing is inside `main.py` file. It uses a modified version of U-Net to work as pix2pix network, trained with the NYU v2 dataset, available at https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html.
