# Repruducing FILM in Tensorflow

## Introduction
This is a tensorflow implementation of the 'FILM: Frame Interpolation for Large Motion'

## Running the code

### train 
'python3 -m src.train'

### evaluation
'python3 -m src.eval'

### Training data
FILM is trained on a proprietary dataset with one million internet video clips, each comprising 3 frames.

- We use the publcicly available Vimeo-90k dataset ([Xue et al., 2019](https://arxiv.org/abs/1711.09078)), which is commonly used dataset for video frame interpolation

