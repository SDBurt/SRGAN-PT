# ECE471-SRGAN

## Overview
This repository holds the final project deliverable for UVics ECE 471 class. This project is one of many implementations of the paper **Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network** by *Christian Ledig, et al.*

```
@article{DBLP:journals/corr/LedigTHCATTWS16,
author    = {Christian Ledig and
            Lucas Theis and
            Ferenc Huszar and
            Jose Caballero and
            Andrew P. Aitken and
            Alykhan Tejani and
            Johannes Totz and
            Zehan Wang and
            Wenzhe Shi},
title     = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
            Network},
journal   = {CoRR},
volume    = {abs/1609.04802},
year      = {2016},
url       = {http://arxiv.org/abs/1609.04802},
archivePrefix = {arXiv},
eprint    = {1609.04802},
timestamp = {Mon, 13 Aug 2018 16:48:38 +0200},
biburl    = {https://dblp.org/rec/bib/journals/corr/LedigTHCATTWS16},
bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Our approach was to imitate the implementation of this paper, and to see its significance to real-world noise. We created our own dataset for testing our theory that the results of the GAN was undoing the downsampling, ie. undoing the gaussian noise.

## Get Started

The following information is needed in order to get this project up and running on your system.

### Environment

1. Create a `virtualenv` using `virtualenv python=python3 .venv`. Run `source .venv/bin/activate` to start the environment, and `deactivate` to close it.
2. Install dependencies using `pip install -r requirements.txt`

### Data

The data for this project uses high resoltuion images. These images can come from a number of sources.
- Berkeley Segmentation Dataset: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- ImageNet: http://www.image-net.org/

Other datasets mention in the paper, such as *Set 5* and *Set 14* were discovered in [this](https://github.com/jbhuang0604/SelfExSR) github repo. Image sets are from their results, but the HR images are in there as well.

### Command Line Arguments

There are a few useful command line arguments.

- `data_dir`: (String) Path to image data
- `cropsize`: (Int) cropped size of HR image
- `pretrain`: (Bool) pretrain the generator prior to SRGAN
- `pretrain_epochs`: (Int) Number of pretraining epochs

A full list of command line parameters can be found in `config.py`

### Train

This project does not have the pretrained SRResNet (MSE pixelwise loss function), so pretraining is needed if this is the first run

**Note:** Once pretraining is done, the program will automatically begin training SRGAN

1. Run `python3 main.py pretrain True` to being the program
2. Grab a coffee or tea because this part takes some time

If the model has been pretrained before, then only `python3 main.py` is needed

### Tensorboard

This project uses tensorboard for viewing the training loss and images. In a separate terminal window, run `tensorboard --logdir logs` to run tensorboard locally. With this, you can view the loss in the *SCALARS* tab, and images in the *IMAGES* tab.