# DataTalksClub Kitchenware Classification Competition

Using convolution neural network (cnn) with data augmentation techniques and transfer learning to classifiy kitchenware images to 6 classes:
 - cups
 - glasses
 - plates
 - spoons
 - forks
 - knives

 # Dataset

This project uses a dataset with more than `9400` images.

# Overview
This dataset contains images of different kitchenware.

### Files

* `train.csv` - the training set (Image IDs and classes)
* `test.csv` - the test set (Just image IDs)
* `sample_submission.csv` - a sample submission file in the correct format
* `images/` - the images in the JPEG format

# Contents of the folder 

# Exploratory Data Analysis
See the [Kitchenware_EDA.ipynb](./notebooks/Kitchenware_EDA.ipynb) for this task.

### Image sizes statistics
![sizes](./figures/sizes.png)
The images have various and varied sizes, ranging from 39 Kilopixels to 976 Kilopixels, also more than half of the images are concentrated at 750 Kpixels. This is bad news since the tensors must imperatively have the same size.

### The width to heigth ratio (lx/ly)
![lxy_ratio](./figures/ratio_lx_ly.png)
Most images are vertical.

### By width (lx)
![width](./figures/width.png)
The width varies from 233 to 1000, and it is concentrated on 1000.
### By heigth (ly)
![heigth](./figures/ly.png)
The heigth varies from 174 to 1000, and it is concentrated on 750.
### Labels statistics
![labels](./figures/labels.png)
Not all labels are representend equaly.
# Deployment of model

# Virtual Environment/venv

# Test the project

# Want to Contribute?
* Fork 🍴 the repository and send PRs.
* Do ⭐ this repository if you like the content.

**Connect with me:**

<p align="center">
  <a href="https://ma.linkedin.com/in/abdeljebbar-boubekri-656b30192" target="blank"><img align="center" src="https://www.vectorlogo.zone/logos/linkedin/linkedin-tile.svg" alt="https://ma.linkedin.com/in/abdeljebbar-boubekri-656b30192" height="30" width="30" /></a>
  <a href="https://www.twitter.com/marokart/" target="blank"><img align="center"  src="https://img.icons8.com/color/48/000000/twitter--v2.png" alt="https://www.twitter.com/marokart/" height="30" width="30" /></a>
  <a href="https://www.kaggle.com/dajebbar" target="blank">
    <img align="center" src="https://img.icons8.com/external-tal-revivo-shadow-tal-revivo/38/external-kaggle-an-online-community-of-data-scientists-and-machine-learners-owned-by-google-logo-shadow-tal-revivo.png" alt="https://www.kaggle.com/dajebbar" height="30" width="30" /></a>
  
</p>


