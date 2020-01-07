# Convolutional Neutral Network for Traffic Sign Recognition. 
In this project, [Russian Traffic Sign Database (RTSD)](http://graphics.cs.msu.ru/en/research/projects/rtsd) is used to classify traffic signs into 67 groups.
All code is represented in [Colab Jupyter Notebook file](main_script_EN.ipynb).


## RTSD info
RTSD consists of 3 parts: 
- *Classification*
- *Detection*
- *Full-Frames*.

*RTSD Classification* part has 2 groups of images: 
- *RTSD-r1* 
- *RTSD-r3*.

*RTSD-r1*: 67 classes, 32 983 images of (48, 48, 3) shape\
*RTSD-r3*: 106 classes, 93 654 RGB images with shapes between (16, 16, 3) and (320, 280, 3)

*RTSD-r1* has been selected for training.

## Project info
Main steps of project are: 
- analyzing and visualizing the RTSD images for traffic signs classification
- RTSD-r1 labeling error elimination
- creating and №1 training a convolutional neural network model
- data augmentation
- №2 training
- representing the results
- errors visualizing.

## Analyzing and visualizing the RTSD images for traffic signs classification
Class №48 takes 10.78% of images whereas class №10 takes up 0.01% of images.
</br></br><img src="./readme_resources/r1_base_en.png"></br></br>
RTSD-r3 classes are more disbalanced than RTSD-r1. Class №79 takes 25.74% of images whereas class №10 takes up 0.01% of images.
</br></br><img src="./readme_resources/r3_base_en.png"></br></br>

*RTSD-r1* has been selected for training. The following picture shows more clearly how *RTSD-r1* classes images are distributed:
</br></br><img src="./readme_resources/r1_fancy_bar_en.png"></br></br>

Examples of images per classes in RTSD-r1:
</br></br><img src="./readme_resources/ImagesSamples_en.png"></br></br>

RTSD-r1 has some problems with labeling and image displaying. For example some of images in class №1 should be labeled as №2.
