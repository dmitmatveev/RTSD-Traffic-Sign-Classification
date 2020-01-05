# Convolutional Neutral Network for Traffic Sign Recognition. 
In this project, [Russian Traffic Sign Database (RTSD)](http://graphics.cs.msu.ru/en/research/projects/rtsd) is used to classify traffic signs into 67 groups.

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
- analyzing and visualizing a RTSD images for traffic signs classification
- creating and training a convolutional neural network model
- data augmentation
- representing the results
- errors visualizing.
