# Convolutional Neutral Network for Traffic Sign Recognition. 
In this project, [Russian Traffic Sign Database (RTSD)](http://graphics.cs.msu.ru/en/research/projects/rtsd) is used. RTSD-r1 (with 32 983 images) was selected to classify traffic signs into 67 groups.\
Results:
- 99.485% accuracy on validation dataset (51 of 9 895 are wrong predicted)
- 99.961% accuracy on train dataset (9 of 23 088 images are wrong predicted)

- All code is represented in [Colab Jupyter Notebook file](main_script_EN.ipynb).
- All results are visualised with [images](readme-resources).

## RTSD info
! ***RTSD-r1 contains several labels errors. Check the "[RTSD-r1 labeling error elimination](##RTSD--r1-labeling-error-elimination)" part of the project!***

RTSD consists of 3 parts: 
- *Classification*
- *Detection*
- *Full-Frames*.

*RTSD Classification* part has 2 groups of images: 
- *RTSD-r1*: 67 classes, 32 983 RGB images of (48, 48, 3) shape 
- *RTSD-r3*: 106 classes, 93 654 RGB images with shapes between (16, 16, 3) and (320, 280, 3)

*RTSD-r1* has been selected for training.

## Colab info
[Google Colaboratory](https://colab.research.google.com/) was used for the project. It is a Google research project created to help disseminate machine learning education and research. It's a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud. *Runtime type* was changed to GPU (free GPU NVIDIA Tesla K80 with 13 GB RAM), so all calculations were carried out on it.

## Project info
Python modules versions:
```python
Python version:  3.6.9 (default, Nov  7 2019, 10:44:02) 
[GCC 8.3.0]

Numpy version:  1.17.4
Pandas version:  0.25.3
matplotlib version:  3.1.2
cv2 version:  4.1.2
keras version:  2.2.5
sklearn version:  0.21.3
skimage version:  0.15.0
SeaBorn version:  0.9.0
```
Main steps of project are: 
- analyzing and visualizing the RTSD images for traffic signs classification
- *RTSD-r1* labeling error elimination
- creating and №1 training a convolutional neural network model
- data augmentation
- №2 training
- representing the results
- errors visualizing.

## Analyzing and visualizing the RTSD images for traffic signs classification
Class №48 takes 10.78% of images whereas class №10 takes up 0.01% of images.
</br></br><img src="./readme_resources/r1_base_en.png"></br></br>
*RTSD-r3* classes are more disbalanced than *RTSD-r1*. Class №79 takes 25.74% of images whereas class №10 takes up 0.01% of images.
</br></br><img src="./readme_resources/r3_base_en.png"></br></br>

*RTSD-r1* has been selected for training. The following picture shows more clearly how *RTSD-r1* classes images are distributed:
</br></br><img src="./readme_resources/r1_fancy_bar_en.png"></br></br>

Examples of images per classes in *RTSD-r1*:
</br></br><img src="./readme_resources/ImagesSamples_en.png"></br></br>
**Size of RTSD-r1: 1.7 GB**

## *RTSD-r1* labeling error elimination
*RTSD-r1* has some problems with labeling and image displaying. For example some of images in class №1 should be labeled as №2 like these images: *[0, 1, 2, 20, 21, 22, 23]*
</br></br><img src="./readme_resources/wrong labels/wrong n1.jpg"></br></br>

For example some of images in class №2 should be labeled as №1 like these images: *[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 28, 29, 32, 33, 34, 466, 467, 468, 469]*
</br></br><img src="./readme_resources/wrong labels/wrong n2.jpg"></br></br>

Labels mistakes in class №14:
</br></br><img src="./readme_resources/wrong labels/wrong n14.jpg"></br></br>

Labels mistakes in class №23:
</br></br><img src="./readme_resources/wrong labels/wrong n23.jpg"></br></br>

Labels mistakes in class №24:
</br></br><img src="./readme_resources/wrong labels/wrong n24.jpg"></br></br>

Also there are another images which labels should be changed. All problems solving in [code](main_script_EN.ipynb).

Labels mistakes in class №24 (should be horizontally flipped):
</br></br><img src="./readme_resources/wrong labels/wrong n49.jpg"></br></br>

## Creating and №1 training a convolutional neural network model
CNN model was created with [Keras library](https://keras.io/). [Netron](https://lutzroeder.github.io/netron/) service helped with the fancy model architecture visualisation. Model file like *model.hdf5* is needed for Netron service:
<p align="center">
  </br></br><img src="./readme_resources/model_architecture.png"></br></br>
</p>

Using of below method of *Keras model* object allows us to see Total, Trainable and Non-trainable params of CNN:
```python
model.summary()
```
```python
Total params: 2,795,875
Trainable params: 2,794,659
Non-trainable params: 1,216
```
Data was splited 70% (train) / 30% (validation).\
Train dataset shape: (23088, 48, 48, 3) -> Train labels shape: (23088, 67)\
Val dataset shape:   (9895, 48, 48, 3)  -> Val labels shape:   (9895, 67)

To prevent an overfitting *Keras.callbacks.EarlyStopping* was used:
```python
# condition of stopping the learning avoiding an overfitting
es_callback = EarlyStopping(monitor='val_loss',  # observed value
                            patience=3,  # during this number of epochs the val_loss should increase to stop learning
                            restore_best_weights=True)
```
To save the model *Keras.callbacks.ModelCheckpoint* was used:
```python
filepath = results_path + "/model-train1_new_en.hdf5"

# for saving the model after learning
checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                      monitor='val_loss',
                                      save_best_only=True)
```
Batch size:
```python
batch_size=5000
```
**Number of epochs: 47\
Past time: 146 s = 2.43 min**

Accuracies (train/val) during №1 training process:
</br></br><img src="./readme_resources/train1_acc_per_epochs_en.png"></br></br>

Loss-functions (train/val) during №1 training process:
</br></br><img src="./readme_resources/train1_loss_per_epochs_en.png"></br></br>

Confision matrix shows which classes may be confused with another ones. Diagonal elements are right predicted classes.
</br></br><img src="./readme_resources/train1_confusion_matrix_en.png"></br></br>

Excel-file with [Sklearn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) is available at this repository's [link](train1_val_report_en.xlsx)

Final results after training process №1:\
**validation loss: 0.09402, validation accuracy: 0.97504**

## Data augmentation
Because of not so mach data of some classes *recall* results are not so good for some of them. Results could be improved by increasing the number of images and by varying the images using [Keras ImageDataGenerator](https://keras.io/preprocessing/image/).\
Used transformation:
- rotation range (± 8°)
- zoom range (from 80% up to 120% of image)
-	width shift range (± 12%)
-	height shift range (± 12%)
- brightness range (from 30% up to 170% of image's brightness)
- *fill mode = 'nearest'* (blank space appeared after transformation will be filled with value of nearest pixel of original image)

### Solving the ImageDataGenerator problem after brightness_range usage
Because of *brightness_range* usage output images after transformation are [0, 255] scaled. To rescale them to [0, 1] we need to use *preprocessing_function*. It applies after all transformations so we define a *rescaling_func* that reduces each pixel of each image by 255.
```python
def rescaling_func(image):
    return image/255

datagen = ImageDataGenerator(
  rotation_range=8,
  zoom_range=[0.8, 1.2],
  width_shift_range=0.12,
  height_shift_range=0.12,
  brightness_range=[0.3, 1.7],
  fill_mode="nearest",
  preprocessing_function=rescaling_func)
```
Example of images transformation and varying is shown below. Left column - is an original images, other columns represents the images after randomly choosed transformation:
</br></br><img src="./readme_resources/data_augumentation_example_en.png"></br></br>
The new training dataset has been increased from 32 983 * 70% = 23 088 to 369 408 samples. The validation set remained untouched. The test set is unused.


## №2 training process
For each original image from *RTSD-r1* ImageDataGenerator creates 16 (number is choosed manually) new images with random transformations that were selected.\
**Number of epochs: 13\
Past time: about 8 300 s = 139 min = 2 h 30 min**

Accuracies (train/val) during №2 training process:
</br></br><img src="./readme_resources/train2_acc_per_epochs_en.png"></br></br>

Loss-functions (train/val) during №2 training process:
</br></br><img src="./readme_resources/train2_loss_per_epochs_en.png"></br></br>

Confusion matrix after №2 training process:
</br></br><img src="./readme_resources/train2_confusion_matrix_en.png"></br></br>

Final results after training process №2:\
**validation loss: 0.0194, validation accuracy: 0.99485**

*Final results after training process №1:\
validation loss:* 0.09402, *validation accuracy:* 0.97504\
**Improvements:\
validation loss: -0.07462, validation accuracy: +0.01981 (+1.981%)**

## Representing the results
Let show the results of the prediction on new images founded in web:
<p align="center">
  </br></br><img src="./readme_resources/results_examples_1_en.png"></br></br>
</p>
Some of them (*new.jpg* and *new_2.jpg*) weren't used in training (no images with these classes) but results are interesting. Other images were predicted correctly with probabilities between 82.255% and 99.999%

## Errors visualizing
Validation dataset contains 9 895 images and 51 of them are wrong predicted. Let show the results of wrong predictions over the validation dataset:
<p align="center">
  </br></br><img src="./readme_resources/incorrect_val_results_examples_new_en.png"></br></br>
</p>

Train dataset contains 23 088 images and only 9 of them are wrong predicted. The results of wrong predictions over the train dataset:
<p align="center">
  </br></br><img src="./readme_resources/incorrect_train_results_examples_new_en.png"></br></br>
</p>
Some of images are difficult to determine even by myself
