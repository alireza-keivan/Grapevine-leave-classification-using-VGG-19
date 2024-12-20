# Grapevine Leaves Classification with Deep Learning

This project focuses on classifying grapevine leaves into different classes using **Deep Learning** techniques. The project leverages **Transfer Learning** with the **VGG19** architecture and includes detailed visualizations, data preprocessing, and model evaluation.

---

## 📘 Overview

The goal of this project is to classify grapevine leaves into 5 distinct classes using images provided in the **Grapevine Leaves Image Dataset**. The project includes:
- Extensive **Exploratory Data Analysis (EDA)** with **Seaborn** and **Matplotlib**.
- An **Image Augmentation Pipeline** for robust model training.
- A **Deep Learning Model** built using **Transfer Learning** with **VGG19**.
- Visualizations inspired by [this Kaggle notebook](https://www.kaggle.com/code/dj67rockers/leaves-classification).

---

## 🔧 Steps in the Project

### 1. Importing Libraries
```python
import os
import gc
import sys
import random
import pickle

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import (
    Input, Dense, Activation, Dropout, Flatten, BatchNormalization,
    Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from livelossplot import PlotLossesKeras

from skimage.feature import hog, canny
from skimage.filters import sobel
from skimage import color

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from PIL import Image
import cv2

from tf_explain.core.activations import ExtractActivations
from tf_explain.core.grad_cam import GradCAM

from tensorflow.keras.utils import get_file
from keras.preprocessing.image import ImageDataGenerator
```
This section imports all the necessary libraries for data handling, visualization, and building deep learning models.  

---

### 2. Dataset and Paths
```python
BASE_PATH = r'P:/1-uni/machine-learning/Grapevine_Leaves_Image_Dataset'
PATH_AK = os.path.join(BASE_PATH, 'Ak')
PATH_ALA_IDRIS = os.path.join(BASE_PATH, 'Ala_Idris')
PATH_BUZGULU = os.path.join(BASE_PATH, 'Buzgulu')
PATH_DIMNIT = os.path.join(BASE_PATH, 'Dimnit')
PATH_NAZLI = os.path.join(BASE_PATH, 'Nazli')
```
The dataset path is defined, and each subdirectory corresponds to a specific grapevine leaf class.

---

### 3. Data Preparation
#### Combine Image File Names and Extract Metadata
```python
data_df = pd.DataFrame()

# Combine image file names into a list
data_df['image_names'] = (
    os.listdir(PATH_AK) 
    + os.listdir(PATH_ALA_IDRIS) 
    + os.listdir(PATH_BUZGULU) 
    + os.listdir(PATH_DIMNIT) 
    + os.listdir(PATH_NAZLI)
)

# Extract class labels and file paths
class_labels = []
file_paths = []

for img_name in data_df['image_names']:
    label = img_name.split(' (')[0]  # Extract class label
    class_labels.append(label)
    file_paths.append(os.path.join(BASE_PATH, label, img_name))

# Add these to the DataFrame
data_df['class_labels'] = class_labels
data_df['file_paths'] = file_paths

plt.figure(figsize=(5,5))
class_cnt = main_df.groupby(['classes']).size().reset_index(name = 'counts')
colors = sns.color_palette('Paired')[0:9]
plt.pie(class_cnt['counts'], labels=class_cnt['classes'], colors=colors, autopct='%1.1f%%')
plt.legend(loc='upper right')
plt.show()
```
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/pie.png)
Here, we process the dataset by extracting class labels (based on file names) and constructing full file paths for loading images later.

#### Dataset Inspection
```python
# Check for missing values
missing_values = data_df.isna().sum()
print("Missing Values:\n", missing_values)

# Print the number of unique classes
unique_classes = len(data_df['class_labels'].value_counts())
print('Number of Unique Leaf Classes:', unique_classes)
```
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/samples%20ak.png)
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/samples%20ala0idris.png)
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/samples%20buzgulu.png)
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/samples%20dimnit.png)
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/samples%20nazli.png)
```python
# Class Distribution
class_distribution = data_df['class_labels'].value_counts()
print("Class Distribution:\n", class_distribution)
```
This step ensures there are no missing values and provides an overview of the dataset structure.


---

### 4. Data Visualization (Inspired by Kaggle Work)
#### Distribution of Classes
```python
sns.set_theme(style="whitegrid", palette="pastel", font="serif")
sns.set(rc={'figure.figsize': (15, 12)})

dist_plot = sns.countplot(x=data_df['class_labels'], color='#1f77b4')

# Customizing
dist_plot.set_title('Distribution of Leaf Classes\n', fontsize=20, fontweight='bold', pad=20)
dist_plot.set_ylabel('Count', fontsize=15, labelpad=10)
dist_plot.set_xlabel('Leaf Classes', fontsize=15, labelpad=10)

# Annotate bar counts
for bar in dist_plot.patches:
    count = int(bar.get_height())
    dist_plot.annotate(
        f'{count}', 
        (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
        ha='center', va='center', 
        xytext=(0, 8), 
        textcoords='offset points', 
        fontsize=12
    )

sns.despine()
```
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/samples%201%20.png)
This visualization shows the distribution of leaf classes and ensures balanced data representation.

---

### 5. Image Augmentation
#### Data Augmentation Using `ImageDataGenerator`
```python
vgg_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.10,
    brightness_range=[0.6,1.4],
    channel_shift_range=0.7,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
) 
```
We apply augmentation techniques like rotation, zoom, brightness adjustment, and flips to make the model more robust to unseen data.
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/Augmented%20images.png)
---

### 6. Transfer Learning with VGG19
#### Building the Model
```python
vgg19 = VGG19(include_top = False, input_shape = (227,227,3), weights = 'imagenet')

# Freeze all convolutional layers
for layer in vgg19.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(vgg19.output)
predictions = Dense(5, activation='softmax')(x)

model_vgg = Model(inputs = vgg19.input, outputs = predictions)
```
We use a pre-trained **VGG19** model with frozen convolutional layers to leverage its feature extraction capabilities. A custom dense layer is added for classification.

#### Training the Model
```python
model_vgg.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
history_vgg = model_vgg.fit(
      train_generator_vgg,
      validation_data=val_generator_vgg,
      epochs=50,
      verbose=2)
```
The model is compiled and trained using the Adam optimizer for 50 epochs, with accuracy and loss tracked.

---

### 7. Model Evaluation
#### Plotting Accuracy and Loss
```python
plt.figure(figsize=(15,5))
plt.plot(history_vgg.history['accuracy'])
plt.plot(history_vgg.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.figure(figsize=(15,5))
plt.plot(history_vgg.history['loss'])
plt.plot(history_vgg.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
```
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/Model%20accuracy.png)
![image](https://github.com/alireza-keivan/leave-segmentation/blob/main/src/Model%20loss.png)
