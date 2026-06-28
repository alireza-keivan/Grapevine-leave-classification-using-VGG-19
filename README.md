# Grapevine Leaves Classification with Deep Learning
* TensorFlow
* Keras
* Transfer Learning
* Computer Vision
* Deep Learning

## Business problem
Manual classification of grapevine leaf varieties is often time-consuming and requires expertise. This project demonstrates how transfer learning with VGG19 can automatically classify grapevine leaf images. It offers a foundation for agricultural quality control and smart farming applications.



## Objective
Develop a deep learning image classification model capable of accurately distinguishing different grapevine leaf classes using transfer learning.


![image](https://github.com/alireza-keivan/Grapevine-leave-classification-using-VGG-19/blob/main/src/complete.png)
---

## 📌 Project at a Glance

| Feature | Details |
|---------|---------|
| **Problem** | Grapevine Leaf Image Classification |
| **Approach** | Transfer Learning |
| **Model** | VGG19 |
| **Framework** | TensorFlow / Keras |
| **Language** | Python |
| **Dataset** | Grapevine Leaves Image Dataset |
| **Number of Classes** | 5 |
| **Task Type** | Multi-Class Image Classification |

## 📘 Overview

This project demonstrates an end-to-end deep learning workflow for classifying grapevine leaf varieties using transfer learning with VGG19. It covers data preprocessing, augmentation, model training, evaluation, and prediction.

---

## 🔧 Steps in the Project

### 1. Dataset and Paths

The dataset path is defined, and each subdirectory corresponds to a specific grapevine leaf class.

---

### 2. Data Organization
#### Combine Image File Names and Extract Metadata

The dataset consists of five grapevine leaf classes. Images are automatically indexed and grouped into a structured dataframe for downstream preprocessing and training.

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
![image](https://github.com/alireza-keivan/Grapevine-leave-classification-using-VGG-19/blob/main/src/overall2.png)

```python
# Class Distribution
class_distribution = data_df['class_labels'].value_counts()
print("Class Distribution:\n", class_distribution)
```
This step ensures there are no missing values and provides an overview of the dataset structure.


---

### 3. Data Visualization
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

### 4. Image Augmentation
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
![image](https://github.com/alireza-keivan/Grapevine-leave-classification-using-VGG-19/blob/main/src/Screenshot%20from%202026-06-28%2016-19-37.png)

---

### 5. Transfer Learning with VGG19
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
The model was trained using the Adam optimizer for 50 epochs while monitoring validation accuracy and loss.

---

### 6. Model Evaluation
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
<p align="center">
  <kbd>
    <img src="https://github.com/alireza-keivan/leave-segmentation/blob/main/src/Model%20accuracy.png" alt="Framed Image" width="800">
    <img src="https://github.com/alireza-keivan/leave-segmentation/blob/main/src/Model%20loss.png" alt="Framed Image" width="800">
  </kbd>
</p>

<p align="center">
  <kbd>
    <img src="https://github.com/alireza-keivan/Grapevine-leave-classification-using-VGG-19/blob/main/src/confusion.png" alt="Framed Image" width="800">
  </kbd>
</p>

# 🚀 Key Takeaways

- Developed an end-to-end deep learning pipeline using TensorFlow and Keras.
- Applied transfer learning with VGG19 to classify five grapevine leaf varieties.
- Improved model robustness using image augmentation techniques.
- Evaluated model performance using training history and confusion matrix visualization.

- ## 🛠 Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn
