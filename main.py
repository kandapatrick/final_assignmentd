import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
import PIL
from time import time
import tensorflow as tf
from keras.regularizers import l2
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, Dropout, BatchNormalization, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import pathlib

train_path = 'footwear/Test'
test_path = 'footwear/training'

data_dir = pathlib.Path(train_path)
img_width = 150 # Define the width of images
img_height = 150 # Define the height of images
batch_size = 32 # Define the size of the batch
# Create a dataset

# Training Data
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print()

# Validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
  train_path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print()
# Test Data

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_path,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
labels = os.listdir(data_dir) # Import the labels name from the training folder as a list of labels
count = {}
for i in labels:
    count[i] = len(list(data_dir.glob(f'{i}/*.jpg')))

pd.DataFrame(count, index = [0])
y = [values for values in count.values()]
plt.figure(figsize = (10,5))
sns.barplot(x = labels, y = y, palette = 'mako')
plt.title('The number of Images for each class', fontsize = 17)
plt.xlabel('Class Name')
plt.ylabel('Frequency')
plt.show()
class_names = train_ds.class_names
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
for images, labels in train_ds:
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    break
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)

model1 = Sequential([
           layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
           layers.Conv2D(filters = 16 ,kernel_size = (5, 5), padding='same', activation='relu', name = 'conv1'),
           layers.MaxPooling2D(pool_size = (2,2)),

           layers.Conv2D(filters = 32 ,kernel_size = (5, 5), padding='same', activation='relu', name = 'conv2'),
           layers.MaxPooling2D(pool_size = (2,2)),

           layers.Conv2D(filters = 64 ,kernel_size = (5, 5), padding='same', activation='relu', name = 'conv3'),
           layers.MaxPooling2D(pool_size=(2,2)),

           layers.Flatten(),
           layers.Dense(128, activation='relu'),
           Dropout(0.5),
           layers.Dense(num_classes, activation = 'softmax')
])
# Compile the model1
model1.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate = .0001),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
model1.summary()
history_model1 = model1.fit(train_ds, validation_data = val_ds, epochs = 5)
test_loss, test_accuracy = model1.evaluate(test_ds, batch_size=batch_size)
print(f'The loss result is {test_loss}')
print(f'The accuracy result is {test_accuracy}')
data_augmentation = tf.keras.Sequential(
  [
    layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
num_classes = len(class_names)

model2 = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),

    layers.Conv2D(filters=64, kernel_size=(5, 5), padding='valid', activation='relu', name='conv1'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu', name='conv2'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filters=16, kernel_size=(3, 3), padding='valid', activation='relu', name='conv3'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    Dropout(0.5),
    layers.Dense(units = num_classes, activation = 'softmax')
])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5)
optimizer = Adam(learning_rate=0.0001)

# Compile the model
model2.compile(
    optimizer= optimizer,
loss=SparseCategoricalCrossentropy(),
         metrics=['accuracy'])
model2.summary()
history_model2 = model2.fit(train_ds, validation_data = val_ds, epochs = 55)
acc = history_model2.history['accuracy']
val_acc = history_model2.history['val_accuracy']

loss=history_model2.history['loss']
val_loss=history_model2.history['val_loss']
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (19,7))
sns.set_style("darkgrid")

ax[0].plot(acc, '*-',label = 'Training accuracy')
ax[0].plot(val_acc, '*-',label = 'Validation accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Epochs & Training Accuracy', fontsize = 17)
ax[0].legend(loc='best')


ax[1].plot(loss, '*-',label = 'Training loss')
ax[1].plot(val_loss, '*-',label = 'Validation loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('loss')
ax[1].set_title('Epochs & loss', fontsize = 17)
ax[1].legend(loc='best')
sns.set_style("darkgrid")

plt.show()
predictions = model2.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

labels = []
for _, label in test_ds:
    labels.append(label)

y_true = []
for i in labels:
    for j in i:
        y_true.append(int(j))

print(y_pred[:10])
print(y_true[:10])
print(classification_report(y_true,y_pred))
