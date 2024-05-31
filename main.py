# %%
import os, shutil
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

train_dir = './dataset/train'
validation_dir = './dataset/validation'
test_dir = './dataset/test'

# Corrected string interpolation using f-strings
# train_cats_dirs = [f'./dataset/train/train{i}/003_cat' for i in [1, 2, 3, 5]]
# train_dogs_dirs = [f'./dataset/train/train{i}/005_dog' for i in [1, 2, 3, 5]]

train_cats_dir = './dataset/train/train1/003_cat'
train_dogs_dir = './dataset/train/train1/005_dog'

val_cats_dir = './dataset/validation/train4/003_cat'
val_dogs_dir = './dataset/validation/train4/005_dog'
test_cats_dir = './dataset/test/test/003_cat'
test_dogs_dir = './dataset/test/test/005_dog'

BATCH_SIZE=10
IMG_SIZE = 32

# Function to list files in directories
def list_files(dirs):
    for dir in dirs:
        if os.path.exists(dir):
            print(f"Files in {dir}: {os.listdir(dir)}")
        else:
            print(f"Directory {dir} does not exist")

# Debugging: List files in directories
# print("Training Cats Directories:")
# list_files(train_cats_dirs)
# print("\nTraining Dogs Directories:")
# list_files(train_dogs_dirs)
print("\nValidation Cats Directory:")
list_files([val_cats_dir])
print("\nValidation Dogs Directory:")
list_files([val_dogs_dir])
print("\nTesting Cats Directory:")
list_files([test_cats_dir])
print("\nTesting Dogs Directory:")
list_files([test_dogs_dir])


# # Count the number of images in each directory
# train_cats_count = sum(len(os.listdir(dir)) for dir in train_cats_dirs)
# train_dogs_count = sum(len(os.listdir(dir)) for dir in train_dogs_dirs)
val_cats_count = len(os.listdir(val_cats_dir))
val_dogs_count = len(os.listdir(val_dogs_dir))
test_cats_count = len(os.listdir(test_cats_dir))
test_dogs_count = len(os.listdir(test_dogs_dir))

print('total validation cat images:', val_cats_count)
print('total validation dog images:', val_dogs_count)
print('total testing cat images:', test_cats_count)
print('total testing dog images:', test_dogs_count)

# Debugging: List contents of directories
print("Contents of Training Directory:")
print(os.listdir(train_dir))

print("Contents of Validation Directory:")
print(os.listdir(validation_dir))

print("Contents of Test Directory:")
print(os.listdir(test_dir))



#Preprocessing the data


train_dataset = image_dataset_from_directory(train_dir,image_size=(IMG_SIZE,IMG_SIZE),batch_size=BATCH_SIZE)
validation_dataset = image_dataset_from_directory(validation_dir,image_size=(IMG_SIZE, IMG_SIZE),batch_size=BATCH_SIZE)
test_dataset = image_dataset_from_directory(test_dir,image_size=(IMG_SIZE, IMG_SIZE),batch_size=BATCH_SIZE)
train_dataset = image_dataset_from_directory(train_dir,image_size=(IMG_SIZE,IMG_SIZE),batch_size=BATCH_SIZE)


# Debugging: Check the first batch
for data_batch, labels_batch in train_dataset:
    print('Data batch shape:', data_batch.shape)
    print('Labels batch shape:', labels_batch.shape)
    break

# Visualizing some images
for data_batch, _ in train_dataset.take(1):
    for i in range(5):
        plt.imshow(data_batch[i].numpy().astype("uint8"))
        plt.show()


from tensorflow import keras
from keras import layers
from keras import models
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

import tensorflow as tf
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])


history = model.fit(train_dataset,epochs=30,validation_data=validation_dataset)


model = keras.models.load_model('modelo_fudido.h5')

val_loss, val_acc = model.evaluate(validation_dataset)
print('val_acc:', val_acc)