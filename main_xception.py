import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from keras.utils import Sequence

# Define constants
BATCH_SIZE = 32
IMG_SIZE = 71  # Adjusted for Xception model
NUM_CLASSES = 10
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
EXPECTED_TRAIN_STEPS = 1250  # Number of steps per epoch (total_train_samples // BATCH_SIZE)

# Define directories
train_dir='./dataset/train'
train_dirs = ['./dataset/train/train1', './dataset/train/train2', './dataset/train/train3', './dataset/train/train5']
validation_dir = './dataset/validation'
test_dir = './dataset/test'

# Function to count files in each folder
def count_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

# Function to count files in each directory
def count_files_in_dirs(dirs):
    total_count = 0
    for dir_path in dirs:
        count = count_files(dir_path)
        total_count += count
        print(f"Number of files in {dir_path}: {count}")
    return total_count

# List all the train folders
train_folders = [folder for folder in os.listdir(train_dir) if folder.startswith('train')]

# Iterate over each train folder
for train_folder in train_folders:
    train_path = os.path.join(train_dir, train_folder)
    print("Path to train folder:", train_path)

# Count files in each directory
num_train_files = count_files(train_dir)
num_validation_files = count_files(validation_dir)
num_test_files = count_files(test_dir)

# Print the information
print(f"Number of files in train directory: {num_train_files}")
print(f"Number of files in validation directory: {num_validation_files}")
print(f"Number of files in test directory: {num_test_files}")

# Count files in each directory
total_train_images = count_files_in_dirs(train_dirs)

# Print the total number of images in the train dataset
print(f"Total number of images in train dataset: {total_train_images}")


# Create image data generators with preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create multiple train generators
train_generators = [train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical') for train_dir in train_dirs]

# Calculate the number of batches for each dataset
num_train_batches = sum(1 for _ in train_generators)

# Calculate the total number of images
total_train_images = num_train_batches * BATCH_SIZE * len(train_dirs)

# Print the information
print(f"Number of batches in train generator: {num_train_batches}")
print(f"Total number of images in train dataset: {total_train_images}")

# Create a custom data generator class
class CustomDataGenerator(Sequence):
    def __init__(self, generators):
        self.generators = generators

    def __len__(self):
        return sum(len(generator) for generator in self.generators)

    def __getitem__(self, idx):
        for generator in self.generators:
            if idx < len(generator):
                return generator[idx]
            idx -= len(generator)
        raise IndexError('Index out of range')

# Create a custom data generator instance
custom_train_generator = CustomDataGenerator(train_generators)

# Print some sample batches from the custom train generator
num_batches_to_print = 10  # You can adjust this number as needed

for batch_index, (images, labels) in enumerate(custom_train_generator):
    if batch_index >= num_batches_to_print:
        break
    print(f"Batch {batch_index + 1}:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")


# Define validation and test generators
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Load the pre-trained Xception model without the top layer
base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Define the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Define custom callback to save the best model only if the epoch is fully trained
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('batch') == EXPECTED_TRAIN_STEPS:
            super().on_epoch_end(epoch, logs)

# Define callbacks
checkpoint = CustomModelCheckpoint("best_model_xception.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Calculate steps per epoch
total_train_samples = sum([gen.samples for gen in train_generators])
steps_per_epoch = total_train_samples // BATCH_SIZE
print(f"total_train_samples = {total_train_samples}" )
print(f"steps_per_epoch = {steps_per_epoch}" )

# Train the model using the custom data generator
history = model.fit(
    custom_train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print("Test Accuracy:", accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('./plot_xception.png')
plt.show()
