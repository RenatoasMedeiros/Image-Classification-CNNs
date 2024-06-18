import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization, Add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision training
set_global_policy('mixed_float16')

# Define constants
BATCH_SIZE = 64
IMG_SIZE = 150
NUM_CLASSES = 10  # Number of classes to identify
NUM_EPOCHS = 60  # Further increase number of epochs
LEARNING_RATE = 0.0001  # Slightly higher learning rate

# Define directories
train_dirs = ['./dataset/train/train1', './dataset/train/train2',
              './dataset/train/train3', './dataset/train/train5']
validation_dir = './dataset/validation'
test_dir = './dataset/test'

# Add more aggressive data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=60,  # Increase rotation range
    width_shift_range=0.3,  # Increase width shift range
    height_shift_range=0.3,  # Increase height shift range
    shear_range=0.3,  # Increase shear range
    zoom_range=0.3,  # Increase zoom range
    horizontal_flip=True,
    vertical_flip=True,  # Additional augmentation
    brightness_range=[0.6, 1.4],  # Increase brightness range
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create multiple train generators
train_generators = [train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical') for train_dir in train_dirs]

# Custom generator to merge multiple directories and repeat
def combined_generator(generators):
    while True:
        for generator in generators:
            for batch in generator:
                yield batch

train_generator = combined_generator(train_generators)

# Validation and test generators
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

# Load the pre-trained ResNet50 model without the top layer and adjust input shape
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Unfreeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = True

# Define the model with adjusted parameters to reduce overfitting
def residual_block(x, units):
    shortcut = x
    x = Dense(units, activation='relu', kernel_regularizer=l2(0.03))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(units, activation='relu', kernel_regularizer=l2(0.03))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    return x

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.03))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = residual_block(x, 512)
x = residual_block(x, 256)
x = residual_block(x, 128)
x = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=x)

# Use label smoothing
loss = CategoricalCrossentropy(label_smoothing=0.1)

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.99

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss=loss,
              metrics=['accuracy'])

model.summary()

# Define callbacks
os.makedirs('outputs', exist_ok=True)
checkpoint = ModelCheckpoint("models/best_model_resnet50_7.keras",
                             monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)  # Increased patience
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)  # More aggressive schedule
csv_logger = CSVLogger(
    'outputs/main_resnet50_all_unfrozen_L2_batch_64.csv', separator=',', append=False)
lr_scheduler = LearningRateScheduler(lr_scheduler)


total_train_samples = sum([gen.samples for gen in train_generators])
steps_per_epoch = total_train_samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stopping, reduce_lr, csv_logger, lr_scheduler]
)

# Plot training history and save the plot
plt.figure()
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
