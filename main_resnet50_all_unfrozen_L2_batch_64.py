from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K
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

# %% metrics


class Precision(Metric):
    def __init__(self, name='precision', **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.predicted_positives = self.add_weight(
            name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.round(y_pred)
        y_true = K.cast(y_true, 'float32')
        self.true_positives.assign_add(K.sum(y_true * y_pred))
        self.predicted_positives.assign_add(K.sum(y_pred))

    def result(self):
        return self.true_positives / (self.predicted_positives + K.epsilon())

    def reset_states(self):
        self.true_positives.assign(0)
        self.predicted_positives.assign(0)


class Recall(Metric):
    def __init__(self, name='recall', **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.actual_positives = self.add_weight(name='ap', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.round(y_pred)
        y_true = K.cast(y_true, 'float32')
        self.true_positives.assign_add(K.sum(y_true * y_pred))
        self.actual_positives.assign_add(K.sum(y_true))

    def result(self):
        return self.true_positives / (self.actual_positives + K.epsilon())

    def reset_states(self):
        self.true_positives.assign(0)
        self.actual_positives.assign(0)


class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


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
    x = Dense(units, activation='relu', kernel_regularizer=l2(0.03))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    shortcut = Dense(units, kernel_regularizer=l2(0.03))(shortcut)
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
              metrics=['accuracy', Precision(), Recall(), F1Score()])

model.summary()

# Define callbacks
os.makedirs('outputs', exist_ok=True)
checkpoint = ModelCheckpoint("models/best_model_main_resnet50_all_unfrozen_L2_batch_64.keras",
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

# Plot training history
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(history.history['val_precision'], label='val_precision')
plt.plot(history.history['val_recall'], label='val_recall')
plt.plot(history.history['val_f1_score'], label='val_f1_score')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Validation Precision, Recall, F1 Score')

plt.tight_layout()
plt.show()
plt.savefig('./plots/main_resnet50_all_unfrozen_L2_batch_64.png')
