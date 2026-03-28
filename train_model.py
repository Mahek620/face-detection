import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths to your split dataset
base_dir = 'split_dataset'

# Image size, batch size, epochs
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8

# Data generators for train/val/test
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    directory=f'{base_dir}/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_test_datagen.flow_from_directory(
    directory=f'{base_dir}/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_gen = val_test_datagen.flow_from_directory(
    directory=f'{base_dir}/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Debug: See class labels and indices to verify correct folder mapping
print("Class Indices:", train_gen.class_indices)

# Load MobileNetV2 base model (imagenet weights, no top layer)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(2, activation='softmax')(x)  # 2 classes

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers (transfer learning)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test accuracy: {test_acc:.2f}')

# Save the trained model
model.save('mask_detector_model.h5')
