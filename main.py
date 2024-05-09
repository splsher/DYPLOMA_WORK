import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

train_dir = r'C:\Users\Tania\PycharmProjects\DYPLOMA_WORK\DATA\train'
test_dir = r'C:\Users\Tania\PycharmProjects\DYPLOMA_WORK\DATA\validation'

img_size = (32, 32)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

validation_datagen = ImageDataGenerator(rescale=1. / 255)


history = model.fit(train_generator, epochs=20, validation_data=test_generator)

model.save('trained_model_1.h5')
print("Model saved successfully.")

train_accuracy = history.history['accuracy'][-1]
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

validation_accuracy = history.history['val_accuracy'][-1]
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

model.save('trained_model_1.h5')
print("Model saved successfully.")

accuracy = model.evaluate(test_generator)[1]
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# ----------------------------------------------
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

loaded_model = load_model('trained_model_1.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

predictions = loaded_model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

print("Classification Report:")
print(classification_report(test_generator.labels, y_pred, target_names=test_generator.class_indices.keys()))