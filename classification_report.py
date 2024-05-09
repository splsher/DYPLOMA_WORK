import numpy as np
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

train_dir = r'C:\Users\Tania\PycharmProjects\DYPLOMA_WORK\DATA\train'
test_dir = r'C:\Users\Tania\PycharmProjects\DYPLOMA_WORK\DATA\validation'

img_size = (32, 32)
batch_size = 32

loaded_model = load_model('trained_model.h5')
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale'
)

predictions = loaded_model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

print("Classification Report:")
print(classification_report(test_generator.labels, y_pred, target_names=test_generator.class_indices.keys()))