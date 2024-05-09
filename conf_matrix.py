import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.preprocessing.image import ImageDataGenerator
from keras.src.saving.saving_api import load_model
from sklearn.metrics import confusion_matrix

train_dir = r'C:\Users\Tania\PycharmProjects\DYPLOMA_WORK\DATA\train'
test_dir = r'C:\Users\Tania\PycharmProjects\DYPLOMA_WORK\DATA\validation'

img_size = (32, 32)
batch_size = 32
loaded_model = load_model('trained_model_1.h5')

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

predictions = loaded_model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
conf_matrix = confusion_matrix(test_generator.classes, y_pred)

conf_matrix = conf_matrix.astype('float') / conf_matrix.max()

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys(),
           linewidths=0.2)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Normalized Confusion Matrix')
plt.show()