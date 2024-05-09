import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model('trained_model.h5')

sample_image_path = r'C:\Users\Tania\PycharmProjects\DYPLOMA_WORK\test_images\bw_image_Г.jpg'
# sample_image_path = r'C:\Users\Tania\PycharmProjects\DYPLOMA_WORK\test_images\bw_image_О.jpg'

img_size = (32, 32)


img = Image.open(sample_image_path).convert('L')
img = img.resize(img_size)
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=2)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
class_labels = ["А", "Б", "В", "Г", "Д", "Е", "Є", "Ж", "З", "И", "І", "К", "Л", "М", "Н", "О", "П",
                "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Ь", "Ю",
                "Я"]

predicted_label = class_labels[predicted_class]
plt.imshow(img, cmap='gray')
plt.title(f"Predicted label: {predicted_label}")
plt.show()