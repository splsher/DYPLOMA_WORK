import io
import os
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, flash, redirect
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'secret key'
model = load_model('trained_model.h5')


@app.route('/')
def upload_file():
    return render_template('page_1.html')


def save_image(img, image_name):
    if not os.path.exists('classified_images'):
        os.makedirs('classified_images')
    image_path = os.path.join('classified_images', image_name)
    img.save(image_path)
    return image_path


class_labels = ["А", "Б", "В", "Г", "Д", "Е", "Є", "Ж", "З", "И", "І", "К", "Л", "М", "Н", "О", "П",
                "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Ь", "Ю",
                "Я"]


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash("No file selected. Please select a file!")
        return redirect('/')

    file = request.files['image']

    if file.filename == '':
        flash("No file selected. Please select a file!")
        return redirect('/')

    if file and allowed_file(file.filename):
        img = Image.open(io.BytesIO(file.read()))
        img_array = prepare_image(img.copy(), target_size=(32, 32))
        image_filename = secure_filename(file.filename)
        image_path = save_image(img, image_filename)
        pred_prob = model.predict(img_array)
        pred_class = np.argmax(pred_prob, axis=-1)
        predicted_label = class_labels[pred_class[0]]

        return render_template('predict.html', prediction=predicted_label,
                               user_image=image_filename)
    else:
        flash("Invalid file format. Please upload a PNG, JPG, or JPEG file.")
        return redirect('/')


def prepare_image(image, target_size=(32, 32)):
    image = image.convert('L')
    image = ImageOps.fit(image, target_size, method=Image.LANCZOS)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(port=5000)