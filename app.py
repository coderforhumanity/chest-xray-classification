from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load your CNN model
model = load_model('model/best_model_vgg19.h5')

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Load and preprocess the image, then make a prediction
            img = load_and_preprocess_image(filepath)
            prediction = model.predict(img)
            if np.argmax(prediction, axis = 1)[0] == 0:
                result = "Classified as: {}".format("Covid 19")
            elif np.argmax(prediction, axis = 1)[0] == 1:
                result = "Classified as: {}".format("Normal")
            elif np.argmax(prediction, axis = 1)[0] == 2:
                result = "Classified as: {}".format("Pneumonia")
            elif np.argmax(prediction, axis = 1)[0] == 3:
                result = "Classified as: {}".format("Tuberculosis")

            print(result)
            return render_template('index.html', result=result, filepath=file.filename)

    return render_template('index.html', result=None)
if __name__ == '__main__':
    app.run(debug=True)