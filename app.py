from base64 import b64encode

import numpy as np
from flask import Flask, request, render_template
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io

classes = ['healthy', 'multiple_diseases', 'rust', 'scab']

model = load_model('./static/model.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    f = request.files['file']
    content = f.read()
    res_image = b64encode(content).decode("utf-8")

    img = Image.open(io.BytesIO(content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    target_size = (150, 150)
    img = img.resize(target_size, Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes_prob = model.predict(x)
    print(classes_prob)

    i = np.argmax(classes_prob)
    result = classes[i]
    print(result)
    return render_template('results.html', result='Detected disease: {}'.format(result),
                           res_image=res_image)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)