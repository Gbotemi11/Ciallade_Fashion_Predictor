from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model("fashion_mnist_cnn_model.h5")

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize if needed
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Normalize

    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
