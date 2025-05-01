from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

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
    try:
        img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Normalize
    except Exception as e:
        return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    # Use PORT from Render environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
