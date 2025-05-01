# 👗 Fashion Classifier API – Built for Ciallade

A real-time fashion item classifier developed for **Ciallade**, a modern fashion designer brand.  
This project uses a trained Convolutional Neural Network (CNN) with TensorFlow and Flask to recognize fashion items from grayscale images (28x28).

## 🌍 Live Demo (once deployed)
> https://your-api-url.onrender.com/predict

## 🧠 Supported Classes
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## 🚀 How It Works
Upload a grayscale clothing image, and the API will return the most likely clothing category using a deep learning model.

## 📦 Project Structure
```
fashion_api/
├── app.py                 # Flask API
├── fashion_mnist_cnn_model.h5  # Trained model (add manually)
├── requirements.txt       # Python dependencies
├── render.yaml            # Render.com deployment config
└── README.md              # Project description
```

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fashion-api-ciallade.git
cd fashion-api-ciallade
```

### 2. Add the Model File
Place your trained model file `fashion_mnist_cnn_model.h5` into the root of the project folder.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
python app.py
```

### 5. Deploy to Render
- Push this repo to GitHub
- Go to [Render.com](https://render.com)
- Click “New Web Service” > Connect your repo
- Done!

## 📷 Example Usage (with `requests`)
```python
import requests

image_path = 'my_test_image.png'
url = 'https://your-api-url.onrender.com/predict'
files = {'file': open(image_path, 'rb')}
response = requests.post(url, files=files)
print(response.json())  # {'prediction': 'Sneaker'}
```

## ✨ About Ciallade
Ciallade is a modern fashion label exploring the intersection of design, technology, and personalization.  
This model supports Ciallade's vision by enabling AI-powered digital fashion intelligence.

---

Made with ❤️ for Ciallade