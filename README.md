# 👗 Ciallade Smart Predictor

An AI-powered fashion classification app that recognizes real-world clothing items from photos. Built for the Ciallade brand to deliver mobile-first predictions using Flutter and a cloud-hosted deep learning model.

---

##  Objective

To build a smart fashion recognition system that can:
- Take photos or upload images from a phone
- Use AI to identify the fashion item (e.g. T-shirt, sandals, etc.)
- Return the result instantly via a Flutter mobile app

---

##  Project Phases

### 1. Initial Setup
- Created a Flutter app with UI for image upload
- Deployed a simple Fashion MNIST model to Render with a Flask API

### 2. Real-World Dataset Model
- Switched to real photo data from Kaggle (Fashion Product Images)
- Used MobileNetV2 with transfer learning on Google Colab
- Applied image resizing and data augmentation
- Exported trained model as `real_fashion_model.h5`

### 3. Deployment
- Flask API accepts image uploads via `/predict`
- Flask returns predicted class name
- Hosted model and API on Render
- Prediction tested using Postman and real photos

### 4. Flutter App Integration
- Integrated camera and gallery picker
- Added loading and result UI
- Connected API using multipart POST requests
- Added app name, internet permission, and camera support
- Built a release APK

---

##  Final Results

- Android app works on real phones
- Sends image to cloud model
- Receives and displays prediction
- `.apk` ready to install or submit to Google Play

---

##  Next Steps

| Feature                         | Status          |
|---------------------------------|------------------|
| App icon + branding             | Optional         |
| Upload to Google Play Store     | Ready to start   |
| Splash screen customization     | Easy to add      |
| Add multi-item detection (YOLO) | In planning    |
| Offline mode (TFLite)           | Optional         |
| iOS support                     | Planned        |

---

## 🛠 Tech Stack

- Flutter (UI)
- Python + Flask (API)
- TensorFlow / Keras (Model training)
- MobileNetV2 (Transfer learning)
- Hosted on Render (Backend)
- Dataset: [Kaggle Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

---

## 👤 Author

**Gbotemi (for Ciallade)**
