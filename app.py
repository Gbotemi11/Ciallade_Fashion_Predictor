from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load your trained model
model = load_model("real_fashion_model.h5")

# Insert the correct class_names list from train_gen.class_indices.keys()
class_names = ['Accessory Gift Set', 'Baby Dolls', 'Backpacks', 'Bangle',
               'Basketballs', 'Bath Robe', 'Beauty Accessory', 'Belts', 
               'Blazers', 'Body Lotion', 'Booties', 'Boxers', 'Bra',
               'Bracelet', 'Briefs', 'Camisoles', 'Capris', 'Caps',
               'Casual Shoes', 'Churidar', 'Clothing Set', 'Clutches',
               'Compact', 'Concealer', 'Cufflinks', 'Deodorant', 'Dresses',
               'Duffel Bag', 'Dupatta', 'Earrings', 'Eye Cream', 'Eyeshadow',
               'Face Moisturisers', 'Face Scrub and Exfoliator',
               'Face Serum and Gel', 'Face Wash and Cleanser',
               'Flats', 'Flip Flops', 'Footballs', 'Formal Shoes',
               'Foundation and Primer', 'Fragrance Gift Set',
               'Free Gifts', 'Gloves', 'Hair Colour', 'Handbags',
               'Hat', 'Headband', 'Heels', 'Highlighter and Blush', 
               'Innerwear Vests', 'Jackets', 'Jeans', 'Jeggings', 
               'Jewellery Set', 'Jumpsuit', 'Kajal and Eyeliner', 
               'Key chain', 'Kurta Sets', 'Kurtas', 'Kurtis',
               'Laptop Bag', 'Leggings', 'Lehenga Choli', 
               'Lip Care', 'Lip Gloss', 'Lip Liner', 
               'Lip Plumper', 'Lipstick', 'Lounge Pants',
               'Lounge Shorts',
               'Lounge Tshirts', 'Makeup Remover', 'Mascara', 'Mask and Peel',
               'Messenger Bag', 'Mobile Pouch', 'Mufflers', 'Nail Essentials',
               'Nail Polish', 'Necklace and Chains', 'Nehru Jackets', 
               'Night suits', 'Nightdress', 'Patiala', 'Pendant', 
               'Perfume and Body Mist', 'Rain Jacket', 'Rain Trousers',
               'Ring', 'Robe', 'Rompers', 'Rucksacks', 'Salwar', 
               'Salwar and Dupatta', 'Sandals', 'Sarees', 'Scarves', 'Shapewear', 'Shirts', 'Shoe Accessories',
               'Shorts', 'Shrug', 'Skirts', 'Socks', 'Sports Sandals', 'Sports Shoes', 'Stockings', 'Stoles',
               'Sunglasses', 'Sunscreen', 'Suspenders', 'Sweaters', 'Sweatshirts', 'Swimwear', 'Tablet Sleeve', 
               'Ties', 'Ties and Cufflinks', 'Tights', 'Toner', 'Tops', 'Track Pants', 
               'Tracksuits', 'Travel Accessory', 'Trolley Bag', 'Trousers', 'Trunk',
               'Tshirts', 'Tunics', 'Umbrellas', 'Waist Pouch', 'Waistcoat', 
               'Wallets', 'Watches', 'Water Bottle', 'Wristbands']


@app.route('/')
def home():
    return "Fashion Model API is running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((128, 128))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        predicted_index = np.argmax(predictions)
        predicted_label = class_names[predicted_index]

        return jsonify({'prediction': predicted_label})
    
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
