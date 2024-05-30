import os
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import logging

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy model (replace with your model)
def load_model():
    model = torch.nn.Linear(256*256*3, 2)  # Dummy model structure
    return model

model = load_model().to(device)

# Load model if available
model_path = 'model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info("Model loaded successfully.")
else:
    logging.warning("No model found at specified path.")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define image class labels
class_labels = {
    0: 'bad_apple',
    1: 'good_apple'
}

@app.route('/')
def index():
    return "Flask server is running!"

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded image file
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        logging.info(f"File saved to {filepath}")

        # Load and process the image
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor.view(image_tensor.size(0), -1))
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        # Return prediction result and image URL
        prediction_label = class_labels[prediction]
        image_url = url_for('static', filename=f'{prediction_label}.jpg')
        return jsonify({'prediction': prediction_label, 'image_url': image_url}), 200

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
