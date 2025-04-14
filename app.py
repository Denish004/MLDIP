from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from image_processing import ImageEnhancer
from region_classifier import RegionClassifier
from flask_cors import CORS

app = Flask(__name__)
# Allow all origins (for development only)
CORS(app, resources={r"/*": {"origins": "*"}})

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture to match the saved weights
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("./models/resnet50_best.pth", map_location=device))
model.to(device)
model.eval()

# Define transform for the model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class names
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Create enhancer with default settings
enhancer = ImageEnhancer(denoise_strength=5, clahe_clip=2.5, sharpen_strength=1.6)

# Create region classifier
region_classifier = RegionClassifier(model, transform, classes, device, threshold=0.7)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        img = Image.open(request.files['image']).convert('RGB')
        
        # Process the image - get both original and enhanced
        original_img, enhanced_img = enhancer.process_for_display(img)
        
        # Convert original image to base64
        import io
        import base64
        buffer_original = io.BytesIO()
        original_img.save(buffer_original, format='PNG')
        img_str_original = base64.b64encode(buffer_original.getvalue()).decode('utf-8')
        
        # Convert enhanced image to base64
        buffer_enhanced = io.BytesIO()
        enhanced_img.save(buffer_enhanced, format='PNG')
        img_str_enhanced = base64.b64encode(buffer_enhanced.getvalue()).decode('utf-8')
        
        # Classify regions in the enhanced image
        region_results = region_classifier.generate_images_for_display(enhanced_img)
        
        # Return results
        return jsonify({
            'original_image': f'data:image/png;base64,{img_str_original}',
            'enhanced_image': f'data:image/png;base64,{img_str_enhanced}',
            'annotated_image': region_results['annotated_image'],
            'predicted_labels': region_results['predicted_labels'],
            'region_details': region_results['region_results']
        })
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
