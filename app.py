from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture to match the saved weights
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # No Sequential, matches saved model
model.load_state_dict(torch.load("resnet50_best_model.pth", map_location=device))
model.to(device)
model.eval()

# Class names
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = Image.open(request.files['image']).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = torch.sigmoid(model(img_tensor))  # Apply sigmoid manually
            preds = outputs.squeeze() > 0.5

        predicted_labels = [classes[i] for i, val in enumerate(preds) if val.item()]
        return jsonify({'predicted_labels': predicted_labels})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
