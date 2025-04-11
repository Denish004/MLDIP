# from flask import Flask, request, jsonify, render_template
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import os

# app = Flask(__name__)

# # Load model
# model = models.resnet18(pretrained=False)
# model.fc = torch.nn.Sequential(
#     torch.nn.Linear(model.fc.in_features, 10),
#     torch.nn.Sigmoid()  # For multi-label output
# )
# model.load_state_dict(torch.load("resnet_eurosat_multilabel1.pth", map_location=torch.device('cpu')))
# model.eval()

# # EuroSAT class names
# classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
#            'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# # Transform
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         img = Image.open(request.files['image']).convert('RGB')
#         img_t = transform(img).unsqueeze(0)

#         with torch.no_grad():
#             outputs = model(img_t)
#             preds = outputs.squeeze() > 0.5  # Threshold for multi-label

#         predicted_classes = [classes[i] for i, val in enumerate(preds) if val.item()]
#         return jsonify({'predicted_labels': predicted_classes})
    
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
# # import cv2
# # import numpy as np
# # from flask import Flask, request, jsonify, render_template
# # import torch
# # from torchvision import models, transforms
# # from PIL import Image
# # import os

# # app = Flask(__name__)

# # # Load model
# # model = models.resnet18(pretrained=False)
# # model.fc = torch.nn.Sequential(
# #     torch.nn.Linear(model.fc.in_features, 10),
# #     torch.nn.Sigmoid()  # For multi-label output
# # )
# # model.load_state_dict(torch.load("resnet_eurosat_multilabel1.pth", map_location=torch.device('cpu')))
# # model.eval()

# # # EuroSAT class names
# # classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
# #            'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# # # Transform for classification
# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                          std=[0.229, 0.224, 0.225])
# # ])

# # def apply_filters(pil_img):
# #     # Convert PIL image to OpenCV image format (numpy array)
# #     cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
# #     # 1. Sharpening filter
# #     sharpening_kernel = np.array([[0, -1, 0],
# #                                   [-1, 5, -1],
# #                                   [0, -1, 0]])
# #     sharpened = cv2.filter2D(cv_img, -1, sharpening_kernel)
    
# #     # 2. Denoising using bilateral filter
# #     denoised = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)
    
# #     # 3. Enhance contrast using histogram equalization on the Y channel in YCrCb space
# #     ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
# #     y_channel, cr, cb = cv2.split(ycrcb)
# #     y_eq = cv2.equalizeHist(y_channel)
# #     ycrcb_eq = cv2.merge((y_eq, cr, cb))
# #     contrast_enhanced = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    
# #     # Convert back to PIL Image (RGB)
# #     processed_img = Image.fromarray(cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2RGB))
# #     return processed_img

# # @app.route('/', methods=['GET', 'POST'])
# # def index():
# #     if request.method == 'POST':
# #         # Load image from the request and convert to RGB
# #         img = Image.open(request.files['image']).convert('RGB')
        
# #         # Apply digital image processing filters to enhance the image
# #         processed_img = apply_filters(img)
        
# #         # Transform the processed image for the model
# #         img_t = transform(processed_img).unsqueeze(0)

# #         with torch.no_grad():
# #             outputs = model(img_t)
# #             preds = outputs.squeeze() > 0.5  # Threshold for multi-label

# #         predicted_classes = [classes[i] for i, val in enumerate(preds) if val.item()]
# #         return jsonify({'predicted_labels': predicted_classes})
    
# #     return render_template('index.html')

# # if __name__ == '__main__':
# #     app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the Keras model
model = load_model('resnet50_best_model.pth')

# EuroSAT class names
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Image transformation for Keras model input
transform = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

def prepare_image(img):
    img = img.resize((224, 224))  # Resize image to the input size expected by ResNet50
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = transform.standardize(img_array)  # Rescale the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = Image.open(request.files['image']).convert('RGB')
        img_array = prepare_image(img)

        # Make prediction
        preds = model.predict(img_array)

        # Apply thresholding for multi-label classification (output of ResNet50)
        threshold = 0.5
        predicted_classes = [classes[i] for i, val in enumerate(preds[0]) if val > threshold]

        return jsonify({'predicted_labels': predicted_classes})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
