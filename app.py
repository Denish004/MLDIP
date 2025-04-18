from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import io
import base64
from torchvision import models, transforms
from PIL import Image
from image_processing import ImageEnhancer
from region_classifier import RegionClassifier
from flask_cors import CORS
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
from skimage.segmentation import slic
from skimage.util import img_as_float
from captum.attr import GuidedGradCam, LayerGradCam
from captum.attr import visualization as viz

app = Flask(__name__)
# Allow all origins (for development only)
CORS(app, resources={r"/*": {"origins": "*"}})

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture to match the saved weights
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("./models/resnet50_final.pth", map_location=device))
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

def split_image_to_grid(image, grid_size=3):
    """Split image into grid_size x grid_size regions"""
    width, height = image.size
    cell_width = width // grid_size
    cell_height = height // grid_size
    
    grid_cells = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * cell_width
            upper = row * cell_height
            right = left + cell_width
            lower = upper + cell_height
            
            cell = image.crop((left, upper, right, lower))
            grid_cells.append(cell)
    
    return grid_cells, (cell_width, cell_height)

def split_image_to_grid_with_overlap(image, grid_size=3, overlap_percent=30):
    """Split image into grid_size x grid_size regions with overlap"""
    width, height = image.size
    
    # Calculate base cell dimensions
    cell_width = width // grid_size
    cell_height = height // grid_size
    
    # Calculate overlap in pixels
    overlap_x = int(cell_width * overlap_percent / 100)
    overlap_y = int(cell_height * overlap_percent / 100)
    
    # Adjust cell dimensions to include overlap
    cell_width_with_overlap = cell_width + 2 * overlap_x
    cell_height_with_overlap = cell_height + 2 * overlap_y
    
    grid_cells = []
    cell_positions = []  # To store the positions for visualization
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate center position
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2
            
            # Calculate crop boundaries with overlap
            left = max(0, center_x - cell_width_with_overlap // 2)
            upper = max(0, center_y - cell_height_with_overlap // 2)
            right = min(width, center_x + cell_width_with_overlap // 2)
            lower = min(height, center_y + cell_height_with_overlap // 2)
            
            # Store crop position for visualization
            cell_positions.append((left, upper, right, lower))
            
            # Crop the cell
            cell = image.crop((left, upper, right, lower))
            grid_cells.append(cell)
    
    return grid_cells, cell_positions

def classify_image_grid(grid_cells):
    """Classify each cell in the grid"""
    results = []
    
    for cell in grid_cells:
        # Apply transform to prepare for model
        img_tensor = transform(cell).unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get top 3 predictions for each cell for more comprehensive analysis
            top_prob, top_class = torch.topk(probs, 3)
            
            # Get primary prediction (highest confidence)
            _, predicted = torch.max(outputs, 1)
            primary_label = classes[predicted.item()]
            primary_confidence = probs[predicted.item()].item()
        
        # Add detailed result with top 3 predictions
        results.append({
            'label': primary_label,
            'confidence': float(primary_confidence),
            'top_predictions': [
                {
                    'label': classes[idx.item()],
                    'confidence': prob.item()
                } for prob, idx in zip(top_prob, top_class)
            ]
        })
    
    return results

def create_grid_visualization(original_image, grid_cells, cell_positions, classifications, grid_size=3):
    """Create a larger, improved visualization of the grid with labels and original image outlines"""
    # Create a larger figure for better visibility
    fig = Figure(figsize=(16, 16), dpi=120)
    
    # First, show the original image with regions outlined
    ax_original = fig.add_subplot(1, 2, 1)
    ax_original.imshow(np.array(original_image))
    ax_original.set_title("Original Image with Analyzed Regions", fontsize=14)
    
    # Draw rectangles for each region with improved styling
    for i, (left, upper, right, lower) in enumerate(cell_positions):
        rect = patches.Rectangle(
            (left, upper), right-left, lower-upper, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax_original.add_patch(rect)
        
        # Add region number for reference
        region_center_x = left + (right - left) // 2
        region_center_y = upper + (lower - upper) // 2
        
        # Create a better looking label background
        ax_original.text(
            region_center_x, region_center_y, 
            f"{i+1}", color='white', fontweight='bold', 
            ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='red', alpha=0.7, boxstyle='round')
        )
    
    ax_original.axis('off')
    
    # Create a grid for individual region analysis
    grid_spec = fig.add_gridspec(grid_size, grid_size, left=0.55, right=0.98, wspace=0.1, hspace=0.25)
    
    # Plot each cell with its classification
    for i, (cell, result) in enumerate(zip(grid_cells, classifications)):
        row = i // grid_size
        col = i % grid_size
        
        ax = fig.add_subplot(grid_spec[row, col])
        ax.imshow(np.array(cell))
        
        # Get confidence for color mapping
        confidence = result['confidence']
        
        # Dynamic color based on confidence (red to green)
        color = plt.cm.RdYlGn(confidence)
        
        # Format title with classification and confidence
        title = f"Region {i+1}: {result['label']}\n{confidence:.2f}"
        ax.set_title(title, color=color, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle("Detailed Region Analysis", fontsize=16, y=0.98)
    fig.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return f'data:image/png;base64,{img_str}'

def create_enhanced_visualization(grid_cells, classifications, grid_size=3):
    """Create enhanced visualization with confidence distribution"""
    fig = Figure(figsize=(18, 12), dpi=120)
    
    # Create grid for individual cell analysis
    for i, (cell, result) in enumerate(zip(grid_cells, classifications)):
        row = i // grid_size
        col = i % grid_size
        
        # Add subplot for this grid cell
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.imshow(np.array(cell))
        
        # Display top 3 predictions with confidence bars
        top_predictions = result['top_predictions']
        
        # Main title is the top prediction
        confidence = result['confidence']
        color = plt.cm.RdYlGn(confidence)
        title = f"{result['label']}\n{confidence:.2f}"
        ax.set_title(title, color=color, fontsize=12, fontweight='bold')
        
        # Add confidence bars for all top predictions
        for j, pred in enumerate(top_predictions):
            ax.add_patch(
                patches.Rectangle(
                    (0, cell.height - 10 - j*15), 
                    pred['confidence'] * cell.width, 
                    10,
                    facecolor=plt.cm.RdYlGn(pred['confidence']),
                    alpha=0.8
                )
            )
            
            # Add label for each bar
            ax.text(
                5, cell.height - 5 - j*15,
                f"{pred['label']} ({pred['confidence']:.2f})",
                color='black', fontsize=8,
                verticalalignment='bottom'
            )
        
        ax.axis('off')
    
    fig.suptitle("Enhanced Grid Analysis with Top Predictions", fontsize=16)
    fig.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return f'data:image/png;base64,{img_str}'

def multi_scale_analysis(image, scales=[1.0, 0.75, 0.5]):
    """Analyze image at multiple scales to catch both large and small features"""
    results = []
    scale_images = []
    
    for scale in scales:
        # Resize image according to scale
        width, height = image.size
        new_width, new_height = int(width * scale), int(height * scale)
        scaled_img = image.resize((new_width, new_height), Image.LANCZOS)
        scale_images.append(scaled_img)
        
        # Apply transform and classify
        img_tensor = transform(scaled_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top 3 predictions for this scale
        top_prob, top_class = torch.topk(probs, 3)
        
        scale_result = {
            'scale': scale,
            'predictions': [
                {
                    'label': classes[idx.item()],
                    'confidence': prob.item()
                } for prob, idx in zip(top_prob, top_class)
            ]
        }
        results.append(scale_result)
    
    return results, scale_images

def generate_saliency_map(image):
    """Generate saliency map highlighting areas the model focuses on"""
    # Prepare image for model
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Select layer for GradCAM - using the last layer of ResNet
    target_layer = model.layer4[-1]
    
    # Create GradCAM object
    grad_cam = LayerGradCam(model, target_layer)
    
    # We need to determine the target class - using the predicted class
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_class = torch.max(outputs, 1)
        target_class = predicted_class.item()
    
    # Get attributions
    attributions = grad_cam.attribute(img_tensor, target=target_class)
    
    # Create visualization
    fig = Figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    
    # Convert image to numpy for visualization
    np_img = np.array(image.resize((224, 224)))
    
    # Upsample attributions to match image size
    attr = attributions.cpu().detach().numpy()[0, 0]
    attr = cv2.resize(attr, (224, 224))
    
    # Normalize attributions
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-10)
    
    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * attr), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    heatmap = heatmap * 0.7 + np_img * 0.3
    
    ax.imshow(heatmap.astype(np.uint8))
    ax.set_title(f"Saliency Map: {classes[target_class]}", fontsize=14)
    ax.axis('off')
    
    # Convert to base64
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return f'data:image/png;base64,{img_str}'

def semantic_segmentation(image, n_segments=10):
    """Perform semantic segmentation on the image"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply SLIC segmentation
    segments = slic(img_as_float(img_array), n_segments=n_segments, compactness=10)
    
    # Create visualization
    fig = Figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create segmented image
    segmented_img = np.zeros_like(img_array)
    
    # Get average color for each segment
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        segmented_img[mask] = np.mean(img_array[mask], axis=0)
    
    # Show segmented image
    ax.imshow(segmented_img)
    ax.set_title("Semantic Segmentation", fontsize=14)
    ax.axis('off')
    
    # Convert to base64
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return f'data:image/png;base64,{img_str}'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get the uploaded image
            img = Image.open(request.files['image']).convert('RGB')
            
            # Process the image - get both original and enhanced
            original_img, enhanced_img = enhancer.process_for_display(img)
            
            # Convert original image to base64
            buffer_original = io.BytesIO()
            original_img.save(buffer_original, format='PNG')
            img_str_original = base64.b64encode(buffer_original.getvalue()).decode('utf-8')
            
            # Convert enhanced image to base64
            buffer_enhanced = io.BytesIO()
            enhanced_img.save(buffer_enhanced, format='PNG')
            img_str_enhanced = base64.b64encode(buffer_enhanced.getvalue()).decode('utf-8')
            
            # Split the enhanced image into overlapping grid regions
            grid_cells, cell_positions = split_image_to_grid_with_overlap(enhanced_img, grid_size=3, overlap_percent=30)
            
            # Classify each grid cell
            grid_classifications = classify_image_grid(grid_cells)
            
            # Create improved grid visualization
            grid_visualization = create_grid_visualization(enhanced_img, grid_cells, cell_positions, grid_classifications)
            
            # Create enhanced visualization with confidence distribution
            enhanced_grid_viz = create_enhanced_visualization(grid_cells, grid_classifications)
            
            # Generate saliency map
            saliency_map = generate_saliency_map(enhanced_img)
            
            # Perform multi-scale analysis
            multi_scale_results, _ = multi_scale_analysis(enhanced_img)
            
            # Generate semantic segmentation
            segmentation_result = semantic_segmentation(enhanced_img, n_segments=15)
            
            # Get overall region classification (original functionality)
            region_results = region_classifier.generate_images_for_display(enhanced_img)
            
            # Collect grid cell labels and confidence scores
            grid_labels = [result['label'] for result in grid_classifications]
            grid_confidences = [result['confidence'] for result in grid_classifications]
            
            # Find the dominant labels - consider frequency and confidence
            label_confidence = {}
            for label, conf in zip(grid_labels, grid_confidences):
                if label in label_confidence:
                    label_confidence[label] = max(label_confidence[label], conf)
                else:
                    label_confidence[label] = conf
                    
            # Sort by confidence
            dominant_labels = sorted(label_confidence.items(), key=lambda x: x[1], reverse=True)
            
            # Return comprehensive analysis results
            return jsonify({
                'original_image': f'data:image/png;base64,{img_str_original}',
                'enhanced_image': f'data:image/png;base64,{img_str_enhanced}',
                'annotated_image': region_results['annotated_image'],
                'grid_visualization': grid_visualization,
                'enhanced_grid_viz': enhanced_grid_viz,
                'saliency_map': saliency_map,
                'segmentation': segmentation_result,
                'multi_scale_results': multi_scale_results,
                'grid_classifications': grid_classifications,
                'dominant_labels': [{"label": label, "confidence": conf} for label, conf in dominant_labels],
                'predicted_labels': list(set(grid_labels)),
                'region_details': region_results['region_results']
            })
        
        except Exception as e:
            return jsonify({
                'error': str(e)
            }), 500
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
