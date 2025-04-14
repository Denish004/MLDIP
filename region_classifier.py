import torch
import numpy as np
from PIL import Image, ImageDraw
import uuid
import os
import io
import base64

class RegionClassifier:
    def __init__(self, model, transform, classes, device, threshold=0.7):
        self.model = model
        self.transform = transform
        self.classes = classes
        self.device = device
        self.threshold = threshold  # Confidence threshold for predictions
        
    def divide_image(self, img, grid_size=(2, 2), overlap=0.2):
        """
        Divide image into a grid of subregions with overlap
        
        Parameters:
            img (PIL.Image): Input image
            grid_size (tuple): Number of regions in (rows, cols)
            overlap (float): Overlap between regions (0.0-0.5)
            
        Returns:
            list: List of (subimage, coords) tuples where coords is (x1, y1, x2, y2)
        """
        width, height = img.size
        rows, cols = grid_size
        
        # Calculate region dimensions with overlap
        region_width = width / cols
        region_height = height / rows
        
        # Calculate overlap in pixels
        overlap_x = int(region_width * overlap)
        overlap_y = int(region_height * overlap)
        
        regions = []
        
        # Generate subregions with overlap
        for i in range(rows):
            for j in range(cols):
                # Calculate region boundaries with overlap
                x1 = max(0, int(j * region_width - overlap_x))
                y1 = max(0, int(i * region_height - overlap_y))
                x2 = min(width, int((j + 1) * region_width + overlap_x))
                y2 = min(height, int((i + 1) * region_height + overlap_y))
                
                # Extract region
                region = img.crop((x1, y1, x2, y2))
                regions.append((region, (x1, y1, x2, y2)))
        
        return regions
    
    def classify_regions(self, img, grid_sizes=[(2, 2), (3, 3)]):
        """
        Classify image regions using multiple grid configurations
        
        Parameters:
            img (PIL.Image): Input image
            grid_sizes (list): List of grid configurations to try
            
        Returns:
            dict: Dictionary mapping class names to regions they appear in
        """
        results = {}
        region_results = []
        
        # For each grid configuration
        for grid_size in grid_sizes:
            # Divide the image into regions
            regions = self.divide_image(img, grid_size)
            
            # Classify each region
            for i, (region_img, coords) in enumerate(regions):
                # Prepare image for model
                img_tensor = self.transform(region_img).unsqueeze(0).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    conf, pred_idx = torch.max(probs, dim=0)
                
                # Only include prediction if confidence is above threshold
                if conf.item() > self.threshold:
                    label = self.classes[pred_idx.item()]
                    
                    # Store result
                    if label not in results:
                        results[label] = []
                    
                    results[label].append(coords)
                    
                    # Add to region results
                    region_results.append({
                        'coords': coords,
                        'label': label,
                        'confidence': conf.item()
                    })
        
        return results, region_results
    
    def visualize_regions(self, img, region_results):
        """
        Create a visualization of classified regions
        
        Parameters:
            img (PIL.Image): Original image
            region_results (list): List of region classification results
            
        Returns:
            PIL.Image: Image with highlighted regions
        """
        # Create a copy of the image to draw on
        img_with_regions = img.copy()
        draw = ImageDraw.Draw(img_with_regions)
        
        # Color palette for different classes (add more colors if needed)
        colors = [
            "#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3",
            "#33FFF3", "#FF8533", "#8533FF", "#33FF85", "#FF3385"
        ]
        
        # Create a map of class to color
        class_colors = {}
        
        # Draw each region
        for result in region_results:
            label = result['label']
            coords = result['coords']
            confidence = result['confidence']
            
            # Assign a color to each class
            if label not in class_colors:
                class_colors[label] = colors[len(class_colors) % len(colors)]
            
            color = class_colors[label]
            
            # Draw rectangle
            draw.rectangle(coords, outline=color, width=3)
            
            # Draw label
            text_pos = (coords[0] + 5, coords[1] + 5)
            draw.text(text_pos, f"{label} ({confidence:.2f})", fill=color)
        
        return img_with_regions
    
    def generate_images_for_display(self, img):
        """
        Generate original, enhanced and region-classified images for display
        
        Parameters:
            img (PIL.Image): Input image
            
        Returns:
            dict: Dictionary with base64 encoded images and detection results
        """
        # Step 1: Classify regions
        regions_dict, region_results = self.classify_regions(img)
        
        # Step 2: Create visualization
        annotated_img = self.visualize_regions(img, region_results)
        
        # Step 3: Convert to base64 for frontend display
        buffer = io.BytesIO()
        annotated_img.save(buffer, format='PNG')
        annotated_img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Step 4: Prepare results for frontend
        unique_labels = list(regions_dict.keys())
        
        return {
            'annotated_image': f'data:image/png;base64,{annotated_img_str}',
            'predicted_labels': unique_labels,
            'region_results': region_results
        }