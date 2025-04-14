import cv2
import numpy as np
from PIL import Image
import io

class ImageEnhancer:
    def __init__(self, denoise_strength=5, clahe_clip=2.5, sharpen_strength=1.6):
        """
        Initialize the image enhancer with configurable parameters.
        
        Parameters:
            denoise_strength (int): Strength of denoising (h parameter for fastNlMeansDenoisingColored)
            clahe_clip (float): Clip limit for CLAHE contrast enhancement
            sharpen_strength (float): Strength of sharpening effect
        """
        self.denoise_strength = denoise_strength
        self.clahe_clip = clahe_clip
        self.sharpen_strength = sharpen_strength
        
    def enhance_image(self, pil_img):
        """
        Enhance a satellite image using multiple image processing techniques.
        
        Parameters:
            pil_img (PIL.Image): Input PIL image
            
        Returns:
            PIL.Image: Enhanced image
        """
        # Convert PIL image to OpenCV format (RGB to BGR)
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Step 1: Mild Denoising (avoid oversmoothing)
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, 
            h=self.denoise_strength, 
            hColor=self.denoise_strength, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
        
        # Step 2: Convert to LAB and apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        contrast_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Step 3: Sharpening using Unsharp Mask (reduce blur)
        blurred = cv2.GaussianBlur(contrast_enhanced, (5, 5), 1.0)
        sharpened = cv2.addWeighted(
            contrast_enhanced, 
            self.sharpen_strength, 
            blurred, 
            -(self.sharpen_strength - 1), 
            0
        )
        
        # Convert back to PIL Image (BGR to RGB)
        result = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)

    def process_for_display(self, pil_img):
        """
        Process an image and return both original and enhanced versions
        for display in the web interface.
        
        Parameters:
            pil_img (PIL.Image): Input PIL image
            
        Returns:
            tuple: (original_pil_image, enhanced_pil_image)
        """
        enhanced_img = self.enhance_image(pil_img)
        return pil_img, enhanced_img
    
    def process_bytes(self, image_bytes):
        """
        Process image from bytes and return enhanced image as bytes.
        Useful for API endpoints.
        
        Parameters:
            image_bytes (bytes): Input image as bytes
            
        Returns:
            bytes: Enhanced image as bytes
        """
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        enhanced = self.enhance_image(image)
        
        # Convert back to bytes
        output = io.BytesIO()
        enhanced.save(output, format='PNG')
        return output.getvalue()