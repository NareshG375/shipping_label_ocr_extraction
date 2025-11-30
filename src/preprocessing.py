"""
Image preprocessing module for OCR enhancement - IMPROVED VERSION
"""
import cv2
import numpy as np
from PIL import Image


def preprocess_image(image, method='adaptive'):
    """
    Preprocess image for better OCR results - LIGHT TOUCH
    
    Args:
        image: PIL Image or numpy array
        method: preprocessing method ('adaptive', 'otsu', 'basic', 'none')
    
    Returns:
        Preprocessed image as numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # For 'none', return grayscale only
    if method == 'none':
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        return gray
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply preprocessing based on method
    if method == 'adaptive':
        # Adaptive thresholding - best for varying lighting
        # Increased block size for better results
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 5  # Changed from 11, 2
        )
    elif method == 'otsu':
        # Otsu's thresholding
        _, processed = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:  # basic
        # Simple thresholding
        _, processed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    return processed


def enhance_image(image):
    """
    Apply LIGHT enhancement techniques (less aggressive)
    
    Args:
        image: Input image
    
    Returns:
        Enhanced image
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # LIGHT denoise only (reduced strength)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    
    # LIGHT contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Reduced from 2.0
    enhanced = clahe.apply(denoised)
    
    return enhanced


def enhance_image_light(image):
    """
    Apply MINIMAL enhancement - just contrast
    
    Args:
        image: Input image
    
    Returns:
        Enhanced image
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Only contrast enhancement - no denoising
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def remove_borders(image, border_size=10):
    """
    Remove borders from image
    
    Args:
        image: Input image
        border_size: Border size to remove in pixels
    
    Returns:
        Image with borders removed
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    h, w = img.shape[:2]
    return img[border_size:h-border_size, border_size:w-border_size]


def deskew_image(image):
    """
    Deskew image by detecting text angle
    
    Args:
        image: Input image
    
    Returns:
        Deskewed image
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is not None:
        # Calculate average angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            
            # Only rotate if angle is significant
            if abs(median_angle) > 0.5:
                # Rotate image
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
                return rotated
    
    return img


def resize_image(image, max_width=2000, max_height=2000):
    """
    Resize image for OCR without losing aspect ratio.
    Uses LANCZOS which is best for text.
    """
    # Ensure PIL image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    width, height = image.size

    # Compute scale factor (never upscale)
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)

    # Resize only if necessary
    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image

def resize_image_old(image, max_width=2000, max_height=2000):
    """
    Resize image while maintaining aspect ratio
    INCREASED max size to preserve quality
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        Resized image
    """
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(image)
    
    width, height = img.size
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img


def auto_preprocess(image):
    """
    Automatically determine best preprocessing
    Returns original if quality is already good
    
    Args:
        image: Input image
    
    Returns:
        Best preprocessed version
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Check image quality metrics
    # 1. Check contrast
    contrast = gray.std()
    
    # 2. Check brightness
    brightness = gray.mean()
    
    # If good quality, return minimal processing
    if contrast > 40 and 50 < brightness < 200:
        # Good quality - return grayscale only
        return gray
    
    # Low contrast - apply enhancement
    if contrast < 40:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    # Poor brightness - apply thresholding
    if brightness < 50 or brightness > 200:
        _, processed = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return processed
    
    return gray