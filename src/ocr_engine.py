"""
Core OCR engine using multiple OCR backends
"""


#for window
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# for streamlit 

import pytesseract, shutil

# On Linux server:
binary = shutil.which("tesseract")
if binary:
    pytesseract.pytesseract.tesseract_cmd = binary
else:
    # Optionally fallback, or raise error
    raise RuntimeError("Tesseract binary not found. Make sure tesseract-ocr is installed.")

from PIL import Image
import numpy as np


try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class OCREngine:
    """OCR Engine supporting multiple backends"""
    
    def __init__(self, backend='tesseract'):
        """
        Initialize OCR engine
        
        Args:
            backend: 'tesseract', 'easyocr', or 'both'
        """
        self.backend = backend
        self.easyocr_reader = None
        
        if backend in ['easyocr', 'both'] and EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                print(f"Warning: Could not initialize EasyOCR: {e}")
                if backend == 'easyocr':
                    self.backend = 'tesseract'
    
    

    def perform_ocr_tesseract(self, image, config='--psm 3'):
        """
        Perform OCR using Tesseract
        
        Args:
            image: PIL Image or numpy array
            config: Tesseract configuration string
        
        Returns:
            Extracted text as string
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # PSM 3 works best for your labels based on test results
        # configs = [
        #     '--psm 6',
        #     '--psm 4',
        #     '--psm 11',
        #     '--psm 3',
        # ]
        configs = ['--psm 3']

      

        results = []
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(image, config=cfg).strip()
                results.append(text)
            except Exception as e:
                print(f"Tesseract error with config {cfg}: {e}")
        
        # Return longest result (usually best)
       
        return max(results, key=len) if results else ""


    
    def perform_ocr_easyocr(self, image):
        """
        Perform OCR using EasyOCR
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Extracted text as string
        """
        if not self.easyocr_reader:
            return ""
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        try:
            results = self.easyocr_reader.readtext(image, detail=0)
            return '\n'.join(results)
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def perform_ocr(self, image):
        """
        Perform OCR using selected backend(s)
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            dict with OCR results and confidence
        """
        results = {
            'tesseract_text': '',
            'easyocr_text': '',
            'combined_text': '',
            'confidence': 0.0
        }
        
        if self.backend in ['tesseract', 'both']:
            results['tesseract_text'] = self.perform_ocr_tesseract(image)
        
        if self.backend in ['easyocr', 'both'] and self.easyocr_reader:
            results['easyocr_text'] = self.perform_ocr_easyocr(image)
        
        # Combine results
        if self.backend == 'both':
            # Merge both results, preferring longer text
            texts = [results['tesseract_text'], results['easyocr_text']]
            results['combined_text'] = max(texts, key=len)
        elif self.backend == 'easyocr':
            results['combined_text'] = results['easyocr_text']
        else:
            results['combined_text'] = results['tesseract_text']
        
        # Calculate confidence based on text length and quality
        text = results['combined_text']
        if len(text) > 50:
            results['confidence'] = 0.85
        elif len(text) > 20:
            results['confidence'] = 0.70
        else:
            results['confidence'] = 0.50
        
        return results
    
    def get_detailed_results(self, image):
        """
        Get detailed OCR results with bounding boxes
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            dict with detailed results
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        try:
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            detailed = {
                'text': [],
                'confidence': [],
                'boxes': []
            }
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 0:
                    text = data['text'][i].strip()
                    if text:
                        detailed['text'].append(text)
                        detailed['confidence'].append(float(data['conf'][i]))
                        detailed['boxes'].append({
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i]
                        })
            
            return detailed
        except Exception as e:
            print(f"Error getting detailed results: {e}")
            return {'text': [], 'confidence': [], 'boxes': []}