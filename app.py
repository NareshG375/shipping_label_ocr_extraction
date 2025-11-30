"""
Streamlit application for Shipping Label OCR Extraction
"""
import streamlit as st
from PIL import Image
import io
import os
import sys
import cv2
import numpy as np
import pytesseract
from datetime import datetime

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import modules
import ocr_engine
import preprocessing
import text_extraction

# Helper functions (inline to avoid import issues)
def format_confidence(confidence: float) -> str:
    """Format confidence value as percentage"""
    return f"{confidence * 100:.1f}%"

def create_timestamp() -> str:
    """Create timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def rotate_image(image, angle=90):
    """
    Rotate image by specified angle
    
    Args:
        image: PIL Image or numpy array
        angle: rotation angle in degrees (positive = counter-clockwise)
    
    Returns:
        Rotated image as numpy array
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Get image dimensions
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    return rotated


def auto_rotate_for_ocr(image):
    """
    Automatically detect and rotate image to correct orientation for OCR
    Uses Tesseract's orientation detection
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        Rotated image as numpy array
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Use Tesseract to detect orientation
    try:
        osd = pytesseract.image_to_osd(img)
        # Extract rotation angle from output
        rotation = int([line for line in osd.split('\n') if 'Rotate' in line][0].split()[-1])
        
        if rotation != 0:
            # Rotate back to correct orientation
            img = rotate_image(img, angle=-rotation)
            print(f"Auto-rotated image by {-rotation} degrees")
    except Exception as e:
        print(f"Could not auto-rotate: {e}")
    
    return img

# Page configuration
st.set_page_config(
    page_title="Shipping Label OCR",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .extracted-text {
        font-family: monospace;
        font-size: 1.2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
        color: #1f77b4;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'ocr_engine' not in st.session_state:
        st.session_state.ocr_engine = None
    if 'text_extractor' not in st.session_state:
        st.session_state.text_extractor = text_extraction.TextExtractor()
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []


def process_image(image, preprocessing_method, apply_enhancement, apply_rotation, result_column):
    """Process uploaded image and extract text"""
    
    with result_column:
        with st.spinner("Processing image..."):
            try:
                # Step 1: Resize if needed
                status = st.empty()
                status.info("Step 1/5: Resizing image...")
                processed_image = preprocessing.resize_image(image)
                
                # Step 2: Enhance
                if apply_enhancement:
                    status.info("Step 2/5: Enhancing image...")
                    processed_image = preprocessing.enhance_image(processed_image)
                else:
                    status.info("Step 2/5: Skipping enhancement...")
                
                # Step 3: Auto-rotate
                if apply_rotation:
                    status.info("Step 3/5: Auto-rotating image...")
                    processed_image = auto_rotate_for_ocr(processed_image)
                else:
                    status.info("Step 3/5: Skipping rotation...")
                
                # Step 4: Preprocess
                status.info("Step 4/5: Preprocessing...")
                processed_image = preprocessing.preprocess_image(processed_image, method=preprocessing_method)
                
                # Show preprocessed image
                st.image(processed_image, caption="Preprocessed Image", width='stretch')
                
                # Step 5: OCR
                status.info("Step 5/5: Performing OCR...")
                ocr_results = st.session_state.ocr_engine.perform_ocr(image)
                
                # Extract target text
                full_text = ocr_results['combined_text']
                extracted_text = st.session_state.text_extractor.extract_target_text(full_text)
                
                # Store result
                result = {
                    'success': extracted_text is not None,
                    'extracted_text': extracted_text,
                    'full_text': full_text,
                    'confidence': ocr_results['confidence'],
                    'timestamp': create_timestamp()
                }
                
                st.session_state.current_result = result
                st.session_state.results_history.append(result)
                
                status.empty()
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.current_result = None


def display_results(result):
    """Display extraction results"""
    
    if result['success']:
        st.markdown('<div class="success-box">✅ Pattern Successfully Extracted!</div>', 
                   unsafe_allow_html=True)
        
        # Display extracted text
        st.markdown("### Extracted Text:")
        st.markdown(f'<div class="extracted-text">{result["extracted_text"]}</div>', 
                   unsafe_allow_html=True)
        
        # Confidence
        st.metric("Confidence", format_confidence(result['confidence']))
        
        # Validation
        validation = st.session_state.text_extractor.validate_pattern(result['extracted_text'])
        if validation['is_valid']:
            st.success("✅ Pattern is valid")
        else:
            st.warning(f"⚠️ Validation issues: {', '.join(validation['issues'])}")
        
        # Download button
        import json
        result_json = {
            'extracted_text': result['extracted_text'],
            'confidence': result['confidence'],
            'timestamp': result['timestamp']
        }
        
        st.download_button(
            label="📥 Download Result (JSON)",
            data=json.dumps(result_json, indent=2),
            file_name=f"ocr_result_{result['timestamp']}.json",
            mime="application/json"
        )
        
        # Full OCR text
        with st.expander("View Full OCR Text"):
            st.text_area("Full Text", result['full_text'], height=200)
        
    else:
        st.markdown('<div class="error-box">❌ Pattern "_1_" not found in image</div>', 
                   unsafe_allow_html=True)
        
        # Show what was found
        if result.get('full_text'):
            with st.expander("View Full OCR Text"):
                st.text_area("Full Text", result['full_text'], height=200)
            
            # Check for similar patterns
            similar = st.session_state.text_extractor.find_similar_patterns(result['full_text'])
            if similar:
                st.info(f"Found similar patterns: {', '.join(similar)}")
        else:
            st.error("No text could be extracted from the image")


def main():
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📦 Shipping Label OCR Extractor</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract tracking numbers with "_1_" pattern from shipping labels</div>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ocr_backend = st.selectbox(
            "OCR Engine",
            ["tesseract", "easyocr", "both"],
            help="Select OCR backend (EasyOCR requires additional installation)"
        )
        
        preprocessing_method = st.selectbox(
            "Preprocessing Method",
            ["adaptive", "otsu", "basic", "none"],
            help="Image preprocessing technique"
        )
        
        apply_enhancement = st.checkbox(
            "Apply Image Enhancement",
            value=True,
            help="Apply denoising and contrast enhancement"
        )
        
        apply_rotation = st.checkbox(
            "Apply Auto-Rotation",
            value=True,
            help="Automatically detect and fix image orientation"
        )
        
        st.divider()
        st.header("📊 About")
        st.info("""
        **Target Pattern:** Text containing "_1_"
        
        **Example:** 163233702292313922_1_lWV
        
        **Accuracy Target:** ≥75%
        
        **Supported Formats:** PNG, JPG, JPEG
        """)
        
        if st.session_state.results_history:
            st.divider()
            st.header("📈 Statistics")
            total = len(st.session_state.results_history)
            successful = sum(1 for r in st.session_state.results_history if r['success'])
            st.metric("Total Processed", total)
            st.metric("Successful", successful)
            st.metric("Success Rate", f"{(successful/total)*100:.1f}%")
    
    # Initialize OCR engine
    if st.session_state.ocr_engine is None or st.session_state.ocr_engine.backend != ocr_backend:
        with st.spinner("Initializing OCR engine..."):
            st.session_state.ocr_engine = ocr_engine.OCREngine(backend=ocr_backend)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a shipping label image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the shipping label"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Process button
            if st.button("🔍 Extract Text", type="primary", use_container_width=True):
                process_image(image, preprocessing_method, apply_enhancement, apply_rotation, col2)
        else:
            st.info("👆 Upload an image to begin extraction")
    
    with col2:
        st.header("📋 Results")
        if 'current_result' in st.session_state and st.session_state.current_result:
            display_results(st.session_state.current_result)
        else:
            st.info("Results will appear here after processing")
    
    # Results history
    if st.session_state.results_history:
        st.divider()
        st.header("📚 Processing History")
        
        with st.expander("View All Results", expanded=False):
            for idx, result in enumerate(reversed(st.session_state.results_history[-10:])):
                st.markdown(f"**Result {len(st.session_state.results_history) - idx}**")
                if result['success']:
                    st.success(f"✅ Extracted: `{result['extracted_text']}`")
                else:
                    st.error("❌ No pattern found")
                st.caption(f"Confidence: {format_confidence(result.get('confidence', 0))}")
                st.divider()


if __name__ == "__main__":
    main()