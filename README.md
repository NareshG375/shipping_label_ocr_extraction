# 📦 shipping_label_ocr_extraction

### OCR-based Shipping Label Text Extraction & Pattern Detection

`shipping_label_ocr_extraction` is a lightweight Python tool designed to extract text from shipping labels using OCR and automatically detect structured patterns such as `_1_`. It is ideal for:

- Logistics automation  
- Parcel classification  
- Barcode & label OCR  
- Pattern-based workflow labeling  

---

## 🚀 Project Overview
This project automates the extraction of structured information from shipping labels. It leverages OCR to read text, applies preprocessing techniques to improve recognition accuracy, and detects patterns like `_1_` for tracking IDs or batch markers. The system is designed to be lightweight, fast, and easy to integrate into existing logistics workflows.

---

## 🛠️ Installation Instructions

### 1️⃣ Install Python
Requires Python **3.9 – 3.13**
```bash
python --version
```

### 2️⃣ Clone Repository
```bash
git clone https://github.com/NareshG375/shipping_label_ocr_extraction.git
cd shipping_label_ocr_extraction
```

### 3️⃣ Create Virtual Environment
```bash
python -m venv .venv
```
Activate:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/macOS:
```bash
source .venv/bin/activate
```

### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Install Tesseract OCR (Windows)
Download (UB Mannheim recommended): [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)  

Ensure:
- Add Tesseract to PATH  
- Install language packs  

Default path:
```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Verify installation:
```bash
tesseract --version
```

Configure in Python:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## ▶️ Usage Guide
Run the Streamlit application:
```bash
streamlit run app.py
```

Run tests:
```bash
pytest -v -s
```

Example input:
```
163629705512179520 _1_ Ips
```

Example output:
```json
{
  "extracted_text": "163629705512179520_1_Ips",
  "confidence": 0.85,
  "accuracy":1.0,
  "timestamp": "20251130_083137"
}
```

---

## 🧠 Technical Approach

### OCR Method / Model Used
- **Tesseract OCR** (primary, best results with `--psm 3`)   
- EasyOCR  

### Preprocessing Techniques
- Resize images to improve recognition  
- Convert to grayscale  
- Apply thresholding  
- Noise removal  

### Text Extraction Logic
1. Preprocess the image  
2. Apply OCR engine  
3. Detect patterns such as `_1_` 
4. Return structured JSON with extracted text, confidence, accuracy and timestamp  

### Accuracy Calculation Methodology
- Compare OCR output against manually labeled ground truth  
- Compute confidence score per extraction  
- Evaluate overall accuracy as:
```text
Accuracy = (Correct Extractions / Total Extractions) * 100%
```

---

## 📊 Performance Metrics
- Achieved **~85% accuracy** on the internal test set using Tesseract OCR `--psm 3`  
- Performance may vary with image quality, font, and orientation  

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|-----------|---------|
| OCR misreads due to noisy images | Applied grayscale, thresholding, and noise removal preprocessing |
| Pattern detection errors | Regex-based pattern matching for robust extraction |
| Handling multiple OCR engines | Configurable OCR modes and PSM settings |

---

## 🌟 Future Improvements
- Integrate deep learning OCR models for higher accuracy (e.g., TrOCR, PaddleOCR advanced models)  
- Automatic detection of label orientation  
- Batch processing of multiple images  
- Web API deployment for real-time label extraction  

---

## 🧪 Test Suite (pytest)
Includes test cases for:
- OCR validation  
- Pattern detection  
- PSM mode testing

