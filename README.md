# 📦 shipping_label_ocr_extraction
### OCR-based Shipping Label Text Extraction & Pattern Detection

shipping_label_ocr_extraction is a lightweight Python tool designed to extract text from shipping labels using OCR and automatically detect structured patterns such as _1_.  
It is ideal for:

- Logistics automation  
- Parcel classification  
- Barcode & label OCR  
- Pattern-based workflow labeling  


## 🚀 Features

### 🔍 OCR Extraction
Supports multiple OCR engines:
- Tesseract OCR   (Current Used and better result on  --psm 3)
- PaddleOCR
- EasyOCR


### 🧠 Pattern Detection
Automatically detects patterns such as:

```
_1_, _5_, _10_
```

Useful for:
- Tracking IDs  
- Routing codes  
- Batch markers  


### 🖼️ Image Preprocessing
- Resize  
- Grayscale  
- Threshold  
- Noise removal  


### 🧪 Test Suite (pytest)
Includes test cases for:
- OCR validation  
- Pattern detection  
- PSM testing  


### ⚙️ Configurable OCR Modes
Supports:
- Multiple Tesseract PSM modes  
- PaddleOCR orientation detection  
- EasyOCR textline options  


## 📥 Example Input
```
163629705512179520 _1_ Ips
```


## 📤 Example Output
```json
{
  "extracted_text": "163629705512179520_1_Ips",
  "confidence": 0.85,
  "timestamp": "20251129_225508"
}
```


# 🛠️ Installation Guide (Complete Steps)

## 1️⃣ Install Python
Requires Python **3.9 – 3.13**
```
python --version
```

## 2️⃣ Clone Repository
```
git clone https://github.com/NareshG375/shipping_label_ocr_extraction.git
cd shipping_label_ocr_extraction
```

## 3️⃣ Create Virtual Environment
```
python -m venv .venv
```

Activate:
```
.venv\Scripts\activate
```

## 4️⃣ Install Dependencies

Install:
```
pip install -r requirements.txt
```


# 🧰 Install Tesseract OCR (Windows)

## 5️⃣ Download Tesseract
Recommended (UB Mannheim):
https://github.com/UB-Mannheim/tesseract/wiki

## 6️⃣ Install Tesseract
Ensure:
✔ Add Tesseract to PATH  
✔ Install language packs  

Default path:
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## 7️⃣ Verify
```
tesseract --version
```

## 8️⃣ Configure in Python
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

# ▶️ Run App
```
streamlit app.py
```

# 🧪 Run Tests
```
pytest -v -s
```
