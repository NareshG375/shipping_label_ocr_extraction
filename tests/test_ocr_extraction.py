import re
import pytesseract
from PIL import Image
import pytest


# OCR configurations
CONFIGS = [
    "--psm 6",
    "--psm 4",
    "--psm 11",
    "--psm 3",
]

# Pattern:
#  - 18 digits
#  - underscore
#  - 1
#  - underscore
#  - word (Ips, UPS, DHL, etc.)
AWB_PATTERN = re.compile(r"\b(\d{18}_1_[A-Za-z]+)\b")


@pytest.mark.parametrize("config", CONFIGS)
def test_ocr_extracts_full_awb(config):
    """
    Validate if OCR detects the full AWB string:
    e.g. 163629705512179520_1_Ips
    """
    img = Image.open("abc.jpg")

    text = pytesseract.image_to_string(img, config=config)

    print("\n========================================")
    print("Config:", config)
    print("OCR Output:")
    print(text)
    print("========================================\n")

    match = AWB_PATTERN.search(text)

    if match:
        print("Detected Pattern:", match.group())
    else:
        print("❌ Pattern  NOT found for config:", config)

    # Test must PASS only if Pattern detected
    assert match, f"Full  pattern not found for config: {config}"
