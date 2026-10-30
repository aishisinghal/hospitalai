from PIL import Image, ImageDraw, ImageFont
import pytesseract

# Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Make a fictional test image with text on it
img = Image.new("RGB", (900, 200), "white")
draw = ImageDraw.Draw(img)
font = ImageFont.load_default(size=40)
draw.text((20, 60), "Test Patient - Fever for 3 days", fill="black", font=font)

# Read the text back with OCR
text = pytesseract.image_to_string(img, lang="eng")
print("OCR RESULT:", text.strip())