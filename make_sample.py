from PIL import Image, ImageDraw, ImageFont

lines = [
    "SAMPLE CLINIC NOTE (FICTIONAL)",
    "Patient: Test Patient One",
    "Date: 12 Oct 2026",
    "Complaint: Cough and mild fever for 3 days",
    "Notes: Drinking fluids, no other symptoms reported",
]

img = Image.new("RGB", (1100, 420), "white")
draw = ImageDraw.Draw(img)
font = ImageFont.load_default(size=36)

y = 30
for line in lines:
    draw.text((30, y), line, fill="black", font=font)
    y += 70

img.save("sample_note.png")
print("Created sample_note.png")