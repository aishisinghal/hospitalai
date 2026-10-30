import pymupdf

doc = pymupdf.open()
page = doc.new_page()
rect = pymupdf.Rect(36, 36, 559, 36 + 523 * 420 / 1100)
page.insert_image(rect, filename="sample_note.png")
doc.save("sample_note.pdf")
print("Created sample_note.pdf")