import pdfplumber

input_pdf = "example.pdf"
output_md = "example_pdfplumber.md"

with pdfplumber.open(input_pdf) as pdf:
    with open(output_md, "w", encoding="utf-8") as f:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                f.write(f"# Page {i}\n\n")
                f.write(text)
                f.write("\n\n")

print("✅ PDF text extracted and saved as Markdown.")
