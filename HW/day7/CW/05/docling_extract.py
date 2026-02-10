from docling.document_converter import DocumentConverter

input_pdf = "example.pdf"
output_md = "example_docling.md"

converter = DocumentConverter()
doc = converter.convert(input_pdf)

with open(output_md, "w", encoding="utf-8") as f:
    f.write(doc.document.export_to_markdown())

print("✅ Docling extraction completed.")
