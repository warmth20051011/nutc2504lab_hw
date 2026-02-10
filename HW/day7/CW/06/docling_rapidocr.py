from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# === OCR Pipeline 設定（IDP 流程）===
pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_options=RapidOcrOptions(
        lang=["ch", "en"]
    )
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

# === 轉換 PDF ===
result = doc_converter.convert("sample_table.pdf")

# === 匯出成 Markdown ===
md_text = result.document.export_to_markdown()

# === 寫入 md 檔案 ===
with open("output_rapidocr.md", "w", encoding="utf-8") as f:
    f.write(md_text)

print("✅ 已成功產出 output_rapidocr.md")

