from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    # === OCR Pipeline 設定 ===
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = RapidOcrOptions(
        force_full_page_ocr=True
    )

    # === 建立 Converter ===
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    # === 轉換 PDF ===
    result = converter.convert("sample_table.pdf")

    # === 輸出 Markdown ===
    with open("output_rapidocr.md", "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())

    print("✅ RapidOCR → Markdown 完成")


if __name__ == "__main__":
    main()

