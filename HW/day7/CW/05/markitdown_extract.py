"""
MarkItDown 套件示範：
將「有文字層」PDF 直接轉換為 Markdown（不使用 OCR）
"""

from markitdown import MarkItDown

def pdf_to_markdown(pdf_path: str, output_md: str):
    md = MarkItDown()
    result = md.convert(pdf_path)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(result.markdown)

    print(f"MarkItDown 轉換完成：{output_md}")


if __name__ == "__main__":
    pdf_path = "example.pdf"
    output_md = "example_markitdown.md"
    pdf_to_markdown(pdf_path, output_md)

