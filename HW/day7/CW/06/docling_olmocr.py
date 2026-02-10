from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    ResponseFormat,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def remote_vlm_options(
    model: str = "gemma-3-27b-it",
    base_url: str = "https://ws-06.huannago.com/v1",
    prompt: str = "Convert this page to clean, structured markdown.",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key: str = "",
) -> ApiVlmOptions:

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return ApiVlmOptions(
        url=f"{base_url}/chat/completions",
        params={
            "model": model,
            "max_tokens": max_tokens,
        },
        headers=headers,
        prompt=prompt,
        timeout=180,        # 遠端模型保守拉長
        scale=2.0,          # OCR + 表格建議 2.0
        temperature=temperature,
        response_format=ResponseFormat.MARKDOWN,
    )


# === VLM Pipeline 設定（CW/06：遠端 IDP）===
pipeline_options = VlmPipelineOptions(
    enable_remote_services=True
)

pipeline_options.vlm_options = remote_vlm_options(
    temperature=0.0,
)

# === 建立文件轉換器 ===
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        )
    }
)

# === 轉換 PDF ===
result = doc_converter.convert("sample_table.pdf")

# === 匯出 Markdown ===
md_text = result.document.export_to_markdown()

with open("output_olmocr.md", "w", encoding="utf-8") as f:
    f.write(md_text)

print("✅ 已成功產出 output_olmocr.md")

