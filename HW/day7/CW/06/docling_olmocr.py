from docling.datamodel.pipeline_options import ApiVlmOptions, ResponseFormat


def olmocr2_vlm_options(
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
    endpoint: str = "https://ws-01.wade0426.me/v1/chat/completions",
):
    return ApiVlmOptions(
        url=endpoint,
        params={
            "model": model,
            "max_tokens": 4096,
        },
        prompt="Convert this page to markdown.",
        response_format=ResponseFormat.MARKDOWN,
        timeout=120,
        scale=2.0,
        temperature=0.0,
    )


if __name__ == "__main__":
    options = olmocr2_vlm_options()
    print("OLM OCR 2 VLM options prepared successfully.")

