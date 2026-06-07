# coding=utf-8
import mimetypes
from pathlib import Path

from archaeo.io.files import list_files, json_dump, get_file_created_time, get_file_modified_time
from peblo.providers import BaseLlmProvider
from peblo.commons.io.images import image_to_base64


def image_to_data_url(file_path: str | Path) -> str:
    path = Path(file_path)
    mime_type = mimetypes.guess_type(path)[0] or 'image/png'
    img_b64 = image_to_base64(path)
    return f'data:{mime_type};base64,{img_b64}'


def ocr_by_llm_openai(provider: BaseLlmProvider, file_path: str | Path) -> dict[str, str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    image_url = image_to_data_url(path)

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': '请从此图片中提取所有文字，并仅返回文字内容。',
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_url,
                    },
                },
            ],
        }
    ]

    text = provider.chat(messages)

    return {
        'file': str(path),
        'engine': 'llm',
        'text': text,
    }


def ocr_by_llm(provider: BaseLlmProvider, file_path: str) -> dict[str, str]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(file_path)

    img_b64 = image_to_base64(file_path)

    messages = [
        {
            'role': 'user',
            # "content": "Extract all readable text from this image. Only return the text.",
            "content": "请从此图片中提取所有文字，并仅返回文字内容。",
            "images": [img_b64],
        }
    ]

    text = provider.chat(messages)
    return {
        'file': file_path,
        'engine': 'llm',
        'text': text
    }


def ocr_dir(provider: BaseLlmProvider, img_dir: str | Path):
    result = []
    result_file = './dir_ocr_result.json'
    files = list_files(img_dir, pattern='*.*')
    files = sorted(files, key=lambda f: get_file_modified_time(f))
    for file in files:
        try:
            ocr_result = ocr_by_llm_openai(provider, file)
        except Exception as e:
            print(f'ocr failed: {file}, reason: {e}')
            ocr_result = {'file': file, 'text': None}
        result.append(ocr_result)
    json_dump(result, result_file)



if __name__ == '__main__':
    import time
    from peblo.providers import OllamaProvider, OpenRouterProvider, OpenRouterModels

    # llm = OllamaProvider(model='deepseek-ocr:3b')
    # llm = OllamaProvider(model='qwen3-vl:8b-instruct')

    # llm = OpenRouterProvider(OpenRouterModels.gpt_5_4_mini) # 中文竖排不可用
    # llm = OpenRouterProvider(OpenRouterModels.gpt_5_4_nano) # ocr 中文不太可用
    llm = OpenRouterProvider(OpenRouterModels.gemini_flash_lite_3_1)
    # llm = OpenRouterProvider(OpenRouterModels.qwen3_6_plus)
    # llm = OpenRouterProvider(OpenRouterModels.qwen3_6_flash)
    # llm = OpenRouterProvider(OpenRouterModels.claude_haiku_4_5)  # ocr 中文不太可用

    # start = time.time()
    # print(ocr_by_llm_openai(llm, '/Users/andersc/Downloads/images/出梁庄记-1.png'))
    # print(f'time elapsed: {round(time.time() - start, ndigits=3)}')

    ocr_dir(llm, '/Users/andersc/data/papers/pdf/LLM Agents/鸽友在看')
