import json
import requests
from pathlib import Path
from typing import Dict, Any, List


# ============================================================
# Ollama configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL_NAME = "qwen2.5:32b"


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = r"""
你是一個「技術影片逐字稿校正器」。

你的工作不是重新寫文章，而是修正語音辨識系統產生的逐字稿。

請遵守以下規則：

1. 只修正明顯的語音辨識錯誤、同音字錯誤、專有名詞錯誤。
2. 保留原本說話者的語氣與句子結構。
3. 不要新增原文沒有提到的資訊。
4. 不要刪除原文的重要資訊。
5. 不要自行總結。
6. 不要把口語內容改寫成正式文章。
7. 如果原文其實可能是正確的，不要任意修改。
8. 對專業術語要特別注意上下文。
9. 中文技術名詞必須依據上下文判斷，而不能只看單一字詞。
10. 如果無法確定，寧可保留原文。
11. 專有名詞優先保守修改；只有當上下文、領域知識與語法三者一致時才修改。若不確定，保留原文。

例如：

原文：
「複數可以表示成負數的形式」

如果上下文是在講電路、交流電、相量：
「負數」可能是 Whisper 將「複數」辨識錯誤。

又例如：
「共二」
如果上下文是在講複數、相量、交流電路，
很可能應該是「共軛」。

但是不要看到「負數」就一律改成「複數」。

請根據上下文判斷。

你只能輸出 JSON。
不要輸出 Markdown。
不要輸出 ```json。

JSON 格式：

{
  "corrected_text": "校正後文字",
  "changes": [
    {
      "original": "原文字",
      "corrected": "修正文字",
      "reason": "修正原因"
    }
  ]
}

如果沒有需要修正：

{
  "corrected_text": "原文",
  "changes": []
}
"""


# ============================================================
# Ollama client
# ============================================================

def call_ollama(
    text: str,
    context: str = "",
    topic: str = "",
) -> Dict[str, Any]:

    user_prompt = f"""
請校正下面這段 Whisper 語音辨識結果。

【影片主題】
{topic if topic else "未知"}

【前後文】
{context if context else "無"}

【需要校正的逐字稿】
{text}

請只修正真正可能的辨識錯誤。
專有名詞優先保守修改；只有當上下文、領域知識與語法三者一致時才修改。若不確定，保留原文。
"""

    payload = {
        "model": MODEL_NAME,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        "stream": False,

        "format": "json",

        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600,
    )

    response.raise_for_status()

    data = response.json()

    content = data["message"]["content"]

    return json.loads(content)


# ============================================================
# Correct one chunk
# ============================================================

def correct_chunk(
    chunk: Dict[str, Any],
    context: str = "",
    topic: str = "",
) -> Dict[str, Any]:

    original_text = chunk["text"]

    result = call_ollama(
        text=original_text,
        context=context,
        topic=topic,
    )

    corrected_text = result.get(
        "corrected_text",
        original_text,
    )

    changes = result.get(
        "changes",
        [],
    )

    return {
        "chunk_id": chunk["chunk_id"],
        "start": chunk["start"],
        "end": chunk["end"],
        "duration": chunk["duration"],

        "original_text": original_text,
        "corrected_text": corrected_text,

        "changes": changes,

        "segment_ids": chunk.get(
            "segment_ids",
            [],
        ),

        "estimated_tokens": chunk.get(
            "estimated_tokens",
            0,
        ),
    }


# ============================================================
# Correct all chunks
# ============================================================

def correct_chunks(
    chunks: List[Dict[str, Any]],
    topic: str = "",
) -> List[Dict[str, Any]]:

    results = []

    for i, chunk in enumerate(chunks):

        print(
            f"\n[LLM] "
            f"Chunk {i + 1}/{len(chunks)}"
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context_parts = []

        if i > 0:
            context_parts.append(
                "上一個 chunk：\n"
                + chunks[i - 1]["text"]
            )

        if i + 1 < len(chunks):
            context_parts.append(
                "下一個 chunk：\n"
                + chunks[i + 1]["text"]
            )

        context = "\n\n".join(
            context_parts
        )

        # ----------------------------------------------------
        # LLM correction
        # ----------------------------------------------------

        result = correct_chunk(
            chunk,
            context=context,
            topic=topic,
        )

        results.append(result)

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            "\n原文："
        )

        print(
            result["original_text"]
        )

        print(
            "\n校正："
        )

        print(
            result["corrected_text"]
        )

        if result["changes"]:

            print(
                "\n修改："
            )

            for change in result["changes"]:

                print(
                    f"  {change.get('original')}"
                    f" → "
                    f"{change.get('corrected')}"
                )

    return results