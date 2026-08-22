import json
import re
import requests
from pathlib import Path
from typing import Dict, Any, List

# ============================================================
# Ollama configuration
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

DEFAULT_FALLBACK_MODEL = "qwen3.5:4b"

def get_ollama_models() -> List[str]:
    """動態取得 Ollama 目前已下載的模型清單"""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
        response.raise_for_status()
        data = response.json()
        models = [model["name"] for model in data.get("models", [])]
        return models if models else [DEFAULT_FALLBACK_MODEL]
    except Exception:
        return [DEFAULT_FALLBACK_MODEL]

def extract_json_from_text(text: str) -> str:
    """提取文字中的 JSON 部分"""
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()

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
4. 保留原意，勿自行總結。

你只能輸出純 JSON 格式，不要包含任何 Markdown 標記或開頭說明。

JSON 格式範例：
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
"""

# ============================================================
# Ollama client
# ============================================================

def call_ollama(
    text: str,
    context: str = "",
    topic: str = "",
    model_name: str = DEFAULT_FALLBACK_MODEL,
) -> Dict[str, Any]:

    user_prompt = f"""
請校正下面這段 Whisper 語音辨識結果。

【影片主題】
{topic if topic else "未知"}

【前後文】
{context if context else "無"}

【需要校正的逐字稿】
{text}
"""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 1024,  # 限制最大生成 token，防止無止盡生成
        },
    }

    try:
        # 連線與回應 Timeout 改為 (5, 90) 秒，超時會立刻跳過，不會讓整隻程式卡住
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=(5, 90),
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        
        json_str = extract_json_from_text(content)
        return json.loads(json_str)

    except requests.exceptions.Timeout:
        warning_msg = f"⚠️ LLM 回應超時 (>90s)，已自動跳過此 Chunk 寫入原始文本 (模型: {model_name})"
        print(f"\n[Warning] {warning_msg}")
        return {
            "corrected_text": text,
            "changes": [],
            "status": "ollama_timeout",
        }
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, Exception) as e:
        warning_msg = f"⚠️ LLM 校正未正常回應或 JSON 解析失敗 (模型: {model_name})"
        print(f"\n[Warning] {warning_msg} | 詳細錯誤: {e}")
        return {
            "corrected_text": text,
            "changes": [],
            "status": "ollama_error",
        }

# ============================================================
# Correct one chunk
# ============================================================

def correct_chunk(
    chunk: Dict[str, Any],
    context: str = "",
    topic: str = "",
    model_name: str = DEFAULT_FALLBACK_MODEL,
) -> Dict[str, Any]:

    original_text = chunk["text"]

    result = call_ollama(
        text=original_text,
        context=context,
        topic=topic,
        model_name=model_name,
    )

    corrected_text = result.get("corrected_text", original_text)
    changes = result.get("changes", [])
    status = result.get("status", "success")

    return {
        "chunk_id": chunk["chunk_id"],
        "start": chunk["start"],
        "end": chunk["end"],
        "duration": chunk["duration"],
        "original_text": original_text,
        "corrected_text": corrected_text,
        "changes": changes,
        "status": status,
        "segment_ids": chunk.get("segment_ids", []),
        "estimated_tokens": chunk.get("estimated_tokens", 0),
    }

# ============================================================
# Correct all chunks
# ============================================================

def correct_chunks(
    chunks: List[Dict[str, Any]],
    topic: str = "",
    model_name: str = DEFAULT_FALLBACK_MODEL,
) -> List[Dict[str, Any]]:

    results = []

    for i, chunk in enumerate(chunks):
        print(f"\n[LLM] Chunk {i + 1}/{len(chunks)} (使用模型: {model_name})")

        context_parts = []
        if i > 0:
            context_parts.append("上一個 chunk：\n" + chunks[i - 1]["text"])
        if i + 1 < len(chunks):
            context_parts.append("下一個 chunk：\n" + chunks[i + 1]["text"])

        context = "\n\n".join(context_parts)

        result = correct_chunk(
            chunk,
            context=context,
            topic=topic,
            model_name=model_name,
        )

        results.append(result)

        print("\n原文：", result.get("original_text", ""))
        print("\n校正：", result.get("corrected_text", ""))

        if result.get("changes"):
            print("\n修改：")
            for change in result["changes"]:
                print(f"  {change.get('original')} → {change.get('corrected')}")

    return results