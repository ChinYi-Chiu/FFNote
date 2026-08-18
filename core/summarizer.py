# core/summarizer.py
import json
import requests
from typing import Dict, Any, List

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:32b"

CHUNK_NOTE_SYSTEM_PROMPT = r"""
你是一個嚴謹的技術筆記整理專家。
請針對輸入的影片逐字稿區塊進行「重點筆記提煉」。

輸出規則：
1. 提煉出 2~4 個核心要點（Bullet points）。
2. 標註出現的關鍵技術名詞或核心概念。
3. 保持客觀，不要加油添醋。
4. 輸出格式必須為 JSON，包含 key: "summary_points" (陣列) 與 "keywords" (陣列)。

範例 JSON:
{
  "summary_points": ["重點一...", "重點二..."],
  "keywords": ["名詞A", "名詞B"]
}
"""

FINAL_SYNTHESIS_SYSTEM_PROMPT = r"""
你是一位高級技術文案主筆。請根據以下多個逐字稿分段筆記，彙整成一份結構完整、清晰易讀的「影片全片技術摘要筆記」。

請包含以下章節格式（Markdown）：
1. 📌 全片核心概述 (100-200字)
2. 🔑 關鍵主題與技術重點 (分點詳細說明)
3. 💡 總結與核心價值
"""

def summarize_single_chunk(
    chunk: Dict[str, Any],
    topic: str = "",
    model_name: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    text = chunk.get("corrected_text", chunk.get("text", ""))
    
    user_prompt = f"""
【影片主題】{topic if topic else "未指定"}
【逐字稿區塊 ({chunk.get('start', 0):.1f}s -> {chunk.get('end', 0):.1f}s)】
{text}
"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": CHUNK_NOTE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2}
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=300)
        res.raise_for_status()
        content = res.json()["message"]["content"]
        data = json.loads(content)
    except Exception as e:
        data = {"summary_points": [f"筆記生成失敗: {str(e)}"], "keywords": []}

    return {
        "chunk_id": chunk.get("chunk_id"),
        "start": chunk.get("start"),
        "end": chunk.get("end"),
        "summary_points": data.get("summary_points", []),
        "keywords": data.get("keywords", [])
    }


def summarize_chunks(
    corrected_chunks: List[Dict[str, Any]],
    topic: str = "",
    model_name: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    分段提煉筆記，並在最後生成全片總結 (Map-Reduce 模式)
    """
    chunk_notes = []
    all_points_text = []

    for i, chunk in enumerate(corrected_chunks):
        note = summarize_single_chunk(chunk, topic=topic, model_name=model_name)
        chunk_notes.append(note)
        
        points_str = "\n".join([f"- {p}" for p in note["summary_points"]])
        all_points_text.append(
            f"### [Chunk {note['chunk_id']:03d}] ({note['start']:.1f}s - {note['end']:.1f}s)\n{points_str}"
        )

    # 全片總結 Synthesis
    combined_notes = "\n\n".join(all_points_text)
    synthesis_prompt = f"【主題】: {topic}\n\n【各分段重點彙整】:\n{combined_notes}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": FINAL_SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.3}
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=300)
        res.raise_for_status()
        final_summary = res.json()["message"]["content"]
    except Exception as e:
        final_summary = f"全片彙整失敗: {str(e)}"

    return {
        "chunk_notes": chunk_notes,
        "final_summary": final_summary
    }