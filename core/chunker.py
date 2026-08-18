
import re
from typing import List, Dict, Any

# ============================================================
# Semantic Chunker
# ============================================================
#
# 目的：
#   將 Whisper segment 組成適合後續 LLM 校正的語意區塊。
#
# 原則：
#   1. 不直接修改原始文字
#   2. 優先在句號/問號/驚嘆號/換段語氣處切割
#   3. 避免單一 chunk 過大
#   4. 保留原始 segment ID 與 timestamp
#   5. 每個 chunk 都可以獨立送給 LLM
#
# 注意：
#   中文沒有空白，所以 token 這裡先使用「字元數估計」，
#   不把它當成真正 tokenizer 的 token 數。
# ============================================================

DEFAULT_MAX_CHARS = 1200
DEFAULT_MIN_CHARS = 250
DEFAULT_MAX_SECONDS = 150

STRONG_BOUNDARY = re.compile(r"[。！？!?；;]\s*$")
WEAK_BOUNDARY = re.compile(r"[，,、：:]\s*$")

# 避免把常見縮寫/小數/編號等誤判成句尾。
SENTENCE_END_RE = re.compile(r"(?<=[。！？!?])")


def estimate_tokens(text: str) -> int:
    """
    粗略估算 token。
    中文/日文/韓文大致以字元為主要成本，英文則以
    whitespace word + 標點做簡化估算。

    這不是 tokenizer，只用來控制 chunk 大小。
    """
    if not text:
        return 0

    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    punctuation = len(re.findall(r"[^\w\s]", text, flags=re.UNICODE))

    return cjk + latin_words + max(0, punctuation // 2)


def split_segment_text(text: str) -> List[str]:
    """
    將單一 Whisper segment 依強句尾切成較小的句子。
    如果沒有強句尾，保留原 segment。
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _new_chunk() -> Dict[str, Any]:
    return {
        "chunk_id": None,
        "start": None,
        "end": None,
        "duration": None,
        "text": "",
        "segment_ids": [],
        "sentence_count": 0,
        "estimated_tokens": 0,
        "avg_word_probability": None,
    }


def build_chunks(
    segments: List[Dict[str, Any]],
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> List[Dict[str, Any]]:
    """
    以 Whisper segment 為基本時間單位，建立 semantic chunks。

    切割優先級：
        強句尾 > 時間上限 > 字數上限 > 弱句尾

    不會修改 segment 原文。
    """
    chunks = []
    current = _new_chunk()

    def flush():
        nonlocal current

        text = current["text"].strip()
        if not text:
            current = _new_chunk()
            return

        current["chunk_id"] = len(chunks) + 1
        current["duration"] = round(
            current["end"] - current["start"], 3
        )

        probs = current.pop("_probabilities", [])
        if probs:
            current["avg_word_probability"] = round(
                sum(probs) / len(probs), 4
            )
        else:
            current["avg_word_probability"] = None

        chunks.append(current)
        current = _new_chunk()

    for segment in segments:
        seg_id = segment.get("id")
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        seg_text = segment.get("text", "").strip()

        if not seg_text:
            continue

        sentences = split_segment_text(seg_text)

        # 如果 segment 被切成多句，先用文字比例估算各句時間。
        # 這不會改動原始 segment timestamp，只用於 chunk 邊界的近似。
        total_len = max(len(seg_text), 1)
        cursor = seg_start

        for sentence in sentences:
            ratio = len(sentence) / total_len
            sentence_duration = max((seg_end - seg_start) * ratio, 0.0)
            sentence_start = cursor
            sentence_end = min(
                seg_end,
                cursor + sentence_duration
            )
            cursor = sentence_end

            sentence_chars = len(sentence)
            sentence_tokens = estimate_tokens(sentence)

            if current["start"] is None:
                current["start"] = sentence_start

            proposed_text = (
                sentence
                if not current["text"]
                else current["text"] + " " + sentence
            )

            proposed_duration = sentence_end - current["start"]

            # 已經有內容，而且下一句會讓 chunk 明顯超標：
            # 優先在句子邊界切。
            exceeds_chars = (
                len(proposed_text) > max_chars
            )
            exceeds_time = (
                proposed_duration > max_seconds
            )

            if current["text"] and (exceeds_chars or exceeds_time):
                flush()
                current["start"] = sentence_start
                proposed_text = sentence
                proposed_duration = sentence_end - sentence_start

            current["text"] = proposed_text
            current["end"] = sentence_end
            current["segment_ids"].append(seg_id)
            current["sentence_count"] += 1
            current["estimated_tokens"] += sentence_tokens

            probabilities = []
            for word in segment.get("words", []) or []:
                p = word.get("probability")
                if isinstance(p, (int, float)):
                    probabilities.append(float(p))

            if "_probabilities" not in current:
                current["_probabilities"] = []

            current["_probabilities"].extend(probabilities)

            # 如果單一句子本身就超大，允許它獨立成 chunk。
            if (
                len(current["text"]) >= max_chars
                or (current["end"] - current["start"]) >= max_seconds
            ):
                flush()

    flush()

    # 後處理：去除重複 segment ID，避免同一 segment 被句子拆分後重複列出。
    for chunk in chunks:
        chunk["segment_ids"] = list(dict.fromkeys(chunk["segment_ids"]))

    return chunks
