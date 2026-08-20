# core/transcriber.py
import subprocess
import yt_dlp
from pathlib import Path
from faster_whisper import WhisperModel
from core.config import AUDIO_DIR
import torch


def download_youtube_audio(url: str) -> dict:
    ydl_opts = {
        # 擴充 Client 嘗試順序，增加備援
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'mweb', 'web']
            }
        },
        # 格式選擇放寬：優先純音訊 (bestaudio/ba)，若無則選擇最佳綜合串流 (b/best)
        'format': 'bestaudio/ba/b/best',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'outtmpl': str(AUDIO_DIR / "%(id)s.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info.get("id")
    candidates = [
        p for p in AUDIO_DIR.glob(f"{video_id}.*")
        if p.suffix.lower() not in {".part", ".ytdl"}
    ]

    if not candidates:
        raise FileNotFoundError(f"找不到下載的音訊：{video_id}")

    return {
        "id": video_id,
        "title": info.get("title", "youtube_video"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "source_file": candidates[0]
    }

def convert_to_wav(source_file: Path, video_id: str) -> Path:
    output_wav = AUDIO_DIR / f"{video_id}.wav"
    cmd = ["ffmpeg", "-y", "-i", str(source_file), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(output_wav)]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_wav


def run_transcription(audio_path: Path, model_size="large-v3-turbo", device=None, compute_type=None):
    # 自動判斷是否有支援 CUDA 的 NVIDIA 顯卡
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), vad_filter=True, beam_size=5)
    
    results = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            results.append({
                "id": len(results) + 1,
                "start": seg.start,
                "end": seg.end,
                "text": text
            })
    return results, info