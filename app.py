# app.py
import json
from pathlib import Path
import gradio as gr

from core.config import OUTPUT_DIR
from core.transcriber import download_youtube_audio, convert_to_wav, run_transcription
from core.chunker import build_chunks
from core.corrector import correct_chunks, get_ollama_models
from core.summarizer import summarize_chunks

# 自動取得本地已安裝的 Ollama 模型
available_ollama_models = get_ollama_models()
default_ollama_model = available_ollama_models[0]

def process_pipeline(url_or_file, topic, whisper_model_name, ollama_model_name, progress=gr.Progress()):
    progress(0.0, desc="開始處理...")
    
    # 1. 取得音訊
    if url_or_file.startswith("http"):
        progress(0.1, desc="正在下載 YouTube 音訊...")
        info = download_youtube_audio(url_or_file)
        wav_path = convert_to_wav(info["source_file"], info["id"])
        title = info["title"]
    else:
        wav_path = Path(url_or_file)
        title = wav_path.stem

    # 2. Whisper 轉錄
    progress(0.3, desc="正在執行 Whisper 轉錄...")
    segments, info = run_transcription(wav_path, model_size=whisper_model_name)
    raw_text = "\n".join([f"[{s['start']:.1f}s -> {s['end']:.1f}s] {s['text']}" for s in segments])

    # 3. 語意切片
    progress(0.5, desc="正在進行語意切片 (Chunking)...")
    chunks = build_chunks(segments)

    # 4. LLM 錯別字校正 (將 UI 選取的 ollama_model_name 帶入)
    progress(0.65, desc="正在執行 Ollama 逐字稿校正...")
    corrected_chunks = correct_chunks(chunks, topic=topic, model_name=ollama_model_name)
    
    corrected_text_list = []
    diff_list = []
    for c in corrected_chunks:
        corrected_text_list.append(f"[{c['start']:.1f}s -> {c['end']:.1f}s] {c['corrected_text']}")
        if c.get("changes"):
            for ch in c["changes"]:
                diff_list.append(f"[{c['chunk_id']:03d}] {ch.get('original')} → {ch.get('corrected')} ({ch.get('reason')})")

    full_corrected_text = "\n".join(corrected_text_list)
    diff_report = "\n".join(diff_list) if diff_list else "未發現明顯錯別字修正。"

    # 5. LLM Chunked 重點筆記與全片總結
    progress(0.85, desc="正在生成 Chunked 重點筆記與總結...")
    summary_res = summarize_chunks(corrected_chunks, topic=topic, model_name=ollama_model_name)

    chunk_notes_md = []
    for n in summary_res["chunk_notes"]:
        pts = "\n".join([f"- {p}" for p in n["summary_points"]])
        kw = ", ".join(n["keywords"])
        chunk_notes_md.append(f"#### ⏱️ [{n['start']:.1f}s - {n['end']:.1f}s] Chunk {n['chunk_id']}\n{pts}\n\n**關鍵字**: `{kw}`")

    final_markdown = f"# 📝 全片技術摘要\n\n{summary_res['final_summary']}\n\n---\n# 🧩 分段詳細筆記\n\n" + "\n\n".join(chunk_notes_md)

    progress(1.0, desc="處理完成！")
    return raw_text, full_corrected_text, diff_report, final_markdown

# Gradio 介面配置
with gr.Blocks(title="YouTube 語音轉錄與 LLM 筆記專家") as demo:
    gr.Markdown("# 🎙️ YouTube 語音轉錄與 LLM 筆記整理系統")
    
    with gr.Row():
        with gr.Column():
            input_source = gr.Textbox(label="YouTube 網址 或 本地音訊路徑", placeholder="[https://www.youtube.com/watch?v=](https://www.youtube.com/watch?v=)...")
            topic_input = gr.Textbox(label="影片主題 / 領域知識 (幫助 LLM 校正專有名詞)", placeholder="例如：交流電路、相量、微控制器")
            
            with gr.Row():
                whisper_model = gr.Dropdown(["large-v3-turbo", "large-v3", "medium", "small"], value="large-v3-turbo", label="Whisper 模型")
                # 動態載入 Ollama 下拉選單與預設值
                ollama_model = gr.Dropdown(choices=available_ollama_models, value=default_ollama_model, label="Ollama 模型")
            
            submit_btn = gr.Button("🚀 開始轉錄與筆記生成", variant="primary")

        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("📄 原始逐字稿"):
                    out_raw = gr.TextArea(lines=18, label="Whisper 原始輸出")
                with gr.TabItem("✏️ 校正後逐字稿"):
                    out_corrected = gr.TextArea(lines=12, label="LLM 校正內文")
                    out_diff = gr.TextArea(lines=6, label="修正紀錄 (Diff)")
                with gr.TabItem("📚 全片重點筆記"):
                    out_notes = gr.Markdown()

    submit_btn.click(
        fn=process_pipeline,
        inputs=[input_source, topic_input, whisper_model, ollama_model],
        outputs=[out_raw, out_corrected, out_diff, out_notes]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)