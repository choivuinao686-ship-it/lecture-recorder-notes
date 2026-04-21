import html
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gradio as gr


MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".webm", ".mov", ".avi"}
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive")
DEFAULT_DRIVE_FOLDER = "Lecture Recorder App"
SEEKTIME_JS = """
() => {
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-seek-seconds]");
    if (!button) {
      return;
    }

    const seconds = Number(button.dataset.seekSeconds);
    const players = Array.from(document.querySelectorAll("audio, video"));
    const player = players.find((item) => item.offsetParent !== null) || players[0];

    if (!player || Number.isNaN(seconds)) {
      return;
    }

    player.currentTime = seconds;
    player.play();
  });
}
"""
LEGAL_TERMS = {
    "act",
    "appeal",
    "burden",
    "case",
    "civil",
    "clause",
    "contract",
    "court",
    "criminal",
    "damages",
    "defendant",
    "doctrine",
    "duty",
    "evidence",
    "judgment",
    "jurisdiction",
    "liability",
    "negligence",
    "offence",
    "plaintiff",
    "precedent",
    "principle",
    "reasonable",
    "remedy",
    "rights",
    "rule",
    "section",
    "statute",
    "tort",
}


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())


def resolve_drive_folder(drive_folder: str | None) -> Path:
    cleaned_folder = (drive_folder or "").strip().strip('"').strip("'")
    if not cleaned_folder:
        cleaned_folder = DEFAULT_DRIVE_FOLDER

    folder_path = Path(cleaned_folder)
    if not folder_path.is_absolute():
        folder_path = DEFAULT_DRIVE_ROOT / cleaned_folder
    return folder_path


def list_drive_media_files(drive_folder: str | None) -> list[str]:
    folder_path = resolve_drive_folder(drive_folder)
    if not folder_path.exists() or not folder_path.is_dir():
        return []

    files = []
    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            try:
                relative = path.relative_to(DEFAULT_DRIVE_ROOT)
                files.append(str(relative).replace("\\", "/"))
            except ValueError:
                files.append(str(path))
    return sorted(files)


def refresh_drive_files(drive_folder: str | None):
    folder_path = resolve_drive_folder(drive_folder)
    files = list_drive_media_files(drive_folder)

    if not folder_path.exists():
        message = f"Chưa thấy folder Google Drive: `{folder_path}`"
    elif not files:
        message = f"Không thấy file audio/video nào trong `{folder_path}`"
    else:
        message = f"Tìm thấy {len(files)} file trong `{folder_path}`"

    return gr.update(choices=files, value=(files[0] if files else None)), message


def resolve_media_path(
    source_mode: str,
    uploaded_file: str | None,
    drive_file: str | None,
) -> str:
    if source_mode == "Google Drive":
        if not drive_file:
            raise gr.Error("Chọn file từ danh sách Google Drive trước nha.")
        media_path = Path(drive_file)
        if not media_path.is_absolute():
            media_path = DEFAULT_DRIVE_ROOT / drive_file
    elif uploaded_file:
        media_path = Path(uploaded_file)
    else:
        raise gr.Error("Upload file audio/video từ máy trước nha.")

    if not media_path.exists():
        raise gr.Error(f"Không tìm thấy file: {media_path}")
    if not media_path.is_file():
        raise gr.Error(f"Đường dẫn này không phải file: {media_path}")
    return str(media_path)


def describe_file(media_path: str) -> str:
    path = Path(media_path)
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"Đã nhận file: `{path.name}` ({size_mb:.1f} MB)"


def validate_media_file(media_path: str) -> None:
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]
    try:
        result = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise gr.Error("Không thấy ffprobe trong môi trường này, nên app chưa kiểm tra được file media.") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "ffprobe không đọc được file."
        raise gr.Error(f"File media này không đọc được: {detail}")

    streams = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if "audio" not in streams:
        raise gr.Error("File này không có audio stream, nên app không thể transcribe.")


def transcribe_file(media_path: str, progress: gr.Progress) -> list[TranscriptSegment]:
    from faster_whisper import WhisperModel

    progress(0.05, desc="Đang tải Whisper model...")
    model = WhisperModel(MODEL_SIZE, device="auto", compute_type="auto")
    progress(0.15, desc="Đã tải model. Đang bắt đầu nghe file...")
    segments, info = model.transcribe(
        media_path,
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    )

    transcript_segments = []
    duration = max(float(getattr(info, "duration", 0) or 0), 1.0)
    for item in segments:
        text = item.text.strip()
        if text:
            transcript_segments.append(
                TranscriptSegment(start=item.start, end=item.end, text=text)
            )

        done = min(0.9, 0.15 + (float(item.end) / duration) * 0.75)
        progress(done, desc=f"Đang transcribe... {format_time(item.end)} / {format_time(duration)}")

    progress(0.92, desc="Transcribe xong. Đang tạo transcript...")
    return transcript_segments


def transcript_to_text(segments: list[TranscriptSegment]) -> str:
    return "\n".join(
        f"[{format_time(segment.start)} - {format_time(segment.end)}] {segment.text}"
        for segment in segments
    )


def transcript_to_html(segments: list[TranscriptSegment]) -> str:
    rows = []
    for segment in segments:
        start = html.escape(format_time(segment.start))
        end = html.escape(format_time(segment.end))
        text = html.escape(segment.text)
        rows.append(
            f"""
            <div class="transcript-row">
              <button class="seek-button" data-seek-seconds="{segment.start:.2f}">
                {start}
              </button>
              <span class="transcript-end">{end}</span>
              <span class="transcript-text">{text}</span>
            </div>
            """
        )

    return (
        """
        <style>
          .transcript-list {
            display: grid;
            gap: 8px;
            margin-top: 8px;
          }

          .transcript-row {
            align-items: start;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            display: grid;
            gap: 8px;
            grid-template-columns: auto auto 1fr;
            padding: 10px;
          }

          .seek-button {
            background: #f97316;
            border: 0;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-weight: 700;
            padding: 5px 9px;
          }

          .seek-button:hover {
            background: #ea580c;
          }

          .transcript-end {
            color: #6b7280;
            font-size: 0.92rem;
            padding-top: 5px;
          }

          .transcript-text {
            line-height: 1.55;
            padding-top: 3px;
          }
        </style>
        <p>Click timestamp màu cam để phát lại đúng đoạn trong player phía trên.</p>
        <div class="transcript-list">
        """
        + "\n".join(rows)
        + "</div>"
    )


def score_segments(segments: list[TranscriptSegment]) -> list[tuple[float, TranscriptSegment]]:
    all_words = normalize_words(" ".join(segment.text for segment in segments))
    frequencies = Counter(word for word in all_words if len(word) > 3)
    if not frequencies:
        return []

    scored = []
    for segment in segments:
        words = normalize_words(segment.text)
        if not words:
            continue
        keyword_score = sum(frequencies[word] for word in words) / len(words)
        legal_bonus = sum(1 for word in words if word in LEGAL_TERMS) * 1.8
        length_penalty = 0.4 if len(words) < 6 else 0
        scored.append((keyword_score + legal_bonus - length_penalty, segment))
    return scored


def build_summary(segments: list[TranscriptSegment], max_items: int = 8) -> str:
    scored = score_segments(segments)
    if not scored:
        return "No summary could be generated."

    selected = sorted(scored, key=lambda item: item[0], reverse=True)[:max_items]
    selected = sorted(selected, key=lambda item: item[1].start)

    bullets = []
    for _, segment in selected:
        sentence = segment.text.strip()
        if not sentence.endswith((".", "?", "!")):
            sentence += "."
        bullets.append(f"- [{format_time(segment.start)}] {sentence}")
    return "\n".join(bullets)


def build_key_terms(segments: list[TranscriptSegment], max_terms: int = 20) -> str:
    words = normalize_words(" ".join(segment.text for segment in segments))
    useful_words = [
        word
        for word in words
        if len(word) > 4 and word not in {"there", "their", "about", "which", "would", "could"}
    ]
    terms = Counter(useful_words).most_common(max_terms)
    if not terms:
        return "No key terms found."
    return ", ".join(term for term, _ in terms)


def build_review_notes(segments: list[TranscriptSegment]) -> str:
    text = " ".join(segment.text for segment in segments)
    uncertain_markers = ["inaudible", "unknown", "[", "]"]
    warnings = []

    if any(marker in text.lower() for marker in uncertain_markers):
        warnings.append("- Review lines with unclear or bracketed words against the original recording.")
    if len(segments) < 3:
        warnings.append("- Recording looks very short, so the notes may be incomplete.")

    warnings.append("- For law lectures, verify important wording, case names, statutes, and definitions against the audio.")
    warnings.append("- Keep the OBS recording as the source of truth; this transcript is an AI-generated working copy.")
    return "\n".join(warnings)


def media_player_updates(media_path: str):
    suffix = Path(media_path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return (
            gr.update(value=media_path, visible=False),
            gr.update(value=media_path, visible=True),
        )
    return (
        gr.update(value=media_path, visible=True),
        gr.update(value=None, visible=False),
    )


def write_download_file(prefix: str, content: str) -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix="lecture_notes_"))
    output_path = temp_dir / f"{prefix}.txt"
    output_path.write_text(content, encoding="utf-8")
    return str(output_path)


def check_upload_status(media_file: str | None) -> str:
    if not media_file:
        return "Chưa có file nào được upload."
    return describe_file(media_file)


def update_source_visibility(source_mode: str):
    use_upload = source_mode == "Upload từ máy"
    return (
        gr.update(visible=use_upload),
        gr.update(visible=not use_upload),
    )


def process_recording(
    source_mode: str,
    media_file: str | None,
    drive_file: str | None,
    progress: gr.Progress = gr.Progress(),
):
    progress(0.01, desc="Đang kiểm tra file...")
    media_path = resolve_media_path(source_mode, media_file, drive_file)
    status = describe_file(media_path)
    validate_media_file(media_path)

    try:
        segments = transcribe_file(media_path, progress)
    except gr.Error:
        raise
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise gr.Error(f"Lúc transcribe file này bị lỗi: {message}") from exc
    if not segments:
        raise gr.Error("Không nhận ra speech trong file này. Kiểm tra lại xem audio có tiếng nói rõ không.")

    progress(0.94, desc="Đang tạo summary và key terms...")
    transcript = transcript_to_text(segments)
    clickable_transcript = transcript_to_html(segments)
    summary = build_summary(segments)
    key_terms = build_key_terms(segments)
    review_notes = build_review_notes(segments)

    full_notes = (
        "LECTURE SUMMARY\n"
        f"{summary}\n\n"
        "KEY TERMS\n"
        f"{key_terms}\n\n"
        "REVIEW NOTES\n"
        f"{review_notes}\n\n"
        "FULL TIMESTAMPED TRANSCRIPT\n"
        f"{transcript}\n"
    )
    download_file = write_download_file("lecture_transcript_and_notes", full_notes)
    audio_update, video_update = media_player_updates(media_path)
    progress(1.0, desc="Xong rồi.")

    return (
        status,
        audio_update,
        video_update,
        transcript,
        clickable_transcript,
        summary,
        key_terms,
        review_notes,
        download_file,
    )


with gr.Blocks(title="Lecture Recorder Notes", js=SEEKTIME_JS) as demo:
    gr.Markdown(
        """
        # Lecture Recorder Notes

        Upload an audio/video recording, or enter a Google Drive file path, then create a timestamped transcript and study notes.
        Keep the original recording for checking exact legal wording.
        """
    )

    gr.Markdown(
        """
        **Lưu ý:** tiến độ upload từ máy lên Gradio chỉ hiện sau khi server nhận file xong.
        Nếu file lớn, cách ổn hơn là để file trong Google Drive rồi nhập đường dẫn Drive bên dưới.
        """
    )

    source_mode = gr.Radio(
        ["Upload từ máy", "Google Drive"],
        value="Upload từ máy",
        label="Nguồn file",
    )

    with gr.Group(visible=True) as upload_group:
        media = gr.File(
            label="Upload audio/video từ máy",
            file_types=sorted(MEDIA_EXTENSIONS),
            type="filepath",
        )
        upload_status = gr.Markdown("Chưa có file nào được upload.")

    with gr.Group(visible=False) as drive_group:
        drive_folder = gr.Textbox(
            label="Folder Google Drive để quét file",
            value=DEFAULT_DRIVE_FOLDER,
            placeholder='Ví dụ: Lecture Recorder App hoặc /content/drive/MyDrive/Lecture Recorder App',
        )
        with gr.Row():
            refresh_drive_button = gr.Button("Quét file Google Drive")
            drive_scan_status = gr.Markdown("Bấm nút quét để app lấy danh sách file từ Drive.")
        drive_file = gr.Dropdown(
            label="Chọn file từ Google Drive",
            choices=[],
            allow_custom_value=False,
        )

    run_button = gr.Button("Transcribe and create notes", variant="primary")

    with gr.Tab("Review"):
        gr.Markdown("Player nằm ngay trên transcript. Click timestamp màu cam để phát lại đúng đoạn.")
        audio_player = gr.Audio(label="Audio player", visible=False)
        video_player = gr.Video(label="Video player", visible=False)
        clickable_transcript_output = gr.HTML(label="Clickable transcript")

    with gr.Tab("Transcript"):
        transcript_output = gr.Textbox(label="Timestamped transcript", lines=18)
    with gr.Tab("Summary"):
        summary_output = gr.Markdown(label="Summary")
    with gr.Tab("Key Terms"):
        terms_output = gr.Textbox(label="Possible important terms", lines=4)
    with gr.Tab("Review Notes"):
        review_output = gr.Markdown(label="Review notes")

    download_output = gr.File(label="Download transcript and notes")

    media.change(
        fn=check_upload_status,
        inputs=media,
        outputs=upload_status,
    )

    source_mode.change(
        fn=update_source_visibility,
        inputs=source_mode,
        outputs=[upload_group, drive_group],
    )

    refresh_drive_button.click(
        fn=refresh_drive_files,
        inputs=drive_folder,
        outputs=[drive_file, drive_scan_status],
    )

    run_button.click(
        fn=process_recording,
        inputs=[source_mode, media, drive_file],
        outputs=[
            upload_status,
            audio_player,
            video_player,
            transcript_output,
            clickable_transcript_output,
            summary_output,
            terms_output,
            review_output,
            download_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        allowed_paths=[
            str(Path.cwd()),
            tempfile.gettempdir(),
            "/content/drive/MyDrive",
        ],
    )
