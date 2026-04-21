# Lecture Recorder Notes

A tiny tl;dv-style prototype for OBS recordings. Upload a lecture recording and get:

- timestamped transcript
- short study summary
- possible important legal terms
- review warnings
- visible processing progress after the app receives the file
- Google Drive file browser for larger recordings
- audio/video player with clickable transcript timestamps
- downloadable `.txt` notes

This version is designed for Google Colab and can run on free Colab, but Colab can disconnect and GPU access is not guaranteed.

## Recommended Recording Flow

1. Record the lecture in OBS.
2. Confirm OBS captured screen and audio.
3. Save as `.mkv`, then remux to `.mp4` if you want.
4. Upload the file into this app.
5. Review important law wording against the original OBS recording.

## Run In Google Colab

Open `Lecture_Recorder_Notes_Colab.ipynb` in Google Colab, upload `app.py` and `requirements.txt` into the Colab file panel, then run all cells.

Or create a new Colab notebook, upload these project files, then run:

```python
!pip install -r requirements.txt
!python app.py
```

Open the public Gradio link that appears in the output.

## Use Files From Google Drive

For long audio files, upload the recording to Google Drive first, then mount Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

If the file is here:

```text
My Drive / Lecture Recorder App / test-audio.m4a
```

In the app:

1. Choose `Google Drive` as the file source.
2. Set the Drive folder to `Lecture Recorder App`.
3. Click `Quet file Google Drive`.
4. Pick `test-audio.m4a` from the dropdown.

The app cannot show exact browser upload percentage before Gradio receives the file. Once the file is received, it shows processing progress while loading Whisper, transcribing, and creating notes.

## Clickable Timestamps

After transcription, open the `Review` tab. The original audio/video player appears above the clickable transcript. Click any orange timestamp to jump the player to that moment.

The timestamp is counted from the start of the uploaded audio/video file.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Model Size

By default this uses the `small` Whisper model. It is a good first balance for free Colab.

To use a faster but less accurate model:

```powershell
$env:WHISPER_MODEL_SIZE="base"
python app.py
```

To use a more accurate but slower model:

```powershell
$env:WHISPER_MODEL_SIZE="medium"
python app.py
```

## Important Law Lecture Warning

The transcript is AI-generated. It can be wrong. For legal study, always verify exact words, case names, statutes, and definitions against the original recording.
