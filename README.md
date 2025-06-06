# Video Translator

Just put your video in `input/video.mp4`, add your GROQ_API_KEY in `main.py` and install the requirements with:

```bash
pip install -r requirements.txt
```

The script accepts parameters to choose what to process:
* `--srt` generates spanish subtitles (`output/translated.srt`)
* `--dub` generates spanish dubbing (`output/final_video.mp4`)

## Examples:

```bash
python main.py --srt
```

Solo doblaje:
```bash
python main.py --dub
```

Ambos:
```bash
python main.py --srt --dub
```