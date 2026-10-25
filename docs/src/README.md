Scripts that draw the images in `docs/images/`:

- `make_charts.py`: `fit-chart-recent-models.png`, `demo-library.png` (needs matplotlib). The fit chart's dots are recent models at the
  sizes of real units in a working library; `fit-chart.png` is the earlier version.
- `make_architecture.py`: `architecture.svg` (open it in a browser and save it as PNG, or use any SVG renderer)
- Screenshots: `python3 deck/mcc_deck.py serve --demo`, then capture the page at 1440×1000 (desktop) and 390×844 (phone).
- `docs/media/lion-ai-models-reel.mp4` and `.gif` (the README reel, v1.5): `reel/` holds the three scripts that make it.
  1. Two demo decks: `python3 deck/mcc_deck.py serve --demo --demo-no-ollama --port 8297` and `… serve --demo --port 8298`.
  2. `python3 docs/src/reel/shoot.py <work>` drives headless Chromium through both pages (`cdp.py`, a small standard-library DevTools client):
     sign in, the shelf, the engine filter, the guard, a TensorFold recipe loading and playing, a long-context decode, the sweep, a quality run,
     history and alerts, then a phone. Stills at 3840 × 2160 (1920 × 1080 CSS px at 2×) go to `<work>/hi`.
  3. A two-phone composite (`14-phone.png`) and small copies in `<work>/png` for the closing wall, a `logo.png` in `<work>`, and
     `reel.json` (the captions and the camera's zoom targets, fractions of each still).
  4. `python3 docs/src/reel/reel.py <work>/reel.json <frames>` (Pillow) renders 1,799 frames at 1920 × 1080, 30 fps: a title card, a camera
     move over each still with its caption, a wall of every still and the end card.
  5. ffmpeg: `-c:v libx264 -pix_fmt yuv420p -crf 24 -tune stillimage -an -movflags +faststart` for the MP4 (no audio track), and a GIF at
     5 fps, 640 px wide, 48 colours (`palettegen` / `paletteuse` with Bayer dither).
- `screenshot-live-deck.png` is a still of a working deck (220 models, a TensorFold lane playing), from the 1.4 clip that the reel replaced.
- `screenshot-long-context.png`, `screenshot-history.png`, `screenshot-alerts.png`, `screenshot-signin.png` and `screenshot-phone.png` (v1.5)
  are crops of the reel's stills.
