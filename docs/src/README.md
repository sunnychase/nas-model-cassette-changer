Scripts that draw the images in `docs/images/`:

- `make_charts.py`: `fit-chart-recent-models.png`, `demo-library.png` (needs matplotlib). The fit chart's dots are recent models at the
  sizes of real units in a working library; `fit-chart.png` is the earlier version.
- `make_architecture.py`: `architecture.svg` (open it in a browser and save it as PNG, or use any SVG renderer)
- Screenshots: `python3 deck/mcc_deck.py serve --demo`, then capture the page at 1440×1000 (desktop) and 390×844 (phone).
- `docs/media/`: the README clip and its preview. Playwright captured a working deck page (headless Chromium, software rendering,
  read-only: GET requests only) as a still plus the cassette's reels at twelve fixed animation times, 5° apart; the reels' own
  spin rule was applied in the capture browser. The frames were composited into a 12 s loop with a slow zoom and encoded with
  ffmpeg: `nas-model-cassette-changer.mp4` (H.264, 1920×1058, 30 fps, no audio track) and `preview.webp` (animated WebP,
  960×530, 15 fps; GitHub plays it inline, the MP4 opens on click). `screenshot-live-deck.png` is the still.
