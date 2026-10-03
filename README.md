# Music-Synchronization

Score–audio alignment, phrase-boundary experiments, and an interactive research portfolio.

## Project layout

| Directory | Purpose |
| --- | --- |
| `src/` | MuseScore export, MIDI beat extraction, Chroma/DLNCO MrMsDTW alignment, beat mapping and evaluation |
| `phrase_boundary_overnight/` | Separate phrase-start research workspace; inspect each experiment's protocol and completion audit before citing results |
| `music_structure_portfolio/` | TypeScript/Vite portfolio with recorded research replay and a read-only Node server |
| `dataset/`, `output/` | Local datasets and generated alignment outputs |

## Alignment

Use the existing Python environment with `numpy`, `pandas`, `scipy`, `mido`,
`librosa`, `soundfile`, `synctoolbox`, and `tqdm`. MuseScore 4 and the local
ASAP/MAESTRO datasets are needed for exports and full alignment.

```powershell
& '.\.venv\Scripts\python.exe' src/pipeline.py
```

Review the input paths and configuration at the top of `src/pipeline.py` first.
`src/pipeline2.py` runs a seeded sample of 24 works; it replaces sampled alignment
outputs and aggregate CSVs, so run it only when intentionally regenerating results.
The 1,000 Hz feature grid is a time resolution, not a measured accuracy guarantee.
Out-of-range beat mappings remain NaN/null.

Validation reports timing errors and threshold hit rates. Its time-DP fallback
uses reference times to choose correspondences; those results are conditional
on matching and are not an independent phrase-boundary accuracy estimate.

## Research portfolio

```powershell
Set-Location music_structure_portfolio
npm ci
npm run build
npm test
npm run test:server
npm start
```

Open `http://127.0.0.1:4173`. See the portfolio README for replay contracts,
provenance, and deployment configuration. Recorded replay is not live inference.

## Lightweight alignment regression check

```powershell
& '.\.venv\Scripts\python.exe' tests/test_time_matching.py
```

Keep raw datasets, model checkpoints, generated caches, and private meeting
materials out of public commits. Existing local files are not automatically
part of a Git commit.
