# Resolve Clip Search

Local React + FastAPI tool for searching timeline clips in the current DaVinci Resolve project and jumping the playhead straight to matches.

## What it does

- Connects to the open DaVinci Resolve project through the official Python scripting API
- Indexes every video `TimelineItem` across every timeline into a local SQLite database
- Searches clip metadata, filenames, tags, markers, transcripts, and stored visual-description text
- Filters results by clip type
- Jumps Resolve to the chosen timeline and timecode with `SetCurrentTimecode()`
- Exposes a local web app that preserves the supplied UI design

## Project layout

- `backend/`: FastAPI app, Resolve integration, indexing, and search services
- `frontend/`: React app using the provided design
- `run.py`: starts the FastAPI server
- `.resolve-clip-search/resolve_clip_search.sqlite3`: local index database after first run

## Setup

1. Create a Python virtual environment and install backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install frontend dependencies:

```bash
cd frontend
npm install
```

3. Add your Gemini or Anthropic API key so visual descriptions can be generated during indexing.

Either export a Gemini key in your shell:

```bash
export GEMINI_API_KEY=your_api_key_here
```

Or create a local `.env` in the project root using `.env.example`. If both providers are configured, set `RESOLVE_CLIP_SEARCH_VISION_PROVIDER` explicitly.

4. Make sure DaVinci Resolve is running and the scripting API is enabled in:

`DaVinci Resolve > Preferences > General > External Scripting`

The backend auto-detects the default Resolve scripting paths on macOS, Windows, and Linux. You can also set:

- `RESOLVE_SCRIPT_API`
- `RESOLVE_SCRIPT_LIB`

## Run it

### Standalone launcher on macOS

You do not need Codex to use the tool after setup. Double-click:

- `open-resolve-clip-search.command` to start the local server and open the app in your browser
- `stop-resolve-clip-search.command` to stop the local server later

You can also drag `open-resolve-clip-search.command` into the Dock for one-click launching.

### Backend

```bash
source .venv/bin/activate
python run.py
```

The API serves at `http://127.0.0.1:8000`.

### Frontend dev server

```bash
cd frontend
npm run dev
```

The UI serves at `http://127.0.0.1:5173` and proxies `/api` to the backend.

### Production-style local app

Build the frontend once:

```bash
cd frontend
npm run build
```

Then start the backend and open:

`http://127.0.0.1:8000`

## Reindex behavior

- The `Reindex` button triggers the backend indexing pipeline
- Status polling reflects live connection state, indexing progress, and index stats
- The index is stored in `.resolve-clip-search/resolve_clip_search.sqlite3` and reused between launches
- On app launch, the backend loads the saved SQLite index if it already exists instead of re-indexing automatically
- Reindex only re-processes clips that are new or changed; unchanged clips are reused from disk by matching saved filename + duration
- If a supported vision key is present, indexing extracts chronological clip frames with `ffmpeg` and sends them to the configured provider for real visual descriptions
- If the key is missing or a frame cannot be analyzed, the app falls back to the heuristic description layer so indexing still completes

## Vision settings

- `RESOLVE_CLIP_SEARCH_VISION_PROVIDER`: defaults to `gemini` when `GEMINI_API_KEY` is present, otherwise `anthropic` when `ANTHROPIC_API_KEY` is present, otherwise `heuristic`
- `GEMINI_API_KEY`: enables Gemini vision indexing
- `ANTHROPIC_API_KEY`: optional alternative provider
- `RESOLVE_CLIP_SEARCH_VISION_MODEL`: defaults to `gemini-3.1-flash-lite` for Gemini and `claude-sonnet-4-6` for Anthropic
- `RESOLVE_CLIP_SEARCH_VISION_MAX_IMAGE_EDGE_PX`: defaults to `768`
- `RESOLVE_CLIP_SEARCH_VISION_TIMEOUT_SEC`: defaults to `60`

## Notes

- The vision path is designed to replace older heuristic descriptions on reindex once your provider key is configured.
- Frame extraction now uses one frame from the middle of clips up to 15 seconds, and one midpoint frame per 10-second chunk for longer clips.
- For large projects, reindex the current timeline or a small subset first so you can validate output quality and cost before running a full-project vision pass.
