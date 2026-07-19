# Changelog

All notable changes to this App are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-13

### Added

- Floating five-action navigation for the current state, camera, manual sleep
  entry, historical statistics, and Settings.
- Dedicated camera view with live controls, nursery signals, and recent
  captures.
- A direct back-to-Home-Assistant control when the app runs through Ingress or
  the Home Assistant proxy.
- Global manual sleep dialog that preserves the caregiver's current view.
- Full two-day sleep planning for today and tomorrow, including predicted naps,
  bedtime, wake-up time, confidence windows, and a complete predicted night.
- Restored sleep entry and editing controls with calendar/hour/minute pickers,
  nap/awake/night types, awake pauses, settling and wake-up context, mood,
  comments, and nearest start/middle/end camera frames.
- Additive storage for structured sleep details and pauses; existing databases
  upgrade in place and portable CSV exports retain the new fields.
- An in-app Boifun Baby 6T guide for enabling ONVIF, adding the integration to
  Home Assistant, and selecting the resulting camera entity.

### Changed

- Live camera sessions now negotiate WebRTC for low latency and show MJPEG only
  as an explicit fallback; the latest stored frame remains visible until the
  first WebRTC frame is playing.
- Cry light alerts adapt to each entity's supported colour mode, including
  XY-only Hue lights, while preserving and restoring the previous state.
- Cry notifications and light alerts start concurrently so a slow Home
  Assistant integration cannot delay the other alert path.
- The Day rhythm now uses the original light visual treatment, while Night
  remains dark.

### Fixed

- Align circular segments to the real wake/bed boundaries in Day mode and the
  real bedtime/wake boundaries in Night mode.
- Exclude explicit awake events and structured awake pauses from sleep totals,
  charts, and the recent-night ribbon.
- Show every stored frame inside an edited sleep segment instead of a misleading
  seven-frame nearest-image subset; Start, Middle, and End now jump within the
  complete interval timeline.
- Restore the hidden delete action for automatically detected sleep segments
  while preserving all associated captures and model analyses.

## [0.3.0] - 2026-07-13

### Added

- Rich daily and nightly sleep visualization restored from the original
  Esteban dashboard: calendar navigation, circular sleep segments, bedtime and
  wake-up boundaries, and duration summaries.
- Responsive day/night controls that work with the public, configurable baby,
  home, time zone, and shared multi-home history model.

## [0.2.1] - 2026-07-13

### Fixed

- Close every SQLite connection after its transaction context exits, avoiding
  file-descriptor exhaustion in long-running Home Assistant and standalone
  installations.

## [0.2.0] - 2026-07-13

### Added

- Public Settings flow for verified history export, import, and source
  retirement between Home Assistant installations.
- Portable ZIP exports with CSV tables and original images grouped by location
  and date for use in external analysis tools.
- Lossless SQLite snapshot, manifest, counts, and SHA-256 validation for every
  imported image.
- Destination import receipts that must match the pending export before source
  history and images can be explicitly deleted.

### Security

- Transfer archives exclude Settings, encryption keys, Home Assistant tokens,
  AI API keys, and private camera URLs.
- Archive extraction rejects unsafe paths, duplicate names, symbolic links,
  unsupported compression, corrupt databases, and mismatched content hashes.

## [0.1.1] - 2026-07-12

### Added

- Configurable location IDs and names for histories shared across homes.
- Location tags on frames, sleep sessions, and cry events.
- `--location-id` support in the legacy SQLite importer.

### Changed

- Version 1 databases upgrade in place without deleting or moving image files.

## [0.1.0] - 2026-07-11

### Added

- Admin-only Home Assistant Ingress application.
- One configurable baby profile and optional camera.
- Manual sleep history and prediction surface.
- Optional cry detection through a Home Assistant binary sensor or audio stream.
- Multi-light alerts with previous-state restoration.
- Optional Gemini, OpenAI, and local OpenAI-compatible image labeling.
- Encrypted local secret storage and configurable frame retention.
- Standalone authenticated Docker deployment.
- Signed multi-architecture `amd64` and `aarch64` image publishing workflow.
