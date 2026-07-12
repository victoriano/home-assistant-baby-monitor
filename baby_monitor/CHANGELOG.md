# Changelog

All notable changes to this App are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
