# Baby Monitor for Home Assistant

An open-source, local-first baby monitor that runs next to Home Assistant. It
combines sleep tracking, optional camera snapshots, cry-triggered light alerts,
notifications, and optional AI image labels in one admin-only web interface.

[![Open your Home Assistant instance and add this app repository](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fvictoriano%2Fhome-assistant-baby-monitor)

> [!IMPORTANT]
> This project is not a medical device and is not a replacement for direct
> adult supervision or a certified baby monitor. Cry and image detection can
> miss events or produce false alarms.

## Preview

![Baby Monitor dashboard showing sleep status and the next rest window](docs/screenshots/dashboard.jpg)

<table>
  <tr>
    <td width="68%">
      <img src="docs/screenshots/rhythm.jpg" alt="Sleep rhythm with sleep and cry event history">
    </td>
    <td width="32%">
      <img src="docs/screenshots/dashboard-mobile.jpg" alt="Responsive Baby Monitor dashboard on mobile">
    </td>
  </tr>
</table>

Screenshots use synthetic demo data. No real child, camera, or Home Assistant
credentials are included.

## What it does

- Lets each household select its own Home Assistant camera or private RTSP URL.
- Tracks sleep manually and keeps a local history with predictions.
- Tags frames, sleep sessions, and cries with a configurable home/location so
  one shared database can preserve history from multiple houses.
- Detects crying from a Home Assistant `binary_sensor` or an optional audio
  stream, then activates one or more selected Home Assistant lights.
- Restores every light to its previous state after the alert.
- Labels camera frames with Gemini, OpenAI, or a local OpenAI-compatible server
  such as Ollama. AI is optional and disabled by default.
- Stores settings, events, and frames under `/data`; no public Home Assistant
  `/local` directory is used.
- Encrypts API keys and private stream URLs at rest. Secret values are never
  returned by the API.

## Install on Home Assistant OS

1. Use the **Add repository** button above.
2. In **Settings → Apps → App store**, open **Baby Monitor**.
3. Select **Install**, then **Start**.
4. Enable **Show in sidebar** and open the app.
5. Complete Settings: baby profile, camera, cry source, lights, notifications,
   retention, and optional AI provider.

The App uses Home Assistant Ingress and the Supervisor token automatically. No
Home Assistant long-lived access token is needed on Home Assistant OS.

See [the App documentation](baby_monitor/DOCS.md) for detailed setup and
privacy notes.

## Run with Docker

Home Assistant Container/Core users can run the same image separately:

```bash
cp .env.example .env
# Replace the token in .env with a long random value.
docker compose up -d
```

Open `http://localhost:8099`, sign in with the token from `.env`, and configure
the Home Assistant URL plus a long-lived access token in Settings.

Docker Compose binds to `127.0.0.1` by default. If a reverse proxy runs on
another host, explicitly change `BABY_MONITOR_BIND_ADDRESS` in `.env`, protect
the proxy with TLS, set `BABY_MONITOR_COOKIE_SECURE=1`, and restrict the network
path to trusted clients. Do not publish port 8099 directly to the internet.

Published images:

```text
ghcr.io/victoriano/home-assistant-baby-monitor:<version>
ghcr.io/victoriano/home-assistant-baby-monitor:latest
```

Both `amd64` and `aarch64` are built and signed by GitHub Actions.

### Verify a release

Release builds use the committed `uv.lock` and `package-lock.json`; CI rejects a
stale Python lock and audits both Python and npm dependencies. GitHub Actions
are pinned to immutable commits and updated through Dependabot.

Each release publishes a keyless Cosign signature, GitHub build-provenance
attestations, and an SPDX JSON SBOM for each architecture. Prefer a numbered
version or image digest over `latest`. After authenticating to GHCR, verify the
GitHub provenance with:

```bash
gh attestation verify \
  oci://ghcr.io/victoriano/home-assistant-baby-monitor:0.1.1 \
  --repo victoriano/home-assistant-baby-monitor \
  --signer-workflow victoriano/home-assistant-baby-monitor/.github/workflows/release.yaml
```

The release assets are named
`home-assistant-baby-monitor-<version>-<architecture>.spdx.json`. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for dependency licenses and
how to locate the exact Debian/FFmpeg corresponding source.

## Local development

Requirements: Python 3.12+, Node.js 24+, and npm.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

cd frontend
npm ci
npm run build
cd ..

BABY_MONITOR_RUNTIME=development \
BABY_MONITOR_DATA_DIR=./data \
BABY_MONITOR_FRONTEND_DIR=./frontend/dist \
python -m uvicorn baby_monitor.main:app --host 127.0.0.1 --port 8099 --reload
```

Run the checks with:

```bash
pytest
ruff check .
cd frontend && npm test && npm run build
```

## Import data from the legacy private schema

The repository includes a conservative migration command for the earlier
private SQLite schema. It opens the source database read-only and performs a
dry run by default:

```bash
baby-monitor-migrate-legacy --source /private/path/legacy.sqlite3
```

After reviewing the JSON plan, apply it to a private data directory outside any
Git worktree:

```bash
baby-monitor-migrate-legacy \
  --source /private/path/legacy.sqlite3 \
  --target /private/path/baby-monitor-data \
  --location-id madrid \
  --apply
```

`--location-id` records where the imported events and frames were captured.
Existing version 1 databases upgrade automatically and keep every image file;
their previous records receive the backwards-compatible location `home`.

## Use one history across multiple homes

Set a stable **Location ID** and readable **Location name** in Settings for each
Home Assistant instance, for example `madrid` / `Madrid` and `granada` /
`Granada`. New frames, sleep sessions, and cry events are tagged with the
active location and shown that way in history.

To share one history, both deployments must point at the same private `/data`
directory, but only one Baby Monitor process may write to that SQLite database
at a time. This works well when one machine moves between homes and only one
Home Assistant instance is active. Do not mount the live SQLite directory over
internet file sharing or run concurrent writers; use a proper synchronization
or server architecture for that scenario.

Changing location never deletes or moves existing frames. Retention remains a
separate explicit setting; choose **Forever** if every image must be preserved.

Stop the App before replacing its `/data` contents and keep both the source and
target out of public web roots and repositories.

## Privacy and security

- Camera and AI features are opt-in. Cloud image upload requires a separate,
  explicit consent toggle.
- Without a cloud AI provider, images and metadata remain on the host running
  the app.
- API keys, access tokens, and stream URLs are encrypted in `/data`. Keep Home
  Assistant and Docker backups private because the encryption key is backed up
  with the encrypted values.
- The Home Assistant App is admin-only, accepts Ingress traffic only from the
  Supervisor gateway, and does not request host network, Docker socket, or full
  host access.
- Standalone mode requires an administrator token.

Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Status

Version `0.1.1` adds multi-home location tags and a non-destructive database
migration. Camera and AI providers remain optional; core sleep tracking works
without either.

## License

The project source is [MIT licensed](LICENSE). Bundled and container runtime
dependencies retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
