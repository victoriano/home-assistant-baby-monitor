# Baby Monitor documentation

## Installation

Add `https://github.com/victoriano/home-assistant-baby-monitor` as an App
repository, install **Baby Monitor**, start it, and open its web interface from
the sidebar. The App requires Home Assistant OS or another installation with
Supervisor support.

The App is available only to Home Assistant administrators. Its manifest relies
on Home Assistant's documented `panel_admin=true` default, uses Ingress on the
documented default port 8099, and receives the Supervisor-provided Home
Assistant API token automatically.

## First-run setup

Open **Settings** in the App and configure only the features you need.

### Baby profile

Set a display name, optional birth date, and IANA timezone such as
`Europe/Madrid`. One baby profile is supported in the first release.

### Home Assistant

In Home Assistant App mode, leave the connection mode on automatic. The App
uses the internal Home Assistant API proxy and does not need a user token.

In standalone Docker mode, set the Home Assistant base URL and a long-lived
access token. Create a dedicated Home Assistant user with only the access the
monitor needs whenever your Home Assistant setup permits it.

Use an HTTPS Home Assistant URL whenever possible. An `http://` URL sends the
long-lived token without transport encryption and is appropriate only on an
isolated, trusted local network.

The provided Compose file binds to `127.0.0.1`. To serve the interface through
a reverse proxy on another host, explicitly change `BABY_MONITOR_BIND_ADDRESS`,
use TLS, set `BABY_MONITOR_COOKIE_SECURE=1`, and restrict access to trusted
clients. Never expose port 8099 directly to the public internet.

### Camera (optional)

Choose either:

- A Home Assistant entity whose ID starts with `camera.`; or
- A private HTTP/RTSP stream URL stored as an encrypted secret.

Set the snapshot interval between 30 seconds and 24 hours. The camera can stay
disabled while sleep tracking and cry alerts continue to work.

### Cry detection (optional)

Choose one source:

- **Disabled**: no automatic cry alerts.
- **Binary sensor**: select an entity whose ID starts with `binary_sensor.`.
- **Audio stream**: provide a private stream URL and adjust the confirmation
  windows. This mode depends on audio quality and may need tuning.

Use the built-in test before relying on an automatic alert.

### Lights

Select up to 32 `light.` entities, an alert duration, brightness, and RGB color.
When a cry begins, the App snapshots each selected light's state, applies the
alert, and restores the previous state after the configured duration.

Do not use safety-critical lighting where an unexpected state change could
create a hazard.

### Notifications (optional)

Choose a Home Assistant notification service whose name starts with `notify.`
and any targets required by that service. Test it from Settings.

### AI image labels (optional)

Supported providers:

- Google Gemini
- OpenAI
- A local OpenAI-compatible endpoint, including suitable Ollama setups

Cloud providers require an API key. Every provider requires explicit consent
before frames are sent to its endpoint. The API key is encrypted at rest and
never returned to the browser. Disabling AI does not disable camera capture or
sleep tracking.

Every AI endpoint, including an OpenAI-compatible one you host yourself,
requires explicit image-sharing consent because the configured server receives
camera frames. Prefer HTTPS; use plain HTTP only for a trusted host on an
isolated local network.

### Retention

Raw frames are retained indefinitely by default. You can instead choose from 1
to 3650 days. Metadata remains available after an expired image is removed so
history does not silently change.

Estimate storage before choosing indefinite retention. Backups may grow quickly.

## Data and backups

The App stores its database, frames, settings, encryption key, and encrypted
secrets in `/data`. Supervisor includes this directory in App backups. Treat
those backups as private household data.

Do not publish:

- `/data` or any backup containing it;
- camera frames or audio samples;
- `settings.json`, `.secret.key`, or encrypted secret files;
- Home Assistant access tokens, AI API keys, or RTSP URLs.

## Network and permissions

The App requests only Home Assistant Core API access. It does not use host
networking, privileged mode, full host access, the Docker API, or Home
Assistant's configuration directory. Port 8099 is disabled by default because
Ingress is the supported interface.

## Troubleshooting

### The App does not start

Check the App log and confirm `/data` is writable. The health endpoint is
`/healthz`; readiness details are available at `/api/v1/health` after opening
the authenticated interface.

### No cameras or lights appear

Confirm that the entities exist and are enabled in Home Assistant, then use the
connection test in Settings. Camera IDs must start with `camera.` and light IDs
with `light.`.

### AI labeling fails

Verify the provider, model, base URL if applicable, API key, and cloud-consent
toggle. A successful connection test does not guarantee that every image will
be accepted by the selected model.

### Cry alerts are noisy or missed

For a binary sensor, tune that sensor in Home Assistant. For audio mode, adjust
the window and positive-window values gradually and test in realistic room
conditions. Always assume automated detection can fail.

## Uninstalling

Download any history you want to retain, stop the App, then uninstall it. Select
the option to remove App data only if you also intend to delete frames, history,
settings, and encrypted credentials permanently.

## Migrating the legacy private database

Advanced users moving from the earlier private deployment can use
`baby-monitor-migrate-legacy`. The command opens the source SQLite database
read-only and prints a dry-run plan unless `--apply` is supplied:

```bash
baby-monitor-migrate-legacy --source /private/path/legacy.sqlite3
baby-monitor-migrate-legacy \
  --source /private/path/legacy.sqlite3 \
  --target /private/path/new-data \
  --apply
```

The target must be private and outside a Git worktree. Stop Baby Monitor before
copying the migrated target into `/data`, retain a backup, start the App, and
verify history and frame counts before deleting the source.
