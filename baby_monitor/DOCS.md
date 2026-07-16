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
`Europe/Madrid`. Use a stable Location ID and a readable Location name when the
same history follows the baby between homes. One baby profile is supported.

### Sleep timeline and manual entries

The main rhythm view has separate Day and Night modes. Day spans the baby's
morning wake-up through bedtime; Night spans bedtime through the following
wake-up, so the circular segments use the real boundaries instead of fixed
clock positions. Solid arcs are recorded sleep, narrow dashed coral arcs are
awake periods, and dashed purple arcs are predictions.

The App calculates plans locally from the profile age and recent history. It
shows remaining predicted naps and bedtime for today, plus the complete plan
for tomorrow. These predictions do not call an AI service and are guidance,
not medical advice.

Use the centre Add action to record a nap, awake period, or night sleep. Start
and end use a calendar plus exact hour/minute picker. Sleep entries can also
store awake pauses, how the baby fell asleep, how the sleep ended, mood on
waking, and a comment. Selecting a recorded segment opens the same fields for
editing and can show the nearest camera frame at the start, midpoint, and end.
Entries created automatically remain editable, but only manually created
entries can be deleted from this screen.

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

The Live button first negotiates WebRTC with a local go2rtc relay and labels the
transport as **WebRTC · low latency**. If that relay is unavailable, the App
shows **MJPEG fallback** instead of presenting a delayed snapshot stream as
low-latency video. The latest stored frame stays visible underneath the video
until the first WebRTC frame is actually playing, avoiding a blank camera panel
during negotiation. Standalone deployments can override the defaults with
`BABY_MONITOR_GO2RTC_URL` (default `http://127.0.0.1:1984`) and
`BABY_MONITOR_GO2RTC_STREAM` (default `baby_monitor_live`). Keep the go2rtc API
private and expose WebRTC media only to trusted clients.

For the fastest joins, keep the selected go2rtc stream preloaded and encode it
with a short H.264 GOP. Hardware decoding is camera-dependent; use a custom
encode-only template when a camera's H.264 stream cannot be decoded by the
host's hardware accelerator.

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
alert using the colour mode each entity actually supports (including XY-only
Hue lights), and restores the previous state after the configured duration.

Do not use safety-critical lighting where an unexpected state change could
create a hazard.

### Notifications (optional)

Select one or more Home Assistant `person.` entities, then configure every
caregiver independently. Each selected person can be enabled or muted, use
Spanish or English, and subscribe to any combination of:

- Crying starts.
- Sleep starts.
- A predicted nap or night sleep is approaching.
- An active sleep is nearing its expected end.
- Sleep ends.
- The camera has stopped producing fresh frames.

Choose a 5, 10, 15, or 20 minute lead time for the two advance alerts. The
expected end is derived from the recent average duration for the current sleep
type; it is guidance rather than a guarantee that the baby will wake then.

Each person needs a Home Assistant Companion App device that exposes a
`notify.mobile_app_*` service. Baby Monitor automatically suggests a service
whose device name matches the person, but Settings keeps the mapping explicit
so a single phone is never silently assigned to a different caregiver.

Notifications are off until a person is selected. Legacy configurations are
migrated conservatively to cry-only alerts. Delivery is deduplicated per
person, event, and sleep/prediction even after the App restarts. Use the test
button after checking the selected mobile service; it sends one real test
notification to every enabled caregiver.

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

### Multiple homes and one history

The active Location ID is stored on every new frame, sleep session, and cry
event. Two Home Assistant OS installations have isolated App data and must not
share a live SQLite database.

To move the complete history while keeping one active copy:

1. At the source, open **Settings → Move or export history** and select
   **Prepare and download ZIP**. The source becomes read-only.
2. At the destination, choose that ZIP, acknowledge replacement, and select
   **Validate and import history**.
3. Check the record counts and images at the destination, then download its
   JSON receipt.
4. Back at the source, upload the receipt and explicitly confirm deletion of
   its history and images. The source keeps its house-specific Settings and is
   marked retired until a newer ZIP is imported there.

If the destination import is not completed, select **Cancel transfer** at the
source. Its history becomes writable again and nothing is deleted.

Import validates the database and every image before it replaces any existing
history. Camera entities, lights, notifications, stream URLs, API keys, and AI
configuration always remain local to the destination installation. This is a
sequential handoff, not simultaneous synchronization. Full design and recovery
details are in [Moving one history between homes](../docs/shared-history.md).

Select **Forever** under Retention if frames must never be removed
automatically. A transfer does not delete images unless the source receipt is
validated and the final destructive confirmation is selected.

## Data and backups

The App stores its database, frames, settings, encryption key, and encrypted
secrets in `/data`. Supervisor includes this directory in App backups. Treat
those backups as private household data.

The history ZIP is suitable for external analysis without running Baby
Monitor. It includes:

- `data/frames.csv`, including labels and the corresponding archive image path;
- `data/sleep_events.csv`;
- `data/cry_events.csv`;
- original images under `images/<location>/<year>/<month>/<day>/`;
- `internal/history.sqlite3` and `manifest.json` for verified, lossless import.

It excludes Settings, API keys, Home Assistant tokens, camera URLs, and the
local encryption key. It still contains sensitive family images and history.

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
  --location-id madrid \
  --timezone Europe/Madrid \
  --apply
```

Use a lowercase Location ID containing letters, numbers, underscores, or
hyphens. The import copies the location onto every migrated frame and event.

The target must be private and outside a Git worktree. The import rebuilds the
legacy sleep timeline from its stored camera observations and retains the
structured visual attributes used by Trends. Existing non-legacy sleep entries
and every existing image file are preserved. Stop Baby Monitor before applying
the import to its live data directory, retain a backup, start the App, and
verify history and frame counts before deleting the source.
