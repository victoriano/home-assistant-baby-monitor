# Contributing

Contributions are welcome. Please keep changes generic: no household-specific
entity IDs, names, camera addresses, frames, API keys, or access tokens belong
in this repository, fixtures, logs, screenshots, or pull requests.

## Development workflow

1. Fork the repository and create a focused branch.
2. Install Python and frontend dependencies as described in `README.md`.
3. Add or update tests for behavior changes.
4. Run `pytest`, `ruff check .`, `npm test`, and `npm run build`.
5. Open a pull request explaining the user-facing behavior, privacy impact, and
   how it was verified.

## Design constraints

- The app must remain usable without a camera and without AI.
- Cloud image upload must remain opt-in and require explicit consent.
- Secrets must be write-only at the API boundary and encrypted at rest.
- Do not write frames under Home Assistant's public `/local` tree.
- Home Assistant App mode must remain admin-only and Ingress-only.
- Standalone mode must remain authenticated.
- Do not add host networking, Docker socket access, privileged mode, or broad
  Home Assistant filesystem mounts.

By contributing, you agree that your contribution is licensed under MIT.

## Maintainer release checklist

1. Update the version in `pyproject.toml`, `baby_monitor/config.yaml`,
   `docker-compose.yaml`, and `baby_monitor/CHANGELOG.md`.
2. Run the full CI suite and a local standalone container smoke test.
3. Create a GitHub release whose tag exactly matches the App version (for
   example, `0.1.0`, without a `v` prefix).
4. Confirm the workflow publishes and signs both architecture images and the
   generic multi-architecture manifest.
5. Confirm `ghcr.io/victoriano/home-assistant-baby-monitor` is public. GitHub
   Container Registry visibility must be checked after the first publication.
6. Add the repository to a clean Home Assistant instance and perform an install,
   start, Ingress, configuration, and update smoke test.
