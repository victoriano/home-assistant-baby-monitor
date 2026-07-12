# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability or accidental
exposure of household data.

Use GitHub's private vulnerability reporting for this repository:

1. Open the repository's **Security** tab.
2. Select **Advisories**.
3. Select **Report a vulnerability**.

Include the affected version, deployment mode, reproduction steps, and impact.
Do not attach real camera frames, Home Assistant tokens, API keys, or private
stream URLs. You should receive an acknowledgement within seven days.

## Supported versions

Until the first stable release, only the latest published version receives
security fixes.

## Release integrity

Official container images are published only by
`.github/workflows/release.yaml`. The workflow signs architecture images and
the multi-architecture manifest with keyless Cosign, creates GitHub
build-provenance attestations, and attaches architecture-specific SPDX JSON
SBOMs to the GitHub release. Every external GitHub Action in the repository is
pinned to a full commit SHA.

Verify an image's GitHub provenance before deploying it:

```bash
gh attestation verify \
  oci://ghcr.io/victoriano/home-assistant-baby-monitor:VERSION \
  --repo victoriano/home-assistant-baby-monitor \
  --signer-workflow victoriano/home-assistant-baby-monitor/.github/workflows/release.yaml
```

Use an immutable image digest for high-assurance deployments. A missing or
invalid signature, provenance attestation, or SBOM is a release-integrity issue
and should be reported through the private vulnerability process above.

## Deployment guidance

- Keep Home Assistant and the App updated.
- Do not expose port 8099 directly when using Home Assistant OS; use Ingress.
- Use a unique, high-entropy administrator token in standalone Docker mode.
- Restrict access to `/data` and to backups containing `/data`.
- Prefer a Home Assistant camera entity over embedding camera credentials in a
  URL. If a private stream URL is needed, create a dedicated least-privilege
  camera account.
- Review cloud AI provider data-retention terms before enabling image upload.
- Review the release SBOM and
  [third-party notices](THIRD_PARTY_NOTICES.md) when redistributing the
  container.
