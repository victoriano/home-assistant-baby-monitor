# Baby Monitor

Local-first sleep tracking and alerts for Home Assistant.

- Select your own camera and one or more lights.
- Trigger light alerts from a cry `binary_sensor` or an optional audio stream.
- Keep manual sleep history and predictions.
- Export and move verified history, CSV tables, and date-organized images from
  Settings without transferring house-specific credentials.
- Optionally label private camera frames with Gemini, OpenAI, or a local
  OpenAI-compatible service.
- Configure everything from the admin-only Ingress interface.

Camera, cry detection, and AI are optional. Images are never written into Home
Assistant's public `/local` directory.

> This App is not a medical device and does not replace adult supervision.
> Automated detection may miss events or produce false alarms.
