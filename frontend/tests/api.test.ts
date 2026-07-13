import { describe, expect, it, vi } from 'vitest';

import { api, apiTesting, apiUrl } from '../src/api';
import { cloneDefaultSettings, isValidHttpBaseUrl, settingsToPayload } from '../src/types';

describe('settings contract', () => {
  it('normalizes the backend snake_case response without exposing secrets', () => {
    const settings = apiTesting.normalizeSettings({
      setup_complete: true,
      schema_version: 1,
      baby: { name: 'Luna', birth_date: '2026-01-02', timezone: 'Europe/Madrid' },
      home_assistant: { mode: 'supervisor', access_token_configured: true },
      camera: { enabled: true, entity_id: 'camera.nursery', capture_interval_seconds: 120, stream_url_configured: false },
      cry: { mode: 'audio', positive_windows: 1, window_seconds: 0.5, clear_after_seconds: 8, sensitivity: 'high', audio_stream_url_configured: true },
      lights: { entity_ids: ['light.hall'], duration_seconds: 30, brightness_percent: 25, color_rgb: [255, 100, 40] },
      ai: { provider: 'openai', model: 'gpt-4.1-mini', api_key_configured: true, cloud_image_consent: true, detail: 'low' },
      notifications: { service: 'notify.mobile_app', targets: ['parent'] },
      retention: { mode: 'days', days: 14 },
      secrets: { ai_api_key: 'must-never-appear' },
    });

    expect(settings.configured).toBe(true);
    expect(settings.baby.name).toBe('Luna');
    expect(settings.camera.entityId).toBe('camera.nursery');
    expect(settings.cry.mode).toBe('audio');
    expect(settings.cry.sensitivity).toBe('high');
    expect(settings.ai.provider).toBe('openai');
    expect(settings.ai.apiKeyConfigured).toBe(true);
    expect(settings.ai.apiKey).toBeUndefined();
    expect(JSON.stringify(settings)).not.toContain('must-never-appear');
  });

  it('writes only new secret values and explicit removals', () => {
    const settings = cloneDefaultSettings();
    settings.baby.name = 'Luna';
    settings.ai.provider = 'openai';
    settings.ai.model = 'gpt-4.1-mini';
    settings.ai.cloudImageConsent = true;
    settings.ai.apiKeyConfigured = true;
    settings.ai.apiKey = 'new-key';
    settings.cry.sensitivity = 'low';

    const payload = settingsToPayload(settings, ['camera_stream_url']);

    expect(payload.secrets.ai_api_key).toBe('new-key');
    expect(payload.secrets.clear).toEqual(['camera_stream_url']);
    expect(payload.ai).not.toHaveProperty('api_key');
    expect(payload.ai).not.toHaveProperty('api_key_configured');
    expect(payload.cry.sensitivity).toBe('low');
  });

  it('maps UI-friendly audio and local provider names to backend aliases', () => {
    const settings = cloneDefaultSettings();
    settings.baby.name = 'Luna';
    settings.cry.mode = 'audio';
    settings.cry.audioStreamUrl = 'rtsp://camera/audio';
    settings.ai.provider = 'local';
    settings.ai.model = 'qwen2.5vl:3b';
    settings.ai.baseUrl = 'http://ollama.local:11434/v1';

    const payload = settingsToPayload(settings);

    expect(payload.cry.mode).toBe('rtsp_audio');
    expect(payload.ai.provider).toBe('ollama');
    expect(payload.ai.base_url).toBe('http://ollama.local:11434/v1');
  });

  it('keeps standalone credentials write-only and never sends a cloud AI base URL', () => {
    const settings = cloneDefaultSettings();
    settings.homeAssistant.mode = 'standalone';
    settings.homeAssistant.baseUrl = 'http://homeassistant.local:8123/';
    settings.homeAssistant.accessToken = 'ha-token';
    settings.ai.provider = 'openai';
    settings.ai.baseUrl = 'https://must-not-be-used.example/v1';
    settings.ai.apiKey = 'ai-token';

    const payload = settingsToPayload(settings);

    expect(payload.home_assistant).toEqual({ mode: 'standalone', base_url: 'http://homeassistant.local:8123' });
    expect(payload.secrets.home_assistant_access_token).toBe('ha-token');
    expect(payload.ai.base_url).toBeNull();
    expect(payload.ai).not.toHaveProperty('api_key');
  });

  it('accepts only absolute HTTP(S) base URLs without embedded credentials', () => {
    expect(isValidHttpBaseUrl('http://homeassistant.local:8123')).toBe(true);
    expect(isValidHttpBaseUrl('https://ha.example.test')).toBe(true);
    expect(isValidHttpBaseUrl('/relative')).toBe(false);
    expect(isValidHttpBaseUrl('rtsp://camera.local')).toBe(false);
    expect(isValidHttpBaseUrl('https://user:pass@ha.example.test')).toBe(false);
  });
});

describe('ingress-safe URLs', () => {
  it('keeps API calls under the current document base', () => {
    document.head.innerHTML = '<base href="http://localhost:8123/api/hassio_ingress/demo/">';
    expect(apiUrl('/api/v1/settings')).toBe('http://localhost:8123/api/hassio_ingress/demo/api/v1/settings');
  });

  it('tests the Home Assistant connection without using the browser cache', async () => {
    const response = new Response(JSON.stringify({ ok: true, message: 'Connected' }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response);
    try {
      const result = await api.testSettings('home_assistant', cloneDefaultSettings());
      expect(result).toEqual({ ok: true, message: 'Connected' });
      expect(fetchMock).toHaveBeenCalledOnce();
      const [url, init] = fetchMock.mock.calls[0];
      expect(String(url)).toContain('/api/v1/settings/test/home_assistant');
      expect(init?.cache).toBe('no-store');
    } finally {
      fetchMock.mockRestore();
    }
  });

  it('normalizes background health errors for the visible warning banner', async () => {
    const response = new Response(JSON.stringify({
      ok: true,
      database: true,
      runtime: 'standalone',
      background: {
        running: true,
        workers: { capture: true, cry: true },
        errors: { cry: 'HomeAssistantError' },
      },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response);
    try {
      const health = await api.getHealth();
      expect(health.runtime).toBe('standalone');
      expect(health.background.errors).toEqual({ cry: 'HomeAssistantError' });
      expect(fetchMock.mock.calls[0][1]?.cache).toBe('no-store');
    } finally {
      fetchMock.mockRestore();
    }
  });
});

describe('portable history transfer contract', () => {
  it('normalizes a prepared export and its integrity metadata', () => {
    const status = apiTesting.normalizeTransferStatus({
      status: 'pending',
      writable: false,
      datasetId: 'dataset-1',
      generation: 2,
      outgoing: {
        archiveId: 'archive-1',
        filename: 'baby-monitor-history.zip',
        manifestSha256: 'abc123',
        bytes: 1234,
        counts: { frames: 10, storedImages: 9, sleepEvents: 3, cryEvents: 2 },
        downloadUrl: 'api/v1/history-transfer/exports/archive-1',
      },
    });

    expect(status.status).toBe('pending');
    expect(status.writable).toBe(false);
    expect(status.outgoing?.counts).toEqual({ frames: 10, storedImages: 9, sleepEvents: 3, cryEvents: 2 });
    expect(status.outgoing?.downloadUrl).toContain('history-transfer/exports/archive-1');
  });

  it('uploads the ZIP as a binary body under the current Ingress base', async () => {
    document.head.innerHTML = '<base href="http://localhost:8123/api/hassio_ingress/demo/">';
    const response = new Response(JSON.stringify({
      ok: true,
      idempotent: false,
      counts: { frames: 1, storedImages: 1, sleepEvents: 0, cryEvents: 0 },
      receipt: {
        datasetId: 'dataset-1', generation: 1, manifestSha256: 'hash',
        destinationInstallationId: 'installation-1', importedAt: '2026-07-13T00:00:00Z',
        counts: { frames: 1, storedImages: 1, sleepEvents: 0, cryEvents: 0 },
      },
      status: { status: 'active', writable: true, datasetId: 'dataset-1', generation: 1 },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response);
    const file = new File(['zip-bytes'], 'history.zip', { type: 'application/zip' });
    try {
      const result = await api.importHistory(file, true);
      expect(result.receipt.datasetId).toBe('dataset-1');
      const [url, init] = fetchMock.mock.calls[0];
      expect(String(url)).toContain('/api/hassio_ingress/demo/api/v1/history-transfer/imports?replace=true');
      expect(init?.body).toBe(file);
      expect(new Headers(init?.headers).get('Content-Type')).toBe('application/zip');
    } finally {
      fetchMock.mockRestore();
    }
  });

  it('uploads the destination receipt with an explicit source-deletion flag', async () => {
    document.head.innerHTML = '<base href="http://localhost:8123/api/hassio_ingress/demo/">';
    const response = new Response(JSON.stringify({
      ok: true,
      deleted: true,
      status: { status: 'retired', writable: false, datasetId: 'dataset-1', generation: 1 },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response);
    const receipt = new File(['{"format":"baby-monitor-import-receipt"}'], 'receipt.json', {
      type: 'application/json',
    });
    try {
      const status = await api.finalizeHistoryExport(receipt, true);
      expect(status.status).toBe('retired');
      const [url, init] = fetchMock.mock.calls[0];
      expect(String(url)).toContain('/api/v1/history-transfer/finalize?delete=true');
      expect(init?.body).toBe(receipt);
      expect(new Headers(init?.headers).get('Content-Type')).toBe('application/json');
    } finally {
      fetchMock.mockRestore();
    }
  });
});

describe('dashboard normalization', () => {
  it('accepts prediction and current sleep in snake_case', () => {
    const summary = apiTesting.normalizeSummary({
      sleep_state: 'sleeping',
      current_sleep: {
        id: 'sleep-1', started_at: '2026-07-11T00:00:00Z', ended_at: null, kind: 'night', source: 'manual', notes: null,
      },
      prediction: {
        next_sleep_at: '2026-07-11T12:00:00Z', window_start: '2026-07-11T11:45:00Z', window_end: '2026-07-11T12:15:00Z', confidence: 0.8,
      },
      sleep_today_minutes: 420,
      cry_active: false,
    });

    expect(summary.state).toBe('sleeping');
    expect(summary.currentSleep?.kind).toBe('night');
    expect(summary.prediction.confidence).toBe(0.8);
    expect(summary.sleepTodayMinutes).toBe(420);
  });
});

describe('paginated history contract', () => {
  it('preserves server pagination metadata and requests the selected offset', async () => {
    const response = new Response(JSON.stringify({
      items: [{
        id: 'sleep-31',
        started_at: '2026-07-10T20:00:00Z',
        ended_at: '2026-07-10T21:00:00Z',
        kind: 'nap',
        source: 'manual',
        notes: null,
      }],
      limit: 30,
      offset: 30,
      total: 91,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response);
    try {
      const page = await api.getSleep(30, 30);
      expect(page).toMatchObject({ limit: 30, offset: 30, total: 91 });
      expect(page.items).toHaveLength(1);
      expect(page.items[0].id).toBe('sleep-31');
      expect(String(fetchMock.mock.calls[0][0])).toContain('/api/v1/sleep?limit=30&offset=30');
    } finally {
      fetchMock.mockRestore();
    }
  });

  it('uses the notifications settings-test endpoint without serializing stored secrets', async () => {
    const response = new Response(JSON.stringify({ ok: true, message: 'Notification sent' }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response);
    const settings = cloneDefaultSettings();
    settings.notifications.service = 'notify.mobile_app_parent';
    settings.ai.apiKeyConfigured = true;
    settings.homeAssistant.accessTokenConfigured = true;
    try {
      await api.testSettings('notifications', settings);
      const [url, init] = fetchMock.mock.calls[0];
      expect(String(url)).toContain('/api/v1/settings/test/notifications');
      expect(String(init?.body)).not.toContain('api_key_configured');
      expect(String(init?.body)).not.toContain('access_token_configured');
    } finally {
      fetchMock.mockRestore();
    }
  });
});
