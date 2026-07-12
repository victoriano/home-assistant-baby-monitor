import {
  cloneDefaultSettings,
  settingsToPayload,
  type AppSettings,
  type ConnectionTestResult,
  type CryEvent,
  type DashboardSummary,
  type FrameRecord,
  type HealthStatus,
  type HomeAssistantEntity,
  type ManualSleepInput,
  type PageResult,
  type RetentionEstimate,
  type SecretName,
  type SleepEvent,
  type VisionLabel,
} from './types';

type JsonRecord = Record<string, unknown>;

export class ApiError extends Error {
  readonly status: number;
  readonly details: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : {};
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function asNullableString(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function asBooleanRecord(value: unknown): Record<string, boolean> {
  return Object.fromEntries(
    Object.entries(asRecord(value)).filter((entry): entry is [string, boolean] => typeof entry[1] === 'boolean'),
  );
}

function asStringRecord(value: unknown): Record<string, string> {
  return Object.fromEntries(
    Object.entries(asRecord(value)).filter((entry): entry is [string, string] => typeof entry[1] === 'string'),
  );
}

function pick(record: JsonRecord, ...names: string[]): unknown {
  for (const name of names) {
    if (record[name] !== undefined && record[name] !== null) return record[name];
  }
  return undefined;
}

function unwrapList(value: unknown, keys: string[]): unknown[] {
  if (Array.isArray(value)) return value;
  const record = asRecord(value);
  for (const key of keys) {
    if (Array.isArray(record[key])) return record[key] as unknown[];
  }
  return [];
}

function normalizePage<T>(
  value: unknown,
  keys: string[],
  normalizeItem: (item: unknown) => T,
  requestedLimit: number,
  requestedOffset: number,
): PageResult<T> {
  const root = asRecord(value);
  const items = unwrapList(value, keys).map(normalizeItem);
  const limit = Math.max(1, asNumber(root.limit, requestedLimit));
  const offset = Math.max(0, asNumber(root.offset, requestedOffset));
  const minimumTotal = offset + items.length;
  return {
    items,
    limit,
    offset,
    total: Math.max(minimumTotal, asNumber(root.total, minimumTotal)),
  };
}

export function apiUrl(path: string): string {
  const normalized = path.replace(/^\/+/, '');
  return new URL(normalized, document.baseURI).toString();
}

export function resolveAppUrl(path: string): string {
  if (/^(https?:|data:|blob:)/i.test(path)) return path;
  return apiUrl(path);
}

function errorMessage(body: unknown, fallback: string): string {
  const data = asRecord(body);
  const detail = data.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => asRecord(item))
      .map((item) => asString(item.msg))
      .filter(Boolean);
    if (messages.length) return messages.join(' · ');
  }
  return asString(data.message) || fallback;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(apiUrl(path), {
    credentials: 'same-origin',
    ...init,
    cache: 'no-store',
    headers: {
      Accept: 'application/json',
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      ...init.headers,
    },
  });

  const contentType = response.headers.get('content-type') ?? '';
  const body: unknown = response.status === 204
    ? undefined
    : contentType.includes('application/json')
      ? await response.json()
      : await response.text();

  if (!response.ok) {
    throw new ApiError(errorMessage(body, response.statusText || 'Request failed'), response.status, body);
  }
  return body as T;
}

export function normalizeSettings(value: unknown): AppSettings {
  const root = asRecord(value);
  const data = asRecord(root.settings ?? root);
  const defaults = cloneDefaultSettings();
  const baby = asRecord(data.baby ?? data.profile);
  const homeAssistant = asRecord(data.home_assistant ?? data.homeAssistant);
  const camera = asRecord(data.camera);
  const cry = asRecord(data.cry);
  const lights = asRecord(data.lights);
  const ai = asRecord(data.ai ?? data.vision);
  const retention = asRecord(data.retention);
  const notifications = asRecord(data.notifications);

  const rawCryMode = asString(pick(cry, 'mode'), defaults.cry.mode);
  const cryMode = rawCryMode === 'rtsp_audio' ? 'audio' : rawCryMode;
  const rawProvider = asString(pick(ai, 'provider'), defaults.ai.provider);
  const provider = rawProvider === 'ollama' ? 'local' : rawProvider;
  const rawSensitivity = asString(pick(cry, 'sensitivity'), defaults.cry.sensitivity);
  const retentionMode = asString(pick(retention, 'mode'), defaults.retention.mode);
  const name = asString(pick(baby, 'name', 'babyName', 'baby_name'), defaults.baby.name);
  const explicitConfigured = pick(root, 'configured', 'setup_complete') ?? pick(data, 'configured', 'setup_complete');

  return {
    configured: typeof explicitConfigured === 'boolean'
      ? explicitConfigured
      : Boolean(name && name.toLowerCase() !== 'baby'),
    schemaVersion: 1,
    baby: {
      name,
      birthDate: asNullableString(pick(baby, 'birthDate', 'birth_date')),
      timezone: asString(pick(baby, 'timezone'), defaults.baby.timezone),
    },
    homeAssistant: {
      mode: ['auto', 'supervisor', 'standalone'].includes(asString(homeAssistant.mode))
        ? asString(homeAssistant.mode) as AppSettings['homeAssistant']['mode']
        : defaults.homeAssistant.mode,
      baseUrl: asNullableString(pick(homeAssistant, 'baseUrl', 'base_url')),
      accessTokenConfigured: asBoolean(pick(homeAssistant, 'accessTokenConfigured', 'access_token_configured')),
    },
    camera: {
      enabled: asBoolean(camera.enabled, Boolean(pick(camera, 'entityId', 'entity_id'))),
      entityId: asNullableString(pick(camera, 'entityId', 'entity_id')),
      captureIntervalSeconds: asNumber(
        pick(camera, 'captureIntervalSeconds', 'capture_interval_seconds'),
        asNumber(pick(ai, 'captureIntervalMinutes', 'capture_interval_minutes'), defaults.camera.captureIntervalSeconds / 60) * 60,
      ),
      streamUrlConfigured: asBoolean(pick(camera, 'streamUrlConfigured', 'stream_url_configured')),
    },
    cry: {
      mode: cryMode === 'binary_sensor' || cryMode === 'audio' ? cryMode : 'disabled',
      entityId: asNullableString(pick(cry, 'entityId', 'entity_id', 'sensorEntityId', 'sensor_entity_id')),
      positiveWindows: asNumber(pick(cry, 'positiveWindows', 'positive_windows'), defaults.cry.positiveWindows),
      windowSeconds: asNumber(pick(cry, 'windowSeconds', 'window_seconds'), defaults.cry.windowSeconds),
      clearAfterSeconds: asNumber(
        pick(cry, 'clearAfterSeconds', 'clear_after_seconds'),
        defaults.cry.clearAfterSeconds,
      ),
      sensitivity: rawSensitivity === 'low' || rawSensitivity === 'high' ? rawSensitivity : 'balanced',
      audioStreamUrlConfigured: asBoolean(
        pick(cry, 'audioStreamUrlConfigured', 'audio_stream_url_configured', 'streamUrlConfigured', 'stream_url_configured'),
      ),
    },
    lights: {
      entityIds: asStringArray(pick(lights, 'entityIds', 'entity_ids')),
      durationSeconds: asNumber(pick(lights, 'durationSeconds', 'duration_seconds'), defaults.lights.durationSeconds),
      brightnessPercent: asNumber(
        pick(lights, 'brightnessPercent', 'brightness_percent'),
        defaults.lights.brightnessPercent,
      ),
      colorRgb: (() => {
        const color = pick(lights, 'colorRgb', 'color_rgb');
        return Array.isArray(color) && color.length === 3
          ? color.map((channel) => asNumber(channel)) as [number, number, number]
          : defaults.lights.colorRgb;
      })(),
      restorePreviousState: true,
    },
    ai: {
      provider: provider === 'gemini' || provider === 'openai' || provider === 'local' ? provider : 'disabled',
      model: asNullableString(ai.model),
      apiKeyConfigured: asBoolean(pick(ai, 'apiKeyConfigured', 'api_key_configured')),
      baseUrl: asNullableString(pick(ai, 'baseUrl', 'base_url')),
      cloudImageConsent: asBoolean(pick(ai, 'cloudImageConsent', 'cloud_image_consent')),
      detail: ['low', 'high', 'auto'].includes(asString(ai.detail))
        ? asString(ai.detail) as AppSettings['ai']['detail']
        : defaults.ai.detail,
    },
    retention: {
      mode: retentionMode === 'days' ? 'days' : 'forever',
      days: retentionMode === 'days' ? asNumber(retention.days, 30) : null,
    },
    notifications: {
      service: asNullableString(pick(notifications, 'service')),
      targets: asStringArray(pick(notifications, 'targets', 'entityIds', 'entity_ids')),
    },
  };
}

function normalizeLabel(value: unknown, descriptionFallback = ''): VisionLabel | null {
  if (!value) return null;
  if (typeof value === 'string') {
    return {
      babyPresent: true,
      state: 'uncertain',
      confidence: 0,
      description: value,
      tags: [],
    };
  }
  const data = asRecord(value);
  const state = asString(data.state);
  return {
    babyPresent: asBoolean(pick(data, 'babyPresent', 'baby_present')),
    state: state === 'awake' || state === 'asleep' ? state : 'uncertain',
    confidence: asNumber(data.confidence),
    description: asString(data.description, descriptionFallback),
    tags: asStringArray(data.tags),
  };
}

export function normalizeFrame(value: unknown): FrameRecord {
  const data = asRecord(value);
  const id = asString(data.id);
  const rawImageUrl = asString(pick(data, 'imageUrl', 'image_url'));
  return {
    id,
    capturedAt: asString(pick(data, 'capturedAt', 'captured_at')),
    cameraEntityId: asNullableString(pick(data, 'cameraEntityId', 'camera_entity_id')),
    imageUrl: rawImageUrl ? resolveAppUrl(rawImageUrl) : id ? apiUrl(`api/v1/frames/${encodeURIComponent(id)}/image`) : '',
    imageAvailable: asBoolean(pick(data, 'imageAvailable', 'image_available'), true),
    mimeType: asString(pick(data, 'mimeType', 'mime_type'), 'image/jpeg'),
    sizeBytes: asNumber(pick(data, 'sizeBytes', 'size_bytes')),
    label: normalizeLabel(data.vision ?? data.label, asString(data.label)),
    provider: asNullableString(data.provider),
    model: asNullableString(data.model),
  };
}

export function normalizeSleep(value: unknown): SleepEvent {
  const data = asRecord(value);
  const kind = asString(data.kind);
  const source = asString(data.source);
  return {
    id: asString(data.id),
    startedAt: asString(pick(data, 'startedAt', 'started_at')),
    endedAt: asNullableString(pick(data, 'endedAt', 'ended_at')),
    kind: kind === 'nap' || kind === 'night' ? kind : 'unknown',
    source: source === 'vision' || source === 'import' || source === 'automatic' ? source : 'manual',
    notes: asNullableString(data.notes),
  };
}

export function normalizeCry(value: unknown): CryEvent {
  const data = asRecord(value);
  const source = asString(data.source);
  return {
    id: asString(data.id),
    detectedAt: asString(pick(data, 'detectedAt', 'detected_at', 'startedAt', 'started_at')),
    endedAt: asNullableString(pick(data, 'endedAt', 'ended_at')),
    source: source === 'rtsp_audio' ? 'audio' : source === 'binary_sensor' || source === 'audio' || source === 'import' ? source : 'manual',
    confidence: data.confidence == null ? null : asNumber(data.confidence),
  };
}

export function normalizeSummary(value: unknown): DashboardSummary {
  const root = asRecord(value);
  const data = asRecord(root.summary ?? root);
  const state = asString(pick(data, 'state', 'sleep_state'));
  const latestFrame = pick(data, 'latestFrame', 'latest_frame');
  const currentSleep = pick(data, 'currentSleep', 'current_sleep');
  const prediction = asRecord(data.prediction);
  const nextSleepAt = pick(prediction, 'nextSleepAt', 'next_sleep_at') ?? pick(data, 'nextSleepAt', 'next_sleep_at');
  return {
    state: state === 'sleeping' || state === 'awake' ? state : 'unknown',
    stateSince: asNullableString(pick(data, 'stateSince', 'state_since')),
    currentSleep: currentSleep ? normalizeSleep(currentSleep) : null,
    prediction: {
      nextSleepAt: asNullableString(nextSleepAt),
      windowStart: asNullableString(pick(prediction, 'windowStart', 'window_start')),
      windowEnd: asNullableString(pick(prediction, 'windowEnd', 'window_end')),
      confidence: prediction.confidence == null ? null : asNumber(prediction.confidence),
      reason: asNullableString(prediction.reason),
    },
    sleepTodayMinutes: asNumber(pick(data, 'sleepTodayMinutes', 'sleep_today_minutes')),
    lastCryAt: asNullableString(pick(data, 'lastCryAt', 'last_cry_at')),
    cryActive: asBoolean(pick(data, 'cryActive', 'cry_active')),
    latestFrame: latestFrame ? normalizeFrame(latestFrame) : null,
    recentSleep: unwrapList(pick(data, 'recentSleep', 'recent_sleep'), ['items']).map(normalizeSleep),
    recentCry: unwrapList(pick(data, 'recentCry', 'recent_cry'), ['items']).map(normalizeCry),
    updatedAt: asNullableString(pick(data, 'updatedAt', 'updated_at')),
  };
}

export const api = {
  async getSettings(): Promise<AppSettings> {
    return normalizeSettings(await request<unknown>('api/v1/settings'));
  },

  async getHealth(): Promise<HealthStatus> {
    const data = asRecord(await request<unknown>('api/v1/health'));
    const background = asRecord(data.background);
    return {
      ok: asBoolean(data.ok),
      database: asBoolean(data.database),
      runtime: asString(data.runtime),
      background: {
        running: asBoolean(background.running),
        workers: asBooleanRecord(background.workers),
        errors: asStringRecord(background.errors),
      },
    };
  },

  async saveSettings(settings: AppSettings, clear: SecretName[] = []): Promise<AppSettings> {
    const result = await request<unknown>('api/v1/settings', {
      method: 'PUT',
      body: JSON.stringify(settingsToPayload(settings, clear)),
    });
    return normalizeSettings(result);
  },

  async getEntities(domain: 'camera' | 'binary_sensor' | 'light' | 'notify'): Promise<HomeAssistantEntity[]> {
    const result = await request<unknown>(`api/v1/home-assistant/entities?domain=${encodeURIComponent(domain)}`);
    return unwrapList(result, ['items', 'entities']).map((value) => {
      const data = asRecord(value);
      const entityId = asString(pick(data, 'entityId', 'entity_id'));
      return {
        entityId,
        name: asString(data.name, entityId),
        state: asString(data.state),
        available: asString(data.state) !== 'unavailable',
        attributes: asRecord(data.attributes),
      };
    }).filter((entity) => entity.entityId.length > 0);
  },

  async getSummary(): Promise<DashboardSummary> {
    return normalizeSummary(await request<unknown>('api/v1/summary'));
  },

  async getFrames(limit = 24, offset = 0): Promise<PageResult<FrameRecord>> {
    const result = await request<unknown>(`api/v1/frames?limit=${limit}&offset=${offset}`);
    return normalizePage(result, ['items', 'frames'], normalizeFrame, limit, offset);
  },

  async getSleep(limit = 50, offset = 0): Promise<PageResult<SleepEvent>> {
    const result = await request<unknown>(`api/v1/sleep?limit=${limit}&offset=${offset}`);
    return normalizePage(result, ['items', 'events'], normalizeSleep, limit, offset);
  },

  async getCryEvents(limit = 50, offset = 0): Promise<PageResult<CryEvent>> {
    const result = await request<unknown>(`api/v1/cry-events?limit=${limit}&offset=${offset}`);
    return normalizePage(result, ['items', 'events'], normalizeCry, limit, offset);
  },

  async addManualSleep(input: ManualSleepInput): Promise<SleepEvent> {
    return normalizeSleep(await request<unknown>('api/v1/sleep', {
      method: 'POST',
      body: JSON.stringify({
        started_at: input.startedAt,
        ended_at: input.endedAt,
        kind: input.kind,
        notes: input.notes || null,
        source: 'manual',
      }),
    }));
  },

  async startSleep(kind: 'nap' | 'night' = 'nap'): Promise<SleepEvent> {
    return normalizeSleep(await request<unknown>('api/v1/sleep/start', {
      method: 'POST',
      body: JSON.stringify({ kind, started_at: new Date().toISOString() }),
    }));
  },

  async stopSleep(): Promise<SleepEvent> {
    return normalizeSleep(await request<unknown>('api/v1/sleep/stop', {
      method: 'POST',
      body: JSON.stringify({ ended_at: new Date().toISOString() }),
    }));
  },

  async refreshSnapshot(): Promise<FrameRecord> {
    return normalizeFrame(await request<unknown>('api/v1/camera/snapshot', { method: 'POST' }));
  },

  async labelFrame(frameId: string): Promise<FrameRecord> {
    return normalizeFrame(await request<unknown>(`api/v1/frames/${encodeURIComponent(frameId)}/label`, {
      method: 'POST',
    }));
  },

  liveCameraUrl(): string {
    return apiUrl('api/v1/camera/live');
  },

  async testSettings(kind: 'home_assistant' | 'camera' | 'cry' | 'lights' | 'notifications' | 'vision', settings: AppSettings): Promise<ConnectionTestResult> {
    const result = asRecord(await request<unknown>(`api/v1/settings/test/${kind}`, {
      method: 'POST',
      body: JSON.stringify(settingsToPayload(settings)),
    }));
    return {
      ok: asBoolean(result.ok),
      message: asString(result.message) || undefined,
    };
  },

  async estimateRetention(days: number): Promise<RetentionEstimate> {
    const result = asRecord(await request<unknown>(`api/v1/retention/estimate?days=${days}`));
    return {
      frames: asNumber(result.frames),
      bytes: asNumber(result.bytes),
    };
  },
};

export const apiTesting = {
  normalizeSettings,
  normalizeSummary,
  normalizeFrame,
};
