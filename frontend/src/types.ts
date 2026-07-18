export type Language = 'en' | 'es';
export type AppPage = 'sleep' | 'data' | 'camera' | 'settings';
export type CryMode = 'disabled' | 'binary_sensor' | 'audio';
export type CrySensitivity = 'low' | 'balanced' | 'high';
export type VisionProvider = 'disabled' | 'gemini' | 'openai' | 'local';
export type RetentionMode = 'forever' | 'days';
export type NotificationEvent =
  | 'cry_started'
  | 'sleep_started'
  | 'sleep_predicted_soon'
  | 'sleep_ending_soon'
  | 'sleep_ended'
  | 'camera_offline';
export type SleepKind = 'nap' | 'night' | 'awake' | 'unknown';
export type SleepState = 'sleeping' | 'awake' | 'unknown';
export type SecretName =
  | 'home_assistant_access_token'
  | 'camera_stream_url'
  | 'cry_audio_stream_url'
  | 'ai_api_key';

export interface BabySettings {
  name: string;
  birthDate: string | null;
  timezone: string;
  locationId: string;
  locationName: string;
}

export interface HomeAssistantSettings {
  mode: 'auto' | 'supervisor' | 'standalone';
  baseUrl: string | null;
  accessTokenConfigured: boolean;
  accessToken?: string;
}

export interface CameraSettings {
  enabled: boolean;
  entityId: string | null;
  captureIntervalSeconds: number;
  streamUrlConfigured: boolean;
  streamUrl?: string;
}

export interface CrySettings {
  mode: CryMode;
  entityId: string | null;
  positiveWindows: number;
  windowSeconds: number;
  clearAfterSeconds: number;
  sensitivity: CrySensitivity;
  audioStreamUrlConfigured: boolean;
  audioStreamUrl?: string;
}

export interface LightSettings {
  entityIds: string[];
  durationSeconds: number;
  brightnessPercent: number;
  colorRgb: [number, number, number];
  restorePreviousState: true;
}

export interface VisionSettings {
  provider: VisionProvider;
  model: string | null;
  apiKeyConfigured: boolean;
  apiKey?: string;
  baseUrl: string | null;
  cloudImageConsent: boolean;
  detail: 'low' | 'high' | 'auto';
}

export interface RetentionSettings {
  mode: RetentionMode;
  days: number | null;
}

export interface NotificationSettings {
  recipients: NotificationRecipient[];
  leadMinutes: number;
}

export interface NotificationRecipient {
  personEntityId: string | null;
  name: string;
  notifyService: string;
  targets: string[];
  enabled: boolean;
  language: Language;
  events: NotificationEvent[];
}

export interface AppSettings {
  configured: boolean;
  schemaVersion: 1;
  baby: BabySettings;
  homeAssistant: HomeAssistantSettings;
  camera: CameraSettings;
  cry: CrySettings;
  lights: LightSettings;
  ai: VisionSettings;
  retention: RetentionSettings;
  notifications: NotificationSettings;
}

export interface SettingsPayload {
  schema_version: 1;
  baby: {
    name: string;
    birth_date: string | null;
    timezone: string;
    location_id: string;
    location_name: string;
  };
  home_assistant: {
    mode: HomeAssistantSettings['mode'];
    base_url: string | null;
  };
  camera: {
    enabled: boolean;
    entity_id: string | null;
    capture_interval_seconds: number;
  };
  cry: {
    mode: 'disabled' | 'binary_sensor' | 'rtsp_audio';
    entity_id: string | null;
    positive_windows: number;
    window_seconds: number;
    clear_after_seconds: number;
    sensitivity: CrySensitivity;
  };
  lights: {
    entity_ids: string[];
    duration_seconds: number;
    brightness_percent: number;
    color_rgb: [number, number, number];
    restore_previous_state: true;
  };
  ai: {
    provider: 'disabled' | 'gemini' | 'openai' | 'ollama';
    model: string | null;
    base_url: string | null;
    cloud_image_consent: boolean;
    detail: VisionSettings['detail'];
  };
  notifications: {
    recipients: Array<{
      person_entity_id: string | null;
      name: string;
      notify_service: string;
      targets: string[];
      enabled: boolean;
      language: Language;
      events: NotificationEvent[];
    }>;
    lead_minutes: number;
  };
  retention: {
    mode: RetentionMode;
    days: number | null;
  };
  secrets: {
    home_assistant_access_token?: string;
    camera_stream_url?: string;
    cry_audio_stream_url?: string;
    ai_api_key?: string;
    clear: SecretName[];
  };
}

export interface HomeAssistantEntity {
  entityId: string;
  name: string;
  state?: string;
  available: boolean;
  attributes: Record<string, unknown>;
}

export interface VisionLabel {
  babyPresent: boolean;
  state: 'awake' | 'asleep' | 'uncertain';
  confidence: number;
  description: string;
  tags: string[];
  inCrib: boolean | null;
  sleepSurface: 'crib' | 'family_bed' | 'other' | 'unknown';
  faceVisible: 'yes' | 'no' | 'unknown';
  headSide: 'left' | 'right' | 'back' | 'face_down' | 'unknown';
  bodyPosition: string;
  clothingItems: string[];
  pacifier: 'yes' | 'no' | 'unknown';
  mouthOpen: 'yes' | 'no' | 'unknown';
}

export interface VisionStatisticSegment { key: string; label: string; minutes: number; percent: number; color: string }
export interface VisionStatisticMetric { total_minutes: number; positive_minutes?: number; negative_minutes?: number; segments: VisionStatisticSegment[] }
export interface VisionStatistics {
  range: { start: string; end: string };
  sample_count: number;
  visible_sample_count: number;
  observed_minutes: number;
  visible_minutes: number;
  metrics: { pacifier: VisionStatisticMetric; mouth_open: VisionStatisticMetric; head_side: VisionStatisticMetric; clothing: VisionStatisticMetric };
  daily: Array<{ date: string; sample_count: number; visible_sample_count: number; observed_minutes: number; visible_minutes: number; pacifier_minutes: number; mouth_open_minutes: number }>;
}

export interface FrameRecord {
  id: string;
  capturedAt: string;
  cameraEntityId: string | null;
  locationId: string;
  imageUrl: string;
  imageAvailable: boolean;
  mimeType: string;
  sizeBytes: number;
  label: VisionLabel | null;
  provider: string | null;
  model: string | null;
}

export interface SleepEvent {
  id: string;
  startedAt: string;
  endedAt: string | null;
  kind: SleepKind;
  source: 'manual' | 'vision' | 'automatic' | 'import';
  notes: string | null;
  details?: SleepEventDetails;
  locationId: string;
}

export interface SleepPause {
  startedAt: string;
  endedAt: string;
}

export interface SleepEventDetails {
  tags: string[];
  pauses: SleepPause[];
}

export interface CryEvent {
  id: string;
  detectedAt: string;
  endedAt: string | null;
  source: 'binary_sensor' | 'audio' | 'manual' | 'import';
  confidence: number | null;
  locationId: string;
}

export interface PageResult<T> {
  items: T[];
  limit: number;
  offset: number;
  total: number;
}

export interface SleepPrediction {
  nextSleepAt: string | null;
  windowStart: string | null;
  windowEnd: string | null;
  confidence: number | null;
  reason: string | null;
}

export interface SleepPredictionTarget {
  kind: 'nap' | 'night';
  label: string;
  recommendedStart: string;
  windowStart: string;
  windowEnd: string;
  durationMinutes: number;
  confidence: number;
  explanation: string;
  calculation?: PredictionCalculation;
}

export interface PredictionCalculation {
  method: 'wake_window' | 'bedtime_pattern';
  anchorAt: string | null;
  anchorType: 'last_observed_wake' | 'typical_morning_wake' | 'previous_predicted_nap_end' | 'recent_bedtime_median' | 'age_guidance';
  baseRecommendedStart: string;
  adjustmentMinutes: number;
  adjustmentReason: 'past_window' | null;
  wakeWindowMinutes: number | null;
  startSampleCount: number;
  durationSampleCount: number;
  plannedNapNumber?: number;
  morningWakeSampleCount?: number;
  expectedWakeAt?: string | null;
  durationSource?: 'bedtime_to_morning_wake' | 'recent_night_duration';
}

export interface PredictionWakeWindowSample {
  previousSleepId: string;
  previousSleepKind: 'nap' | 'night';
  previousSleepEndedAt: string;
  nextSleepId: string;
  nextSleepKind: 'nap' | 'night';
  nextSleepStartedAt: string;
  minutes: number;
}

export interface PredictionDurationSample {
  eventId?: string;
  nightDate?: string;
  startedAt: string;
  endedAt: string;
  minutes: number;
  source?: string;
}

export interface PredictionClockSample {
  date: string;
  at: string;
  minuteOfDay: number;
}

export interface PredictionNumericEvidence<TSample> {
  count: number;
  medianMinutes: number | null;
  minMinutes: number | null;
  maxMinutes: number | null;
  valuesMinutes: number[];
  finalMinutes: number;
  samples: TSample[];
}

export interface PredictionModelDetails {
  generatedAt: string;
  lookbackClosedSleepCount: number;
  baseline: {
    ageBand: string;
    birthDateKnown: boolean;
    wakeWindowMinutes: number;
    expectedNaps: number;
  };
  wakeWindows: PredictionNumericEvidence<PredictionWakeWindowSample> & {
    medianAbsoluteDeviationMinutes: number | null;
    historyWeight: number;
  };
  napDurations: PredictionNumericEvidence<PredictionDurationSample>;
  bedtimes: {
    count: number;
    medianMinuteOfDay: number;
    usedFallback: boolean;
    samples: PredictionClockSample[];
  };
  morningWakes: {
    count: number;
    medianMinuteOfDay: number;
    usedFallback: boolean;
    samples: PredictionClockSample[];
  };
  nightDurations: PredictionNumericEvidence<PredictionDurationSample>;
  confidence: {
    value: number;
    sampleCount: number;
    rule: 'recent_wake_samples' | 'age_guidance_fallback';
  };
}

export interface SleepDayPlan {
  date: string;
  morningWakeAt: string;
  nightStartAt: string;
  nightEndAt: string;
  dayNapPredictions: SleepPredictionTarget[];
  nightPrediction: SleepPredictionTarget;
  explanation: string;
}

export interface SleepPlan {
  generatedAt: string;
  ageBand: string;
  confidence: number;
  reason: string;
  recentSampleCount: number;
  wakeWindowMinutes: number;
  wakeWindowMarginMinutes: number;
  averageNapMinutes: number;
  averageNightMinutes: number;
  modelDetails?: PredictionModelDetails;
  nextSleepAt: string | null;
  windowStart: string | null;
  windowEnd: string | null;
  nextKind: 'nap' | 'night' | null;
  plans: SleepDayPlan[];
}

export interface DashboardSummary {
  state: SleepState;
  stateSince: string | null;
  currentSleep: SleepEvent | null;
  prediction: SleepPrediction;
  sleepTodayMinutes: number;
  lastCryAt: string | null;
  cryActive: boolean;
  latestFrame: FrameRecord | null;
  recentSleep: SleepEvent[];
  recentCry: CryEvent[];
  updatedAt: string | null;
}

export interface ManualSleepInput {
  startedAt: string;
  endedAt: string | null;
  kind: SleepKind;
  notes: string;
  details: SleepEventDetails;
}

export interface ConnectionTestResult {
  ok: boolean;
  message?: string;
}

export interface RetentionEstimate {
  frames: number;
  bytes: number;
}

export interface HistoryTransferCounts {
  frames: number;
  storedImages: number;
  sleepEvents: number;
  cryEvents: number;
}

export interface HistoryTransferExport {
  archiveId: string;
  filename: string;
  createdAt: string;
  manifestSha256: string;
  bytes: number;
  counts: HistoryTransferCounts;
  downloadUrl: string;
}

export interface HistoryImportReceipt {
  format: 'baby-monitor-import-receipt';
  formatVersion: 1;
  datasetId: string;
  generation: number;
  manifestSha256: string;
  destinationInstallationId: string;
  importedAt: string;
  counts: HistoryTransferCounts;
}

export interface HistoryTransferStatus {
  status: 'active' | 'preparing' | 'pending' | 'retired';
  writable: boolean;
  datasetId: string;
  generation: number;
  outgoing: HistoryTransferExport | null;
  lastImport: HistoryImportReceipt | null;
}

export interface HistoryImportResult {
  ok: boolean;
  idempotent: boolean;
  receipt: HistoryImportReceipt;
  counts: HistoryTransferCounts;
  status: HistoryTransferStatus;
}

export interface HealthStatus {
  ok: boolean;
  database: boolean;
  runtime: string;
  background: {
    running: boolean;
    workers: Record<string, boolean>;
    errors: Record<string, string>;
  };
}

export const DEFAULT_SETTINGS: AppSettings = {
  configured: false,
  schemaVersion: 1,
  baby: {
    name: '',
    birthDate: null,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
    locationId: 'home',
    locationName: 'Home',
  },
  homeAssistant: {
    mode: 'auto',
    baseUrl: null,
    accessTokenConfigured: false,
  },
  camera: {
    enabled: false,
    entityId: null,
    captureIntervalSeconds: 300,
    streamUrlConfigured: false,
  },
  cry: {
    mode: 'disabled',
    entityId: null,
    positiveWindows: 2,
    windowSeconds: 0.5,
    clearAfterSeconds: 8,
    sensitivity: 'balanced',
    audioStreamUrlConfigured: false,
  },
  lights: {
    entityIds: [],
    durationSeconds: 45,
    brightnessPercent: 35,
    colorRgb: [255, 125, 72],
    restorePreviousState: true,
  },
  ai: {
    provider: 'disabled',
    model: null,
    apiKeyConfigured: false,
    baseUrl: null,
    cloudImageConsent: false,
    detail: 'low',
  },
  retention: {
    mode: 'forever',
    days: null,
  },
  notifications: {
    recipients: [],
    leadMinutes: 10,
  },
};

export function cloneDefaultSettings(): AppSettings {
  return structuredClone(DEFAULT_SETTINGS);
}

function cleanSecret(value: string | undefined): string | undefined {
  const normalized = value?.trim();
  return normalized || undefined;
}

export function normalizeHttpBaseUrl(value: string | null | undefined): string | null {
  const normalized = value?.trim().replace(/\/+$/, '');
  return normalized || null;
}

export function isValidHttpBaseUrl(value: string | null | undefined): boolean {
  const normalized = value?.trim();
  if (!normalized) return false;
  try {
    const parsed = new URL(normalized);
    return (parsed.protocol === 'http:' || parsed.protocol === 'https:')
      && Boolean(parsed.host)
      && !parsed.username
      && !parsed.password;
  } catch {
    return false;
  }
}

export function settingsToPayload(settings: AppSettings, clear: SecretName[] = []): SettingsPayload {
  const secrets: SettingsPayload['secrets'] = { clear: [...clear] };
  const accessToken = cleanSecret(settings.homeAssistant.accessToken);
  const cameraStream = cleanSecret(settings.camera.streamUrl);
  const cryStream = cleanSecret(settings.cry.audioStreamUrl);
  const apiKey = cleanSecret(settings.ai.apiKey);
  if (accessToken) secrets.home_assistant_access_token = accessToken;
  if (cameraStream) secrets.camera_stream_url = cameraStream;
  if (cryStream) secrets.cry_audio_stream_url = cryStream;
  if (apiKey) secrets.ai_api_key = apiKey;

  return {
    schema_version: 1,
    baby: {
      name: settings.baby.name.trim() || 'Baby',
      birth_date: settings.baby.birthDate || null,
      timezone: settings.baby.timezone.trim() || 'UTC',
      location_id: settings.baby.locationId.trim() || 'home',
      location_name: settings.baby.locationName.trim() || 'Home',
    },
    home_assistant: {
      mode: settings.homeAssistant.mode,
      base_url: normalizeHttpBaseUrl(settings.homeAssistant.baseUrl),
    },
    camera: {
      enabled: settings.camera.enabled,
      entity_id: settings.camera.enabled ? settings.camera.entityId : null,
      capture_interval_seconds: settings.camera.captureIntervalSeconds,
    },
    cry: {
      mode: settings.cry.mode === 'audio' ? 'rtsp_audio' : settings.cry.mode,
      entity_id: settings.cry.mode === 'binary_sensor' ? settings.cry.entityId : null,
      positive_windows: settings.cry.positiveWindows,
      window_seconds: settings.cry.windowSeconds,
      clear_after_seconds: settings.cry.clearAfterSeconds,
      sensitivity: settings.cry.sensitivity,
    },
    lights: {
      entity_ids: [...settings.lights.entityIds],
      duration_seconds: settings.lights.durationSeconds,
      brightness_percent: settings.lights.brightnessPercent,
      color_rgb: [...settings.lights.colorRgb] as [number, number, number],
      restore_previous_state: true,
    },
    ai: {
      provider: settings.ai.provider === 'local' ? 'ollama' : settings.ai.provider,
      model: settings.ai.model?.trim() || null,
      base_url: settings.ai.provider === 'local' ? normalizeHttpBaseUrl(settings.ai.baseUrl) : null,
      cloud_image_consent: settings.ai.cloudImageConsent,
      detail: settings.ai.detail,
    },
    notifications: {
      recipients: settings.notifications.recipients.map((recipient) => ({
        person_entity_id: recipient.personEntityId,
        name: recipient.name,
        notify_service: recipient.notifyService,
        targets: [...recipient.targets],
        enabled: recipient.enabled,
        language: recipient.language,
        events: [...recipient.events],
      })),
      lead_minutes: settings.notifications.leadMinutes,
    },
    retention: {
      mode: settings.retention.mode,
      days: settings.retention.mode === 'days' ? settings.retention.days : null,
    },
    secrets,
  };
}
