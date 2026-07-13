import { LitElement, html, nothing, svg, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';

import { ApiError, api } from './api';
import { icon } from './icons';
import {
  buildRhythmModel,
  localDateKey,
  rhythmArcPath,
  rhythmMarkerPosition,
  shiftDateKey,
  type RhythmMode,
} from './sleep-rhythm';
import {
  formatBytes,
  formatClock,
  formatDateTime,
  formatDuration,
  formatRelative,
  preferredLanguage,
  sleepDuration,
  translate,
  type TranslationKey,
} from './i18n';
import {
  cloneDefaultSettings,
  isValidHttpBaseUrl,
  normalizeHttpBaseUrl,
  type AppPage,
  type AppSettings,
  type CryEvent,
  type DashboardSummary,
  type FrameRecord,
  type HealthStatus,
  type HistoryImportReceipt,
  type HistoryTransferExport,
  type HistoryTransferStatus,
  type HomeAssistantEntity,
  type Language,
  type RetentionEstimate,
  type SecretName,
  type SleepEvent,
  type SleepKind,
  type VisionProvider,
} from './types';

type EntityDomain = 'camera' | 'binary_sensor' | 'light' | 'notify';
type TestKind = 'home_assistant' | 'camera' | 'cry' | 'lights' | 'notifications' | 'vision';
type HistoryKind = 'sleep' | 'cry' | 'frames';
type TestState = { busy: boolean; ok?: boolean; message?: string };
type Toast = { tone: 'success' | 'error' | 'info'; message: string };
type HistoryPageState = { total: number; nextOffset: number; loading: boolean; error: string };

const HISTORY_PAGE_LIMITS: Record<HistoryKind, number> = { sleep: 30, cry: 30, frames: 24 };

function emptyHistoryPages(): Record<HistoryKind, HistoryPageState> {
  return {
    sleep: { total: 0, nextOffset: 0, loading: false, error: '' },
    cry: { total: 0, nextOffset: 0, loading: false, error: '' },
    frames: { total: 0, nextOffset: 0, loading: false, error: '' },
  };
}

const EMPTY_SUMMARY: DashboardSummary = {
  state: 'unknown',
  stateSince: null,
  currentSleep: null,
  prediction: { nextSleepAt: null, windowStart: null, windowEnd: null, confidence: null, reason: null },
  sleepTodayMinutes: 0,
  lastCryAt: null,
  cryActive: false,
  latestFrame: null,
  recentSleep: [],
  recentCry: [],
  updatedAt: null,
};

function inputValue(event: Event): string {
  return (event.currentTarget as HTMLInputElement).value;
}

function inputChecked(event: Event): boolean {
  return (event.currentTarget as HTMLInputElement).checked;
}

function localDateTime(date: Date): string {
  const offset = date.getTimezoneOffset() * 60_000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 16);
}

function asIso(value: string): string {
  return new Date(value).toISOString();
}

@customElement('baby-monitor-app')
export class BabyMonitorApp extends LitElement {
  createRenderRoot(): HTMLElement {
    return this;
  }

  @state() private language: Language = preferredLanguage();
  @state() private page: AppPage = 'dashboard';
  @state() private loading = true;
  @state() private fatalError: { title: string; body: string } | null = null;
  @state() private inlineError = '';
  @state() private toast: Toast | null = null;
  @state() private onboarding = false;
  @state() private onboardingStep = 0;
  @state() private safetyConfirmed = false;
  @state() private settings: AppSettings = cloneDefaultSettings();
  @state() private draft: AppSettings = cloneDefaultSettings();
  @state() private saving = false;
  @state() private pendingSecretClears: SecretName[] = [];
  @state() private cameraSource: 'entity' | 'stream' = 'entity';
  @state() private entities: Record<EntityDomain, HomeAssistantEntity[]> = {
    camera: [], binary_sensor: [], light: [], notify: [],
  };
  @state() private summary: DashboardSummary = EMPTY_SUMMARY;
  @state() private health: HealthStatus | null = null;
  @state() private sleepEvents: SleepEvent[] = [];
  @state() private cryEvents: CryEvent[] = [];
  @state() private frames: FrameRecord[] = [];
  @state() private historyPages = emptyHistoryPages();
  @state() private refreshingData = false;
  @state() private liveView = false;
  @state() private cameraBusy: 'snapshot' | 'label' | '' = '';
  @state() private sleepBusy: 'start' | 'stop' | 'add' | '' = '';
  @state() private manualOpen = false;
  @state() private rhythmDate = localDateKey(new Date());
  @state() private rhythmMode: RhythmMode = new Date().getHours() >= 19 || new Date().getHours() < 9 ? 'night' : 'day';
  @state() private manualForm = {
    startedAt: localDateTime(new Date(Date.now() - 60 * 60_000)),
    endedAt: localDateTime(new Date()),
    kind: 'nap' as SleepKind,
    notes: '',
  };
  @state() private tests: Partial<Record<TestKind, TestState>> = {};
  @state() private retentionEstimate: RetentionEstimate | null = null;
  @state() private retentionEstimateError = false;
  @state() private historyTransfer: HistoryTransferStatus | null = null;
  @state() private transferBusy: 'export' | 'import' | 'cancel' | 'retire' | '' = '';
  @state() private transferFile: File | null = null;
  @state() private replaceHistoryConfirmed = false;
  @state() private importReceipt: HistoryImportReceipt | null = null;
  @state() private receiptFile: File | null = null;
  @state() private retireHistoryConfirmed = false;

  private pollTimer?: number;
  private toastTimer?: number;
  private operationalRequest = 0;

  connectedCallback(): void {
    super.connectedCallback();
    void this.load();
  }

  disconnectedCallback(): void {
    if (this.pollTimer) window.clearInterval(this.pollTimer);
    if (this.toastTimer) window.clearTimeout(this.toastTimer);
    super.disconnectedCallback();
  }

  private t(key: TranslationKey, values: Record<string, string | number> = {}): string {
    return translate(this.language, key, values);
  }

  private async load(): Promise<void> {
    this.loading = true;
    this.fatalError = null;
    try {
      const settings = await api.getSettings();
      const locallyComplete = localStorage.getItem('baby-monitor-setup-complete') === '1';
      const configured = settings.configured || locallyComplete;
      this.settings = { ...settings, configured };
      this.draft = structuredClone(this.settings);
      if (!configured && this.draft.baby.name.toLowerCase() === 'baby') this.draft.baby.name = '';
      this.cameraSource = settings.camera.entityId ? 'entity' : settings.camera.streamUrlConfigured ? 'stream' : 'entity';
      this.onboarding = !configured;
      await this.loadHealth();
      await this.loadHistoryTransfer();
      await this.loadEntities();
      if (configured) await this.loadOperationalData(false);
      this.startPolling();
    } catch (error) {
      const forbidden = error instanceof ApiError && (error.status === 401 || error.status === 403);
      this.fatalError = forbidden
        ? { title: this.t('forbiddenTitle'), body: this.t('forbiddenBody') }
        : { title: this.t('loadErrorTitle'), body: this.t('loadError') };
    } finally {
      this.loading = false;
    }
  }

  private startPolling(): void {
    if (this.pollTimer) window.clearInterval(this.pollTimer);
    this.pollTimer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void this.loadHealth();
        if (!this.onboarding && this.page === 'dashboard') void this.loadOperationalData(false);
      }
    }, 30_000);
  }

  private async loadHealth(): Promise<void> {
    try {
      this.health = await api.getHealth();
    } catch {
      // Operational requests retain their own error states. Keep the most
      // recent health snapshot rather than replacing it with a false alarm.
    }
  }

  private async loadEntities(): Promise<void> {
    const domains: EntityDomain[] = ['camera', 'binary_sensor', 'light', 'notify'];
    const results = await Promise.allSettled(domains.map((domain) => api.getEntities(domain)));
    const next = { ...this.entities };
    results.forEach((result, index) => {
      if (result.status === 'fulfilled') next[domains[index]] = result.value;
    });
    this.entities = next;
  }

  private async loadOperationalData(showSpinner = true): Promise<void> {
    const requestId = ++this.operationalRequest;
    if (showSpinner) this.refreshingData = true;
    this.historyPages = Object.fromEntries(
      (Object.entries(this.historyPages) as Array<[HistoryKind, HistoryPageState]>).map(([kind, state]) => [
        kind,
        { ...state, nextOffset: 0, loading: true, error: '' },
      ]),
    ) as Record<HistoryKind, HistoryPageState>;
    const results = await Promise.allSettled([
      api.getSummary(),
      api.getSleep(HISTORY_PAGE_LIMITS.sleep, 0),
      api.getCryEvents(HISTORY_PAGE_LIMITS.cry, 0),
      api.getFrames(HISTORY_PAGE_LIMITS.frames, 0),
    ]);
    if (requestId !== this.operationalRequest) return;
    if (results[0].status === 'fulfilled') this.summary = results[0].value;
    const nextPages = structuredClone(this.historyPages);
    if (results[1].status === 'fulfilled') {
      const page = results[1].value;
      this.sleepEvents = page.items;
      nextPages.sleep = {
        total: page.total,
        nextOffset: page.items.length ? page.offset + page.items.length : page.total,
        loading: false,
        error: '',
      };
    } else {
      nextPages.sleep = { ...nextPages.sleep, nextOffset: 0, loading: false, error: this.historyError(results[1].reason) };
    }
    if (results[2].status === 'fulfilled') {
      const page = results[2].value;
      this.cryEvents = page.items;
      nextPages.cry = {
        total: page.total,
        nextOffset: page.items.length ? page.offset + page.items.length : page.total,
        loading: false,
        error: '',
      };
    } else {
      nextPages.cry = { ...nextPages.cry, nextOffset: 0, loading: false, error: this.historyError(results[2].reason) };
    }
    if (results[3].status === 'fulfilled') {
      const page = results[3].value;
      this.frames = page.items;
      nextPages.frames = {
        total: page.total,
        nextOffset: page.items.length ? page.offset + page.items.length : page.total,
        loading: false,
        error: '',
      };
    } else {
      nextPages.frames = { ...nextPages.frames, nextOffset: 0, loading: false, error: this.historyError(results[3].reason) };
    }
    this.historyPages = nextPages;
    this.refreshingData = false;
  }

  private historyError(error: unknown): string {
    return error instanceof Error && error.message ? error.message : this.t('historyLoadError');
  }

  private async loadMoreHistory(kind: HistoryKind): Promise<void> {
    const current = this.historyPages[kind];
    if (current.loading || current.nextOffset >= current.total) return;
    this.historyPages = {
      ...this.historyPages,
      [kind]: { ...current, loading: true, error: '' },
    };

    try {
      let total = current.total;
      let nextOffset = current.nextOffset;
      if (kind === 'sleep') {
        const page = await api.getSleep(HISTORY_PAGE_LIMITS.sleep, current.nextOffset);
        const known = new Set(this.sleepEvents.map((event) => event.id));
        this.sleepEvents = [...this.sleepEvents, ...page.items.filter((event) => !known.has(event.id))];
        total = page.total;
        nextOffset = page.items.length ? page.offset + page.items.length : page.total;
      } else if (kind === 'cry') {
        const page = await api.getCryEvents(HISTORY_PAGE_LIMITS.cry, current.nextOffset);
        const known = new Set(this.cryEvents.map((event) => event.id));
        this.cryEvents = [...this.cryEvents, ...page.items.filter((event) => !known.has(event.id))];
        total = page.total;
        nextOffset = page.items.length ? page.offset + page.items.length : page.total;
      } else {
        const page = await api.getFrames(HISTORY_PAGE_LIMITS.frames, current.nextOffset);
        const known = new Set(this.frames.map((frame) => frame.id));
        this.frames = [...this.frames, ...page.items.filter((frame) => !known.has(frame.id))];
        total = page.total;
        nextOffset = page.items.length ? page.offset + page.items.length : page.total;
      }
      this.historyPages = {
        ...this.historyPages,
        [kind]: { total, nextOffset, loading: false, error: '' },
      };
    } catch (error) {
      this.historyPages = {
        ...this.historyPages,
        [kind]: { ...current, loading: false, error: this.historyError(error) },
      };
    }
  }

  private setLanguage(language: Language): void {
    this.language = language;
    localStorage.setItem('baby-monitor-language', language);
  }

  private setPage(page: AppPage): void {
    this.page = page;
    this.inlineError = '';
    this.liveView = false;
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (page === 'history') void this.loadOperationalData(true);
    if (page === 'settings') {
      this.draft = structuredClone(this.settings);
      this.cameraSource = this.draft.camera.entityId ? 'entity' : this.draft.camera.streamUrlConfigured ? 'stream' : 'entity';
      void this.loadEntities();
      void this.loadHistoryTransfer();
    }
  }

  private async loadHistoryTransfer(): Promise<void> {
    try {
      this.historyTransfer = await api.getHistoryTransfer();
    } catch {
      this.historyTransfer = null;
    }
  }

  private updateDraft(change: (draft: AppSettings) => void): void {
    const next = structuredClone(this.draft);
    change(next);
    this.draft = next;
    this.inlineError = '';
  }

  private setSecretValue(name: SecretName, value: string): void {
    this.updateDraft((draft) => {
      if (name === 'home_assistant_access_token') draft.homeAssistant.accessToken = value;
      if (name === 'camera_stream_url') draft.camera.streamUrl = value;
      if (name === 'cry_audio_stream_url') draft.cry.audioStreamUrl = value;
      if (name === 'ai_api_key') draft.ai.apiKey = value;
    });
    if (value.trim()) {
      this.pendingSecretClears = this.pendingSecretClears.filter((item) => item !== name);
    }
  }

  private markSecretForRemoval(name: SecretName): void {
    const removing = !this.pendingSecretClears.includes(name);
    this.pendingSecretClears = removing
      ? [...this.pendingSecretClears, name]
      : this.pendingSecretClears.filter((item) => item !== name);
    if (removing) {
      this.updateDraft((draft) => {
        if (name === 'home_assistant_access_token') draft.homeAssistant.accessToken = undefined;
        if (name === 'camera_stream_url') draft.camera.streamUrl = undefined;
        if (name === 'cry_audio_stream_url') draft.cry.audioStreamUrl = undefined;
        if (name === 'ai_api_key') draft.ai.apiKey = undefined;
      });
    }
  }

  private homeAssistantUrlChanged(): boolean {
    return this.settings.homeAssistant.accessTokenConfigured
      && normalizeHttpBaseUrl(this.settings.homeAssistant.baseUrl) !== normalizeHttpBaseUrl(this.draft.homeAssistant.baseUrl);
  }

  private aiEndpointChanged(): boolean {
    if (!this.settings.ai.apiKeyConfigured || this.draft.ai.provider === 'disabled') return false;
    const current = this.settings.ai.provider === 'disabled'
      ? null
      : `${this.settings.ai.provider}:${normalizeHttpBaseUrl(this.settings.ai.baseUrl) ?? ''}`;
    const candidate = `${this.draft.ai.provider}:${this.draft.ai.provider === 'local' ? normalizeHttpBaseUrl(this.draft.ai.baseUrl) ?? '' : ''}`;
    return current !== candidate;
  }

  private showToast(message: string, tone: Toast['tone'] = 'success'): void {
    this.toast = { message, tone };
    if (this.toastTimer) window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => { this.toast = null; }, 4_500);
  }

  private homeAssistantValidationError(): string {
    const settings = this.draft.homeAssistant;
    if (settings.mode !== 'standalone') return '';
    if (!settings.baseUrl?.trim()) return this.t('requiredHomeAssistantUrl');
    if (!isValidHttpBaseUrl(settings.baseUrl)) return this.t('invalidHomeAssistantUrl');
    const newToken = Boolean(settings.accessToken?.trim());
    const storedToken = settings.accessTokenConfigured
      && !this.pendingSecretClears.includes('home_assistant_access_token');
    if (this.homeAssistantUrlChanged() && !newToken) return this.t('requiredHomeAssistantTokenAgain');
    if (!newToken && !storedToken) return this.t('requiredHomeAssistantToken');
    return '';
  }

  private validationError(stage: number | 'all'): string {
    const settings = this.draft;
    if ((stage === 0 || stage === 'all') && !settings.baby.name.trim()) return this.t('requiredName');
    if ((stage === 0 || stage === 'all')
      && (!settings.baby.locationName.trim() || !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(settings.baby.locationId))) {
      return this.t('requiredLocation');
    }
    if (stage === 1 || stage === 'all') {
      const homeAssistantError = this.homeAssistantValidationError();
      if (homeAssistantError) return homeAssistantError;
      if (settings.camera.enabled) {
        const hasCamera = this.cameraSource === 'entity'
          ? Boolean(settings.camera.entityId)
          : Boolean(settings.camera.streamUrl?.trim() || (settings.camera.streamUrlConfigured && !this.pendingSecretClears.includes('camera_stream_url')));
        if (!hasCamera) return this.t('requiredCamera');
      }
      if (settings.cry.mode === 'binary_sensor' && !settings.cry.entityId) return this.t('requiredCrySensor');
      if (settings.cry.mode === 'audio'
        && !settings.cry.audioStreamUrl?.trim()
        && (!settings.cry.audioStreamUrlConfigured || this.pendingSecretClears.includes('cry_audio_stream_url'))) {
        return this.t('requiredCryStream');
      }
    }
    if (stage === 2 || stage === 'all') {
      if (settings.ai.provider !== 'disabled' && !settings.ai.model?.trim()) return this.t('requiredModel');
      if (settings.ai.provider === 'local' && !settings.ai.baseUrl?.trim()) return this.t('requiredBaseUrl');
      const newApiKey = Boolean(settings.ai.apiKey?.trim());
      const storedApiKey = settings.ai.apiKeyConfigured && !this.pendingSecretClears.includes('ai_api_key');
      const endpointChanged = this.aiEndpointChanged();
      if (endpointChanged && !newApiKey && !this.pendingSecretClears.includes('ai_api_key')) {
        return this.t('requiredApiKeyAgain');
      }
      if ((settings.ai.provider === 'gemini' || settings.ai.provider === 'openai')
        && !newApiKey
        && (!storedApiKey || endpointChanged)) {
        return this.t('requiredApiKey');
      }
      if (settings.ai.provider !== 'disabled' && !settings.ai.cloudImageConsent) {
        return this.t('requiredConsent');
      }
    }
    if (stage === 3 || stage === 'all') {
      const days = settings.retention.days;
      if (settings.retention.mode === 'days' && (!days || days < 1 || days > 3650)) return this.t('invalidRetention');
      if (stage === 3 && !this.safetyConfirmed) return this.t('requiredSafety');
    }
    return '';
  }

  private nextOnboardingStep(): void {
    const error = this.validationError(this.onboardingStep);
    if (error) {
      this.inlineError = error;
      return;
    }
    this.onboardingStep = Math.min(3, this.onboardingStep + 1);
    this.inlineError = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  private async saveSettings(finishOnboarding = false): Promise<void> {
    const error = this.validationError(finishOnboarding ? 3 : 'all');
    if (error) {
      this.inlineError = error;
      return;
    }
    this.saving = true;
    this.inlineError = '';
    try {
      const saved = await api.saveSettings(this.draft, this.pendingSecretClears);
      saved.configured = true;
      saved.camera.streamUrl = undefined;
      saved.cry.audioStreamUrl = undefined;
      saved.ai.apiKey = undefined;
      saved.homeAssistant.accessToken = undefined;
      this.settings = saved;
      this.draft = structuredClone(saved);
      this.pendingSecretClears = [];
      localStorage.setItem('baby-monitor-setup-complete', '1');
      this.showToast(this.t('saved'));
      if (finishOnboarding) {
        this.onboarding = false;
        this.page = 'dashboard';
        await this.loadOperationalData(true);
      }
    } catch (caught) {
      this.inlineError = caught instanceof Error ? caught.message : this.t('saveError');
    } finally {
      this.saving = false;
    }
  }

  private async testConnection(kind: TestKind): Promise<void> {
    if (kind === 'home_assistant') {
      const error = this.homeAssistantValidationError();
      if (error) {
        this.tests = { ...this.tests, [kind]: { busy: false, ok: false, message: error } };
        return;
      }
    }
    this.tests = { ...this.tests, [kind]: { busy: true } };
    try {
      const result = await api.testSettings(kind, this.draft);
      this.tests = { ...this.tests, [kind]: { busy: false, ok: result.ok, message: result.message } };
    } catch (error) {
      this.tests = {
        ...this.tests,
        [kind]: { busy: false, ok: false, message: error instanceof Error ? error.message : this.t('testFailed') },
      };
    }
  }

  private async refreshSnapshot(): Promise<FrameRecord | null> {
    this.cameraBusy = 'snapshot';
    try {
      const frame = await api.refreshSnapshot();
      this.summary = { ...this.summary, latestFrame: frame, updatedAt: new Date().toISOString() };
      this.frames = [frame, ...this.frames.filter((item) => item.id !== frame.id)];
      return frame;
    } catch (error) {
      this.showToast(error instanceof Error ? error.message : this.t('liveUnavailable'), 'error');
      return null;
    } finally {
      this.cameraBusy = '';
    }
  }

  private async labelSnapshot(): Promise<void> {
    if (this.settings.ai.provider === 'disabled') {
      this.setPage('settings');
      return;
    }
    this.cameraBusy = 'label';
    try {
      const source = this.summary.latestFrame ?? await api.refreshSnapshot();
      const labeled = await api.labelFrame(source.id);
      this.summary = { ...this.summary, latestFrame: labeled, updatedAt: new Date().toISOString() };
      this.frames = [labeled, ...this.frames.filter((item) => item.id !== labeled.id)];
      this.showToast(this.t('imageLabeled'));
    } catch (error) {
      this.showToast(error instanceof Error ? error.message : this.t('testFailed'), 'error');
    } finally {
      this.cameraBusy = '';
    }
  }

  private async toggleSleep(kind: 'nap' | 'night' = 'nap'): Promise<void> {
    const isSleeping = Boolean(this.summary.currentSleep) || this.summary.state === 'sleeping';
    this.sleepBusy = isSleeping ? 'stop' : 'start';
    try {
      if (isSleeping) {
        await api.stopSleep();
        this.showToast(this.t('sleepStopped'));
      } else {
        await api.startSleep(kind);
        this.showToast(this.t('sleepStarted'));
      }
      await this.loadOperationalData(false);
    } catch (error) {
      this.showToast(error instanceof Error ? error.message : this.t('saveError'), 'error');
    } finally {
      this.sleepBusy = '';
    }
  }

  private async addManualSleep(): Promise<void> {
    const start = new Date(this.manualForm.startedAt);
    const end = new Date(this.manualForm.endedAt);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || end <= start) {
      this.inlineError = this.t('invalidSleepRange');
      return;
    }
    this.sleepBusy = 'add';
    try {
      await api.addManualSleep({
        startedAt: asIso(this.manualForm.startedAt),
        endedAt: asIso(this.manualForm.endedAt),
        kind: this.manualForm.kind,
        notes: this.manualForm.notes,
      });
      this.manualOpen = false;
      this.showToast(this.t('sleepAdded'));
      await this.loadOperationalData(false);
    } catch (error) {
      this.inlineError = error instanceof Error ? error.message : this.t('saveError');
    } finally {
      this.sleepBusy = '';
    }
  }

  private async estimateRetention(): Promise<void> {
    const days = this.draft.retention.days;
    if (this.draft.retention.mode !== 'days' || !days || days < 1 || days > 3650) {
      this.retentionEstimate = null;
      return;
    }
    this.retentionEstimateError = false;
    try {
      this.retentionEstimate = await api.estimateRetention(days);
    } catch {
      this.retentionEstimate = null;
      this.retentionEstimateError = true;
    }
  }

  private downloadHistoryExport(item: HistoryTransferExport): void {
    const link = document.createElement('a');
    link.href = api.historyExportUrl(item);
    link.download = item.filename;
    link.rel = 'noopener';
    document.body.append(link);
    link.click();
    link.remove();
  }

  private async prepareHistoryExport(): Promise<void> {
    this.transferBusy = 'export';
    this.inlineError = '';
    try {
      const item = await api.prepareHistoryExport();
      await this.loadHistoryTransfer();
      this.downloadHistoryExport(item);
      this.showToast(this.t('historyExportReady'));
    } catch (error) {
      this.inlineError = error instanceof Error ? error.message : this.t('historyTransferFailed');
    } finally {
      this.transferBusy = '';
    }
  }

  private async cancelHistoryExport(): Promise<void> {
    this.transferBusy = 'cancel';
    this.inlineError = '';
    try {
      this.historyTransfer = await api.cancelHistoryExport();
      this.showToast(this.t('historyTransferCancelled'));
    } catch (error) {
      this.inlineError = error instanceof Error ? error.message : this.t('historyTransferFailed');
    } finally {
      this.transferBusy = '';
    }
  }

  private async retireExportedHistory(): Promise<void> {
    if (!this.receiptFile) {
      this.inlineError = this.t('historyReceiptChooseFile');
      return;
    }
    if (!this.retireHistoryConfirmed) {
      this.inlineError = this.t('historyRetireConfirmRequired');
      return;
    }
    this.transferBusy = 'retire';
    this.inlineError = '';
    try {
      this.historyTransfer = await api.finalizeHistoryExport(this.receiptFile, true);
      this.receiptFile = null;
      this.retireHistoryConfirmed = false;
      this.showToast(this.t('historyRetired'));
      await this.loadOperationalData(false);
    } catch (error) {
      this.inlineError = error instanceof Error ? error.message : this.t('historyTransferFailed');
    } finally {
      this.transferBusy = '';
    }
  }

  private async importHistory(): Promise<void> {
    if (!this.transferFile) {
      this.inlineError = this.t('historyImportChooseFile');
      return;
    }
    if (!this.replaceHistoryConfirmed) {
      this.inlineError = this.t('historyImportConfirmRequired');
      return;
    }
    this.transferBusy = 'import';
    this.inlineError = '';
    try {
      const result = await api.importHistory(this.transferFile, true);
      this.historyTransfer = result.status;
      this.importReceipt = result.receipt;
      this.transferFile = null;
      this.replaceHistoryConfirmed = false;
      this.showToast(this.t(result.idempotent ? 'historyAlreadyImported' : 'historyImportComplete'));
      await this.loadOperationalData(false);
    } catch (error) {
      this.inlineError = error instanceof Error ? error.message : this.t('historyTransferFailed');
    } finally {
      this.transferBusy = '';
    }
  }

  private downloadImportReceipt(): void {
    if (!this.importReceipt) return;
    const content = `${JSON.stringify(this.importReceipt, null, 2)}\n`;
    const url = URL.createObjectURL(new Blob([content], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `baby-monitor-import-receipt-g${this.importReceipt.generation}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  private period(): TranslationKey {
    const hour = new Date().getHours();
    if (hour < 12) return 'periodMorning';
    if (hour < 19) return 'periodAfternoon';
    return 'periodEvening';
  }

  private currentSleep(): SleepEvent | null {
    return this.summary.currentSleep ?? this.sleepEvents.find((event) => !event.endedAt) ?? null;
  }

  private renderLanguageToggle(): TemplateResult {
    return html`
      <div class="language-toggle" role="group" aria-label=${this.t('language')}>
        <button class=${this.language === 'es' ? 'active' : ''} @click=${() => this.setLanguage('es')} aria-pressed=${this.language === 'es'}>ES</button>
        <button class=${this.language === 'en' ? 'active' : ''} @click=${() => this.setLanguage('en')} aria-pressed=${this.language === 'en'}>EN</button>
      </div>
    `;
  }

  private renderHeader(): TemplateResult {
    const nav: Array<[AppPage, TranslationKey, 'moon' | 'history' | 'settings']> = [
      ['dashboard', 'navDashboard', 'moon'], ['history', 'navHistory', 'history'], ['settings', 'navSettings', 'settings'],
    ];
    return html`
      <header class="app-header">
        <a class="brand" href="#main" @click=${(event: Event) => event.preventDefault()}>
          <span class="brand-mark">${icon('baby', 22)}</span>
          <span><strong>${this.t('brand')}</strong><small>${this.t('brandSuffix')}</small></span>
        </a>
        <nav class="primary-nav" aria-label="Primary">
          ${nav.map(([page, label, itemIcon]) => html`
            <button class=${this.page === page ? 'active' : ''} @click=${() => this.setPage(page)} aria-current=${this.page === page ? 'page' : nothing}>
              ${icon(itemIcon, 18)}<span>${this.t(label)}</span>
            </button>
          `)}
        </nav>
        <div class="header-tools">
          <span class="privacy-chip">${icon('lock', 14)} ${this.t('localPrivate')}</span>
          ${this.renderLanguageToggle()}
        </div>
      </header>
    `;
  }

  private renderHealthBanner(): TemplateResult | typeof nothing {
    const errors = Object.keys(this.health?.background.errors ?? {});
    if (!errors.length) return nothing;
    const labels: Record<string, TranslationKey> = {
      capture: 'healthCapture',
      cry: 'healthCry',
      retention: 'healthRetention',
    };
    const services = errors.map((name) => labels[name] ? this.t(labels[name]) : name).join(', ');
    return html`
      <section class="health-banner" role="alert">
        <span class="health-banner-icon">!</span>
        <div>
          <strong>${this.t('backgroundIssueTitle')}</strong>
          <small>${this.t('backgroundIssueBody')} ${this.t('backgroundIssueServices', { services })}</small>
        </div>
        <button class="button compact tertiary" @click=${() => this.setPage('settings')}>${this.t('openSettings')}</button>
      </section>
    `;
  }

  private renderFatalError(): TemplateResult {
    return html`
      <main class="center-state" id="main">
        <div class="state-orbit error-orbit">!</div>
        <h1>${this.fatalError?.title}</h1>
        <p>${this.fatalError?.body}</p>
        <button class="button primary" @click=${() => this.load()}>${icon('refresh', 18)} ${this.t('retry')}</button>
      </main>
    `;
  }

  private renderLoading(): TemplateResult {
    return html`
      <main class="center-state" id="main" aria-busy="true">
        <div class="listening-pulse"><span></span><span></span><span></span></div>
        <p>${this.t('loading')}</p>
      </main>
    `;
  }

  private renderOnboarding(): TemplateResult {
    const labels: TranslationKey[] = ['setupProfile', 'setupHome', 'setupIntelligence', 'setupPrivacy'];
    const titles: TranslationKey[] = ['profileStepTitle', 'homeStepTitle', 'intelligenceStepTitle', 'privacyStepTitle'];
    const bodies: TranslationKey[] = ['profileStepBody', 'homeStepBody', 'intelligenceStepBody', 'privacyStepBody'];
    return html`
      <div class="onboarding-shell">
        <header class="onboarding-header">
          <div class="brand">
            <span class="brand-mark">${icon('baby', 22)}</span>
            <span><strong>${this.t('brand')}</strong><small>${this.t('brandSuffix')}</small></span>
          </div>
          ${this.renderLanguageToggle()}
        </header>
        <main class="onboarding-layout" id="main">
          <aside class="onboarding-intro">
            <span class="eyebrow">${this.t('onboardingEyebrow')}</span>
            <h1>${this.t('onboardingTitle')}</h1>
            <p>${this.t('onboardingIntro')}</p>
            <div class="onboarding-illustration" aria-hidden="true">
              <div class="moon-disc">${icon('moon', 36)}</div>
              <div class="crib-line"><span></span><span></span><span></span></div>
              <div class="quiet-wave"></div>
            </div>
            <div class="privacy-note">${icon('lock', 18)}<span>${this.t('privacyPromise')}</span></div>
          </aside>
          <section class="setup-panel">
            <div class="setup-progress" aria-label=${this.t('stepOf', { current: this.onboardingStep + 1, total: 4 })}>
              ${labels.map((label, index) => html`
                <div class=${index === this.onboardingStep ? 'active' : index < this.onboardingStep ? 'done' : ''}>
                  <span>${index < this.onboardingStep ? icon('check', 14) : index + 1}</span><small>${this.t(label)}</small>
                </div>
              `)}
            </div>
            <div class="setup-heading">
              <span class="step-label">${this.t('stepOf', { current: this.onboardingStep + 1, total: 4 })}</span>
              <h2>${this.t(titles[this.onboardingStep])}</h2>
              <p>${this.t(bodies[this.onboardingStep])}</p>
            </div>
            <div class="setup-content">
              ${this.onboardingStep === 0 ? this.renderProfileSection(true) : nothing}
              ${this.onboardingStep === 1 ? html`
                ${this.renderHomeAssistantSection(true)}${this.renderCameraSection(true)}${this.renderCrySection(true)}${this.renderLightsSection(true)}${this.renderNotificationsSection(true)}
              ` : nothing}
              ${this.onboardingStep === 2 ? this.renderVisionSection(true) : nothing}
              ${this.onboardingStep === 3 ? this.renderRetentionSection(true) : nothing}
            </div>
            ${this.inlineError ? html`<div class="inline-error" role="alert">${this.inlineError}</div>` : nothing}
            <div class="setup-actions">
              <button class="button ghost" ?disabled=${this.onboardingStep === 0 || this.saving} @click=${() => { this.onboardingStep -= 1; this.inlineError = ''; }}>
                ${this.t('back')}
              </button>
              ${this.onboardingStep < 3
                ? html`<button class="button primary" @click=${() => this.nextOnboardingStep()}>${this.t('continue')} ${icon('chevron', 16)}</button>`
                : html`<button class="button primary" ?disabled=${this.saving} @click=${() => this.saveSettings(true)}>
                    ${this.saving ? this.t('saving') : this.t('finishSetup')} ${this.saving ? nothing : icon('check', 16)}
                  </button>`}
            </div>
          </section>
        </main>
      </div>
    `;
  }

  private renderDashboard(): TemplateResult {
    const current = this.currentSleep();
    const sleeping = Boolean(current) || this.summary.state === 'sleeping';
    const stateKey: TranslationKey = sleeping ? 'sleeping' : this.summary.state === 'awake' ? 'awake' : 'unknown';
    const name = this.settings.baby.name || this.t('dashboardFallbackName');
    return html`
      <main class="page dashboard-page" id="main">
        <section class="dashboard-heading">
          <div>
            <span class="eyebrow">${this.t('nurseryNow')}</span>
            <h1>${this.t('dashboardHello', { period: this.t(this.period()), name })}</h1>
          </div>
          <button class="icon-button" aria-label=${this.t('refresh')} ?disabled=${this.refreshingData} @click=${() => this.loadOperationalData(true)}>
            <span class=${this.refreshingData ? 'spin' : ''}>${icon('refresh', 19)}</span>
          </button>
        </section>

        <section class="now-grid">
          <article class=${`sleep-scene ${sleeping ? 'sleeping' : 'awake'}`}>
            <div class="scene-glow" aria-hidden="true"></div>
            <div class="state-orbit">${sleeping ? icon('moon', 30) : icon('sun', 30)}<span></span></div>
            <div class="scene-copy">
              <span class="scene-label">${this.t('nurseryNow')}</span>
              <h2>${this.t(stateKey)}</h2>
              <p>${current ? this.t('since', { time: formatClock(current.startedAt, this.language) }) : this.summary.stateSince ? this.t('since', { time: formatClock(this.summary.stateSince, this.language) }) : this.t('updated', { time: formatRelative(this.summary.updatedAt, this.language) })}</p>
            </div>
            <div class="sleep-actions">
              ${sleeping
                ? html`<button class="button wake-button" ?disabled=${Boolean(this.sleepBusy)} @click=${() => this.toggleSleep()}>
                    ${this.sleepBusy === 'stop' ? this.t('stoppingSleep') : this.t('stopSleep')}
                  </button>`
                : html`
                    <button class="button primary-light" ?disabled=${Boolean(this.sleepBusy)} @click=${() => this.toggleSleep('nap')}>
                      ${this.sleepBusy === 'start' ? this.t('startingSleep') : this.t('startNap')}
                    </button>
                    <button class="button scene-secondary" ?disabled=${Boolean(this.sleepBusy)} @click=${() => this.toggleSleep('night')}>${this.t('startNight')}</button>
                  `}
            </div>
            <div class="scene-stats">
              <div><span>${this.t('sleepToday')}</span><strong>${formatDuration(this.summary.sleepTodayMinutes)}</strong></div>
              <div><span>${this.t('nextRest')}</span><strong>${this.summary.prediction.nextSleepAt ? formatClock(this.summary.prediction.nextSleepAt, this.language) : this.t('noPrediction')}</strong></div>
            </div>
          </article>
          ${this.renderCameraCard()}
        </section>

        <section class="signal-row">
          <article class=${`signal-card ${this.summary.cryActive ? 'alert' : ''}`}>
            <span class="signal-icon">${icon('waves', 20)}</span>
            <div><span>${this.t('cryStatus')}</span><strong>${this.health?.background.errors.cry ? this.t('monitorAttention') : this.settings.cry.mode === 'disabled' ? this.t('cryDisabled') : this.summary.cryActive ? this.t('cryActive') : this.t('allQuiet')}</strong></div>
            <small>${this.summary.lastCryAt ? this.t('lastCry', { time: formatRelative(this.summary.lastCryAt, this.language) }) : '—'}</small>
          </article>
          <article class="signal-card">
            <span class="signal-icon">${icon('clock', 20)}</span>
            <div><span>${this.t('nextRest')}</span><strong>${this.summary.prediction.nextSleepAt ? formatRelative(this.summary.prediction.nextSleepAt, this.language) : this.t('noPrediction')}</strong></div>
            <small>${this.summary.prediction.windowStart && this.summary.prediction.windowEnd ? this.t('predictionWindow', { start: formatClock(this.summary.prediction.windowStart, this.language), end: formatClock(this.summary.prediction.windowEnd, this.language) }) : '—'}</small>
          </article>
        </section>

        ${this.renderNightRibbon()}

        <section class="recent-section">
          <div class="section-heading">
            <div><span class="eyebrow">${this.t('recentRhythm')}</span><h2>${this.t('sleepTimeline')}</h2></div>
            <button class="text-button" @click=${() => this.setPage('history')}>${this.t('viewRhythm')} ${icon('chevron', 15)}</button>
          </div>
          ${this.renderSleepList(this.sleepEvents.slice(0, 4))}
          <button class="button secondary full-mobile" @click=${() => { this.manualOpen = !this.manualOpen; }}>${icon('plus', 17)} ${this.t('manualTitle')}</button>
          ${this.manualOpen ? this.renderManualForm() : nothing}
        </section>
      </main>
    `;
  }

  private renderCameraCard(): TemplateResult {
    if (!this.settings.camera.enabled) {
      return html`
        <article class="camera-card camera-empty">
          <div class="camera-empty-art">${icon('camera', 30)}<span></span></div>
          <h2>${this.t('noCameraTitle')}</h2>
          <p>${this.t('noCameraBody')}</p>
          <button class="button secondary" @click=${() => this.setPage('settings')}>${this.t('connectCamera')}</button>
        </article>
      `;
    }
    const frame = this.summary.latestFrame;
    const imageUrl = this.liveView ? api.liveCameraUrl() : frame?.imageUrl;
    return html`
      <article class="camera-card">
        <div class="camera-visual">
          ${imageUrl
            ? html`<img src=${imageUrl} alt=${this.t('imageAlt')} @error=${() => { if (this.liveView) { this.liveView = false; this.showToast(this.t('liveUnavailable'), 'error'); } }}>`
            : html`<div class="camera-placeholder">${icon('camera', 34)}<span>${this.t('cameraRefreshing')}</span></div>`}
          <div class="camera-overlay">
            <span class=${this.liveView ? 'live-badge active' : 'live-badge'}>${this.liveView ? html`<i></i> LIVE` : this.t('snapshot')}</span>
            ${frame ? html`<small>${formatRelative(frame.capturedAt, this.language)}</small>` : nothing}
          </div>
        </div>
        <div class="camera-body">
          <div class="camera-heading"><div><span>${this.t('cameraTitle')}</span><strong>${frame ? this.t('latestCapture', { time: formatClock(frame.capturedAt, this.language) }) : this.t('noVisionLabel')}</strong></div></div>
          ${frame?.label ? html`
            <div class="vision-observation">
              <span>${icon('sparkle', 17)}</span>
              <div><small>${this.t('visionLabel')}</small><strong>${frame.label.description || (frame.label.babyPresent ? this.t(frame.label.state === 'asleep' ? 'sleeping' : frame.label.state === 'awake' ? 'awake' : 'unknown') : this.t('babyNotVisible'))}</strong></div>
              <em>${this.t('confidence', { value: Math.round(frame.label.confidence * 100) })}</em>
            </div>
          ` : nothing}
          <div class="camera-actions">
            <button class="button compact secondary" ?disabled=${Boolean(this.cameraBusy)} @click=${() => this.refreshSnapshot()}>
              ${this.cameraBusy === 'snapshot' ? html`<span class="spinner"></span>` : icon('camera', 16)} ${this.t('snapshot')}
            </button>
            <button class=${`button compact ${this.liveView ? 'active' : 'secondary'}`} @click=${() => { this.liveView = !this.liveView; }}>
              ${icon('eye', 16)} ${this.t(this.liveView ? 'stopLive' : 'live')}
            </button>
            <button class="button compact secondary" ?disabled=${Boolean(this.cameraBusy)} @click=${() => this.labelSnapshot()}>
              ${this.cameraBusy === 'label' ? html`<span class="spinner"></span>` : icon('sparkle', 16)} ${this.t(this.cameraBusy === 'label' ? 'labeling' : 'labelImage')}
            </button>
          </div>
        </div>
      </article>
    `;
  }

  private renderNightRibbon(): TemplateResult {
    const now = Date.now();
    const duration = 12 * 60 * 60_000;
    const start = now - duration;
    const percent = (value: number): number => Math.max(0, Math.min(100, ((value - start) / duration) * 100));
    const sleeps = this.sleepEvents.filter((event) => new Date(event.endedAt ?? now).getTime() >= start && new Date(event.startedAt).getTime() <= now);
    const cries = this.cryEvents.filter((event) => new Date(event.detectedAt).getTime() >= start);
    const frames = this.frames.filter((frame) => new Date(frame.capturedAt).getTime() >= start).slice(0, 18);
    return html`
      <section class="night-ribbon-card">
        <div class="ribbon-heading">
          <div><span>${icon('moon', 18)} ${this.t('nightRibbon')}</span><small>${this.t('nightRibbonHint')}</small></div>
          <div class="ribbon-legend"><span class="sleep-dot">${this.t('ribbonSleep')}</span><span class="cry-dot">${this.t('ribbonCry')}</span><span class="frame-dot">${this.t('ribbonFrame')}</span></div>
        </div>
        <button class="ribbon-track" @click=${() => this.setPage('history')} aria-label=${this.t('viewRhythm')}>
          <span class="ribbon-midline"></span>
          ${sleeps.map((event) => {
            const left = percent(new Date(event.startedAt).getTime());
            const right = percent(new Date(event.endedAt ?? now).getTime());
            return html`<i class="sleep-band" style=${`left:${left}%;width:${Math.max(1.5, right - left)}%`} title=${`${this.t(event.kind === 'night' ? 'nightSleep' : 'nap')} · ${sleepDuration(event.startedAt, event.endedAt)}`}></i>`;
          })}
          ${cries.map((event) => html`<i class="cry-mark" style=${`left:${percent(new Date(event.detectedAt).getTime())}%`} title=${`${this.t('cryActive')} · ${formatClock(event.detectedAt, this.language)}`}></i>`)}
          ${frames.map((frame) => html`<i class="frame-mark" style=${`left:${percent(new Date(frame.capturedAt).getTime())}%`} title=${`${this.t('snapshot')} · ${formatClock(frame.capturedAt, this.language)}`}></i>`)}
          <i class="now-mark"></i>
        </button>
        <div class="ribbon-times"><span>${formatClock(new Date(start).toISOString(), this.language)}</span><span>${formatClock(new Date(start + duration / 2).toISOString(), this.language)}</span><span>${formatClock(new Date(now).toISOString(), this.language)}</span></div>
      </section>
    `;
  }

  private rhythmDateValue(dateKey: string): Date {
    const [year, month, day] = dateKey.split('-').map(Number);
    return new Date(year, month - 1, day, 12, 0, 0, 0);
  }

  private rhythmDayKeys(): string[] {
    const today = localDateKey(new Date());
    const firstOffset = this.rhythmDate === today ? -6 : -3;
    return Array.from({ length: 7 }, (_, index) => shiftDateKey(this.rhythmDate, firstOffset + index));
  }

  private moveRhythmDate(days: number): void {
    const today = localDateKey(new Date());
    const next = shiftDateKey(this.rhythmDate, days);
    this.rhythmDate = next > today ? today : next;
  }

  private renderDailyRhythm(): TemplateResult {
    const today = localDateKey(new Date());
    const model = buildRhythmModel(this.sleepEvents, this.rhythmDate, this.rhythmMode);
    const selectedDate = this.rhythmDateValue(this.rhythmDate);
    const locale = this.language === 'es' ? 'es-ES' : 'en-GB';
    const titleDate = new Intl.DateTimeFormat(locale, { weekday: 'long', day: 'numeric', month: 'long' }).format(selectedDate);
    const coreDate = new Intl.DateTimeFormat(locale, { weekday: 'short', day: 'numeric' }).format(selectedDate);
    const dayKeys = this.rhythmDayKeys();
    const napCount = model.segments.filter((segment) => segment.event.kind !== 'night').length;
    const nightCount = model.segments.filter((segment) => segment.event.kind === 'night').length;
    const averageNap = napCount ? Math.round(model.napMinutes / napCount) : 0;
    const lastDay = dayKeys.at(-1) ?? this.rhythmDate;

    return html`
      <section class=${`rhythm-visual-card ${this.rhythmMode}`} aria-label=${this.t('rhythmVisualTitle')}>
        <header class="rhythm-visual-head">
          <div>
            <span class="eyebrow">${this.t(this.rhythmMode === 'night' ? 'rhythmNight' : 'rhythmDay')}</span>
            <h2>${titleDate}</h2>
          </div>
          <div class="rhythm-date-nav">
            <button class="icon-button small rhythm-prev" aria-label=${this.t('rhythmPreviousDays')} @click=${() => this.moveRhythmDate(-7)}>${icon('chevron', 17)}</button>
            <button class="text-button rhythm-today" ?disabled=${this.rhythmDate === today} @click=${() => { this.rhythmDate = today; }}>${this.t('rhythmToday')}</button>
            <button class="icon-button small" aria-label=${this.t('rhythmNextDays')} ?disabled=${lastDay >= today} @click=${() => this.moveRhythmDate(7)}>${icon('chevron', 17)}</button>
          </div>
        </header>

        <div class="rhythm-week" aria-label=${this.t('rhythmChooseDay')}>
          ${dayKeys.map((dateKey) => {
            const date = this.rhythmDateValue(dateKey);
            const future = dateKey > today;
            const selected = dateKey === this.rhythmDate;
            const isToday = dateKey === today;
            return html`
              <button
                class=${`rhythm-day ${selected ? 'selected' : ''} ${isToday ? 'today' : ''}`}
                ?disabled=${future}
                aria-pressed=${selected}
                @click=${() => { this.rhythmDate = dateKey; }}
              >
                <span>${new Intl.DateTimeFormat(locale, { weekday: 'short' }).format(date)}</span>
                <strong>${date.getDate()}</strong>
                <small>${isToday ? this.t('rhythmToday') : new Intl.DateTimeFormat(locale, { month: 'short' }).format(date)}</small>
              </button>
            `;
          })}
        </div>

        <div class="rhythm-orbit-wrap">
          <div class="rhythm-orbit" aria-label=${this.t('rhythmRecordedSleep', { duration: formatDuration(model.totalMinutes) })}>
            <svg class="rhythm-ring" viewBox="0 0 320 320" aria-hidden="true">
              <circle class="rhythm-ring-track" cx="160" cy="160" r="122"></circle>
              <circle class="rhythm-ring-inner" cx="160" cy="160" r="82"></circle>
              <line class="rhythm-midnight-line" x1="160" y1="30" x2="160" y2="56"></line>
              ${model.segments.map((segment) => svg`
                <path
                  class=${`rhythm-arc ${segment.event.kind === 'night' ? 'night-sleep' : 'nap'} ${segment.event.endedAt ? '' : 'ongoing'}`}
                  d=${rhythmArcPath(segment.startRatio, segment.endRatio)}
                ></path>
              `)}
            </svg>
            ${model.segments.map((segment) => {
              const position = rhythmMarkerPosition(segment);
              const label = this.t(segment.event.kind === 'night' ? 'nightSleep' : 'nap');
              const detail = `${label} · ${formatClock(segment.start.toISOString(), this.language)}–${formatClock(segment.end.toISOString(), this.language)} · ${formatDuration(segment.minutes)}`;
              return html`
                <span class=${`rhythm-marker ${segment.event.kind === 'night' ? 'night-sleep' : 'nap'}`} style=${`--x:${position.x}%;--y:${position.y}%`} title=${detail} aria-label=${detail}>
                  ${icon('moon', 15)}
                </span>
              `;
            })}
            <div class="rhythm-core">
              <small>${this.t(this.rhythmMode === 'night' ? 'rhythmNightTo' : 'rhythmDayOf')}</small>
              <strong>${coreDate}</strong>
              <div class="rhythm-mode" role="group" aria-label=${this.t('rhythmMode')}>
                <button class=${this.rhythmMode === 'night' ? 'active' : ''} aria-pressed=${this.rhythmMode === 'night'} @click=${() => { this.rhythmMode = 'night'; }}>${icon('moon', 18)}<span>${this.t('rhythmNight')}</span></button>
                <button class=${this.rhythmMode === 'day' ? 'active' : ''} aria-pressed=${this.rhythmMode === 'day'} @click=${() => { this.rhythmMode = 'day'; }}>${icon('sun', 18)}<span>${this.t('rhythmDay')}</span></button>
              </div>
            </div>
            <div class="rhythm-endpoint left"><span>${icon(this.rhythmMode === 'night' ? 'moon' : 'sun', 16)}</span><small>${this.t(this.rhythmMode === 'night' ? 'rhythmBed' : 'rhythmWake')}</small><strong>${model.bedAt && this.rhythmMode === 'night' ? formatClock(model.bedAt.toISOString(), this.language) : model.wakeAt && this.rhythmMode === 'day' ? formatClock(model.wakeAt.toISOString(), this.language) : '—'}</strong></div>
            <div class="rhythm-endpoint right"><span>${icon(this.rhythmMode === 'night' ? 'sun' : 'moon', 16)}</span><small>${this.t(this.rhythmMode === 'night' ? 'rhythmWake' : 'rhythmBed')}</small><strong>${model.wakeAt && this.rhythmMode === 'night' ? formatClock(model.wakeAt.toISOString(), this.language) : model.bedAt && this.rhythmMode === 'day' ? formatClock(model.bedAt.toISOString(), this.language) : '—'}</strong></div>
          </div>
        </div>

        <div class="rhythm-summary">
          <div class="rhythm-total"><span>${icon('moon', 20)}</span><div><small>${this.t('rhythmTotal')}</small><strong>${model.totalMinutes ? formatDuration(model.totalMinutes) : this.t('rhythmNoSleep')}</strong></div><b>${model.segments.length}</b></div>
          <div class="rhythm-duration-track" aria-hidden="true">
            ${model.segments.map((segment) => html`<i class=${segment.event.kind === 'night' ? 'night-sleep' : 'nap'} style=${`--width:${model.totalMinutes ? Math.max(4, segment.minutes / model.totalMinutes * 100) : 0}%`}></i>`)}
          </div>
          <div class="rhythm-stats">
            <div><span>${this.t('rhythmNaps')}</span><strong>${napCount} · ${formatDuration(model.napMinutes)}</strong></div>
            <div><span>${this.t('rhythmNightPeriods')}</span><strong>${nightCount} · ${formatDuration(model.nightMinutes)}</strong></div>
            <div><span>${this.t('rhythmAverageNap')}</span><strong>${formatDuration(averageNap)}</strong></div>
          </div>
        </div>
        ${!model.segments.length ? html`<p class="rhythm-empty">${this.t('rhythmEmptyHint')}</p>` : nothing}
      </section>
    `;
  }

  private renderSleepList(events: SleepEvent[]): TemplateResult {
    if (!events.length) return html`<div class="empty-state compact-empty">${icon('moon', 24)}<p>${this.t('noSleepEvents')}</p></div>`;
    return html`<div class="moment-list">${events.map((event) => html`
      <article class="moment-row">
        <span class=${`moment-symbol ${event.endedAt ? '' : 'active'}`}>${icon(event.endedAt ? 'moon' : 'waves', 17)}</span>
        <div class="moment-main"><strong>${this.t(event.kind === 'night' ? 'nightSleep' : event.kind === 'nap' ? 'nap' : 'unknownType')}</strong><small>${formatDateTime(event.startedAt, this.language)} · ${this.t('location')}: ${event.locationId}${event.notes ? ` · ${event.notes}` : ''}</small></div>
        <div class="moment-meta"><strong>${sleepDuration(event.startedAt, event.endedAt)}</strong><small>${event.endedAt ? this.t(event.source === 'vision' ? 'vision' : event.source === 'import' ? 'imported' : event.source === 'automatic' ? 'automatic' : 'manual') : this.t('ongoing')}</small></div>
      </article>
    `)}</div>`;
  }

  private renderManualForm(): TemplateResult {
    return html`
      <form class="manual-form" @submit=${(event: SubmitEvent) => { event.preventDefault(); void this.addManualSleep(); }}>
        <div class="form-heading"><div><h3>${this.t('manualTitle')}</h3><p>${this.t('manualHint')}</p></div><button type="button" class="icon-button small" aria-label=${this.t('dismiss')} @click=${() => { this.manualOpen = false; }}>&times;</button></div>
        <div class="field-grid two">
          <label class="field"><span>${this.t('startedAt')}</span><input type="datetime-local" .value=${this.manualForm.startedAt} @input=${(event: Event) => { this.manualForm = { ...this.manualForm, startedAt: inputValue(event) }; }} required></label>
          <label class="field"><span>${this.t('endedAt')}</span><input type="datetime-local" .value=${this.manualForm.endedAt} @input=${(event: Event) => { this.manualForm = { ...this.manualForm, endedAt: inputValue(event) }; }} required></label>
        </div>
        <div class="field-grid two">
          <div class="field"><span>${this.t('sleepType')}</span>${this.renderChoiceRow([
            ['nap', 'nap'], ['night', 'nightSleep'],
          ], this.manualForm.kind, (value) => { this.manualForm = { ...this.manualForm, kind: value as SleepKind }; })}</div>
          <label class="field"><span>${this.t('notes')}</span><input .value=${this.manualForm.notes} placeholder=${this.t('notesPlaceholder')} @input=${(event: Event) => { this.manualForm = { ...this.manualForm, notes: inputValue(event) }; }}></label>
        </div>
        ${this.inlineError ? html`<div class="inline-error" role="alert">${this.inlineError}</div>` : nothing}
        <div class="form-actions"><button type="button" class="button ghost" @click=${() => { this.manualOpen = false; }}>${this.t('cancel')}</button><button class="button primary" ?disabled=${this.sleepBusy === 'add'}>${this.sleepBusy === 'add' ? this.t('addingSleep') : this.t('addSleep')}</button></div>
      </form>
    `;
  }

  private renderHistory(): TemplateResult {
    return html`
      <main class="page history-page" id="main">
        <section class="page-heading">
          <div><span class="eyebrow">${this.t('navHistory')}</span><h1>${this.t('historyTitle')}</h1><p>${this.t('historyIntro')}</p></div>
          <div class="heading-actions"><button class="button secondary" @click=${() => { this.manualOpen = !this.manualOpen; }}>${icon('plus', 17)} ${this.t('manualTitle')}</button><button class="icon-button" aria-label=${this.t('refresh')} ?disabled=${this.refreshingData} @click=${() => this.loadOperationalData(true)}><span class=${this.refreshingData ? 'spin' : ''}>${icon('refresh', 19)}</span></button></div>
        </section>
        ${this.manualOpen ? this.renderManualForm() : nothing}
        ${this.renderDailyRhythm()}
        ${this.renderNightRibbon()}
        <section class="history-grid">
          <article class="history-panel" aria-busy=${this.historyPages.sleep.loading}><div class="panel-heading"><span class="panel-icon">${icon('moon', 18)}</span><div><h2>${this.t('sleepTimeline')}</h2><small>${this.historyPages.sleep.total}</small></div></div>${this.sleepEvents.length || (!this.historyPages.sleep.loading && !this.historyPages.sleep.error) ? this.renderSleepList(this.sleepEvents) : nothing}${this.renderHistoryPager('sleep', this.sleepEvents.length)}</article>
          <article class="history-panel" aria-busy=${this.historyPages.cry.loading}><div class="panel-heading"><span class="panel-icon coral">${icon('waves', 18)}</span><div><h2>${this.t('cryTimeline')}</h2><small>${this.historyPages.cry.total}</small></div></div>${this.cryEvents.length || (!this.historyPages.cry.loading && !this.historyPages.cry.error) ? this.renderCryList() : nothing}${this.renderHistoryPager('cry', this.cryEvents.length)}</article>
        </section>
        <section class="frame-section" aria-busy=${this.historyPages.frames.loading}>
          <div class="section-heading"><div><span class="eyebrow">${this.t('cameraTitle')}</span><h2>${this.t('imageTimeline')}</h2></div></div>
          ${this.frames.length ? html`<div class="frame-grid">${this.frames.map((frame) => this.renderFrame(frame))}</div>` : !this.historyPages.frames.loading && !this.historyPages.frames.error ? html`<div class="empty-state">${icon('camera', 26)}<p>${this.t('noFrames')}</p></div>` : nothing}
          ${this.renderHistoryPager('frames', this.frames.length)}
        </section>
      </main>
    `;
  }

  private renderHistoryPager(kind: HistoryKind, loaded: number): TemplateResult | typeof nothing {
    const state = this.historyPages[kind];
    const hasMore = state.nextOffset < state.total;
    if (!state.loading && !state.error && !hasMore && state.total <= HISTORY_PAGE_LIMITS[kind]) return nothing;
    return html`
      <div class="history-pager" aria-live="polite">
        <small>${this.t('historyShowing', { shown: loaded, total: state.total })}</small>
        ${state.error ? html`
          <span class="history-pager-error" role="alert">${state.error}</span>
          <button type="button" class="button compact tertiary" @click=${() => state.nextOffset === 0 ? this.loadOperationalData(true) : this.loadMoreHistory(kind)}>${this.t('retry')}</button>
        ` : state.loading ? html`
          <span class="history-pager-loading"><span class="spinner"></span> ${this.t(state.nextOffset === 0 ? 'loadingHistory' : 'loadingMore')}</span>
        ` : hasMore ? html`
          <button type="button" class="button compact tertiary" @click=${() => this.loadMoreHistory(kind)}>${this.t('loadOlder')}</button>
        ` : html`<span class="history-pager-complete">${this.t('historyComplete')}</span>`}
      </div>
    `;
  }

  private renderCryList(): TemplateResult {
    if (!this.cryEvents.length) return html`<div class="empty-state compact-empty">${icon('waves', 24)}<p>${this.t('noCryEvents')}</p></div>`;
    return html`<div class="moment-list">${this.cryEvents.map((event) => html`
      <article class="moment-row cry-row"><span class="moment-symbol coral">${icon('waves', 17)}</span><div class="moment-main"><strong>${this.t('cryActive')}</strong><small>${formatDateTime(event.detectedAt, this.language)} · ${this.t('location')}: ${event.locationId}</small></div><div class="moment-meta"><strong>${event.confidence == null ? '—' : `${Math.round(event.confidence * 100)}%`}</strong><small>${event.source === 'binary_sensor' ? this.t('crySensor') : event.source === 'audio' ? this.t('cryAudio') : this.t('manual')}</small></div></article>
    `)}</div>`;
  }

  private renderFrame(frame: FrameRecord): TemplateResult {
    return html`
      <article class="frame-card">
        <div class="frame-image">${frame.imageAvailable ? html`<img loading="lazy" src=${frame.imageUrl} alt=${this.t('frameAlt', { time: formatClock(frame.capturedAt, this.language) })}>` : html`<span>${icon('camera', 26)}</span>`}</div>
        <div class="frame-copy"><span>${formatDateTime(frame.capturedAt, this.language)} · ${this.t('location')}: ${frame.locationId}</span><strong>${frame.label?.description || this.t('noVisionLabel')}</strong>${frame.label ? html`<small>${this.t('confidence', { value: Math.round(frame.label.confidence * 100) })}</small>` : nothing}</div>
      </article>
    `;
  }

  private renderSettings(): TemplateResult {
    return html`
      <main class="page settings-page" id="main">
        <section class="page-heading settings-heading"><div><span class="eyebrow">${this.t('admin')}</span><h1>${this.t('settingsTitle')}</h1><p>${this.t('settingsIntro')}</p></div><span class="admin-badge">${icon('lock', 15)} ${this.t('admin')}</span></section>
        <div class="settings-layout">
          <aside class="settings-index">
            ${[
              ['profile', 'settingsProfile', 'baby'], ['home-assistant', 'settingsHomeAssistant', 'lock'], ['camera', 'settingsCamera', 'camera'], ['cry', 'settingsCry', 'waves'],
              ['lights', 'settingsLights', 'light'], ['notifications', 'settingsNotifications', 'heart'], ['vision', 'settingsVision', 'sparkle'],
              ['transfer', 'settingsHistoryTransfer', 'history'], ['privacy', 'settingsRetention', 'lock'],
            ].map(([anchor, label, itemIcon]) => html`<a href=${`#settings-${anchor}`}>${icon(itemIcon as 'baby', 16)} ${this.t(label as TranslationKey)}</a>`)}
          </aside>
          <div class="settings-sections">
            ${this.renderProfileSection()}${this.renderHomeAssistantSection()}${this.renderCameraSection()}${this.renderCrySection()}${this.renderLightsSection()}${this.renderNotificationsSection()}${this.renderVisionSection()}${this.renderHistoryTransferSection()}${this.renderRetentionSection()}
          </div>
        </div>
        <div class="settings-savebar">
          <div>${icon('lock', 17)}<span><strong>${this.t('localPrivate')}</strong><small>${this.t('adminOnly')}</small></span></div>
          ${this.inlineError ? html`<p class="savebar-error" role="alert">${this.inlineError}</p>` : nothing}
          <button class="button primary" ?disabled=${this.saving} @click=${() => this.saveSettings(false)}>${this.saving ? this.t('saving') : this.t('save')}</button>
        </div>
      </main>
    `;
  }

  private renderProfileSection(compact = false): TemplateResult {
    const content = html`
      <div class="field-grid two">
        <label class="field"><span>${this.t('babyName')}</span><input maxlength="80" autocomplete="off" .value=${this.draft.baby.name} placeholder=${this.t('babyNamePlaceholder')} @input=${(event: Event) => this.updateDraft((draft) => { draft.baby.name = inputValue(event); })}></label>
        <label class="field"><span>${this.t('birthDate')} <em>${this.t('optional')}</em></span><input type="date" .value=${this.draft.baby.birthDate ?? ''} @input=${(event: Event) => this.updateDraft((draft) => { draft.baby.birthDate = inputValue(event) || null; })}></label>
      </div>
      <label class="field"><span>${this.t('timezone')}</span><input .value=${this.draft.baby.timezone} autocomplete="off" @input=${(event: Event) => this.updateDraft((draft) => { draft.baby.timezone = inputValue(event); })}></label>
      <div class="field-grid two">
        <label class="field"><span>${this.t('locationName')}</span><input maxlength="80" autocomplete="off" .value=${this.draft.baby.locationName} @input=${(event: Event) => this.updateDraft((draft) => { draft.baby.locationName = inputValue(event); })}></label>
        <label class="field"><span>${this.t('locationId')}</span><input maxlength="64" pattern="[a-z0-9][a-z0-9_-]{0,63}" autocomplete="off" .value=${this.draft.baby.locationId} @input=${(event: Event) => this.updateDraft((draft) => { draft.baby.locationId = inputValue(event).toLowerCase(); })}><small>${this.t('locationIdHint')}</small></label>
      </div>
    `;
    return compact ? html`<div class="compact-section">${content}</div>` : this.renderSettingsCard('profile', 'baby', 'settingsProfile', 'settingsProfileHint', content);
  }

  private renderHomeAssistantSection(compact = false): TemplateResult {
    const standalone = this.draft.homeAssistant.mode === 'standalone';
    const tokenChanged = this.homeAssistantUrlChanged() && !this.draft.homeAssistant.accessToken?.trim();
    const content = standalone ? html`
      <div class="connection-banner">
        <span>${icon('lock', 19)}</span>
        <div><strong>${this.t('homeAssistantStandalone')}</strong><small>${this.t('homeAssistantStandaloneHint')}</small></div>
      </div>
      <label class="field">
        <span>${this.t('homeAssistantUrl')}</span>
        <input type="url" inputmode="url" autocomplete="url" .value=${this.draft.homeAssistant.baseUrl ?? ''} placeholder=${this.t('homeAssistantUrlPlaceholder')} @input=${(event: Event) => this.updateDraft((draft) => { draft.homeAssistant.baseUrl = inputValue(event) || null; })}>
      </label>
      <label class="field">
        <span>${this.t('homeAssistantToken')}</span>
        <input type="password" autocomplete="new-password" .value=${this.draft.homeAssistant.accessToken ?? ''} placeholder=${this.t('homeAssistantTokenPlaceholder')} @input=${(event: Event) => this.setSecretValue('home_assistant_access_token', inputValue(event))}>
        <small>${this.t('homeAssistantTokenHint')}</small>
        ${this.renderSecretNote(this.draft.homeAssistant.accessTokenConfigured, 'home_assistant_access_token')}
      </label>
      ${tokenChanged ? html`<div class="credential-warning" role="status">${icon('lock', 16)}<span>${this.t('homeAssistantCredentialChanged')}</span></div>` : nothing}
      ${this.renderTestButton('home_assistant')}
    ` : html`
      <div class="connection-banner managed">
        <span>${icon('check', 19)}</span>
        <div><strong>${this.t('homeAssistantManaged')}</strong><small>${this.t('homeAssistantManagedHint')}</small></div>
      </div>
      ${this.draft.homeAssistant.accessTokenConfigured ? html`
        <div class="stored-secret-control"><strong>${this.t('inactiveSecret')}</strong>${this.renderSecretNote(true, 'home_assistant_access_token')}</div>
      ` : nothing}
      ${this.renderTestButton('home_assistant')}
    `;
    return compact
      ? html`<div class="compact-section"><h3>${icon('lock', 18)} ${this.t('settingsHomeAssistant')}</h3>${content}</div>`
      : this.renderSettingsCard('home-assistant', 'lock', 'settingsHomeAssistant', 'settingsHomeAssistantHint', content);
  }

  private renderCameraSection(compact = false): TemplateResult {
    const content = html`
      <label class="toggle-line"><span><strong>${this.t('cameraEnabled')}</strong><small>${this.t('settingsCameraHint')}</small></span><input type="checkbox" .checked=${this.draft.camera.enabled} @change=${(event: Event) => this.updateDraft((draft) => { draft.camera.enabled = inputChecked(event); })}><i></i></label>
      ${this.draft.camera.enabled ? html`
        <div class="field"><span>${this.t('cameraSource')}</span>${this.renderChoiceRow([
          ['entity', 'cameraEntitySource'], ['stream', 'cameraStreamSource'],
        ], this.cameraSource, (value) => {
          this.cameraSource = value as 'entity' | 'stream';
          if (value === 'stream') this.updateDraft((draft) => { draft.camera.entityId = null; });
          this.inlineError = '';
        })}</div>
        ${this.cameraSource === 'entity'
          ? html`<div class="field"><span>${this.t('cameraEntity')}</span>${this.renderSinglePicker(this.entities.camera, this.draft.camera.entityId, this.t('noCamera'), (id) => this.updateDraft((draft) => { draft.camera.entityId = id; }))}</div>`
          : html`<label class="field"><span>${this.t('cameraStream')}</span><input type="password" autocomplete="new-password" .value=${this.draft.camera.streamUrl ?? ''} placeholder=${this.t('streamPlaceholder')} @input=${(event: Event) => this.setSecretValue('camera_stream_url', inputValue(event))}>${this.renderSecretNote(this.draft.camera.streamUrlConfigured, 'camera_stream_url')}</label>`}
        <label class="field range-field"><span><b>${this.t('captureInterval')}</b><output>${Math.round(this.draft.camera.captureIntervalSeconds / 60)} min</output></span><input type="range" min="1" max="60" step="1" .value=${String(Math.round(this.draft.camera.captureIntervalSeconds / 60))} @input=${(event: Event) => this.updateDraft((draft) => { draft.camera.captureIntervalSeconds = Number(inputValue(event)) * 60; })}></label>
        ${this.renderTestButton('camera')}
      ` : nothing}
      ${this.draft.camera.streamUrlConfigured && (!this.draft.camera.enabled || this.cameraSource !== 'stream') ? html`
        <div class="stored-secret-control"><strong>${this.t('inactiveSecret')}</strong>${this.renderSecretNote(true, 'camera_stream_url')}</div>
      ` : nothing}
    `;
    return compact ? html`<div class="compact-section subsection"><h3>${icon('camera', 18)} ${this.t('settingsCamera')}</h3>${content}</div>` : this.renderSettingsCard('camera', 'camera', 'settingsCamera', 'settingsCameraHint', content);
  }

  private renderCrySection(compact = false): TemplateResult {
    const content = html`
      <div class="field"><span>${this.t('cryMode')}</span>${this.renderChoiceCards([
        ['disabled', 'cryOff', 'moon'], ['binary_sensor', 'crySensor', 'waves'], ['audio', 'cryAudio', 'camera'],
      ], this.draft.cry.mode, (value) => this.updateDraft((draft) => { draft.cry.mode = value as AppSettings['cry']['mode']; draft.cry.entityId = value === 'binary_sensor' ? draft.cry.entityId : null; }))}</div>
      ${this.draft.cry.mode === 'binary_sensor' ? html`<div class="field"><span>${this.t('sensorEntity')}</span>${this.renderSinglePicker(this.entities.binary_sensor, this.draft.cry.entityId, this.t('notConfigured'), (id) => this.updateDraft((draft) => { draft.cry.entityId = id; }))}</div>` : nothing}
      ${this.draft.cry.mode === 'audio' ? html`
        <label class="field"><span>${this.t('audioStream')}</span><input type="password" autocomplete="new-password" .value=${this.draft.cry.audioStreamUrl ?? ''} placeholder=${this.t('audioStreamPlaceholder')} @input=${(event: Event) => this.setSecretValue('cry_audio_stream_url', inputValue(event))}>${this.renderSecretNote(this.draft.cry.audioStreamUrlConfigured, 'cry_audio_stream_url')}</label>
        <div class="field"><span>${this.t('sensitivity')}</span>${this.renderChoiceRow([
          ['low', 'sensitivityLow'], ['balanced', 'sensitivityBalanced'], ['high', 'sensitivityHigh'],
        ], this.draft.cry.sensitivity, (value) => this.updateDraft((draft) => { draft.cry.sensitivity = value as AppSettings['cry']['sensitivity']; }))}</div>
      ` : nothing}
      ${this.draft.cry.mode !== 'disabled' ? this.renderTestButton('cry') : nothing}
      ${this.draft.cry.audioStreamUrlConfigured && this.draft.cry.mode !== 'audio' ? html`
        <div class="stored-secret-control"><strong>${this.t('inactiveSecret')}</strong>${this.renderSecretNote(true, 'cry_audio_stream_url')}</div>
      ` : nothing}
    `;
    return compact ? html`<div class="compact-section subsection"><h3>${icon('waves', 18)} ${this.t('settingsCry')}</h3>${content}</div>` : this.renderSettingsCard('cry', 'waves', 'settingsCry', 'settingsCryHint', content);
  }

  private renderLightsSection(compact = false): TemplateResult {
    const content = html`
      <div class="field"><span>${this.t('chooseLights')}</span>${this.renderMultiPicker(this.entities.light, this.draft.lights.entityIds, (ids) => this.updateDraft((draft) => { draft.lights.entityIds = ids; }))}</div>
      ${this.draft.lights.entityIds.length ? html`
        <div class="field-grid two">
          <label class="field range-field"><span><b>${this.t('alertDuration')}</b><output>${this.t('seconds', { count: this.draft.lights.durationSeconds })}</output></span><input type="range" min="5" max="300" step="5" .value=${String(this.draft.lights.durationSeconds)} @input=${(event: Event) => this.updateDraft((draft) => { draft.lights.durationSeconds = Number(inputValue(event)); })}></label>
          <label class="field range-field"><span><b>${this.t('brightness')}</b><output>${this.draft.lights.brightnessPercent}%</output></span><input type="range" min="1" max="100" .value=${String(this.draft.lights.brightnessPercent)} @input=${(event: Event) => this.updateDraft((draft) => { draft.lights.brightnessPercent = Number(inputValue(event)); })}></label>
        </div>
        ${this.renderTestButton('lights')}
      ` : nothing}
    `;
    return compact ? html`<div class="compact-section subsection"><h3>${icon('light', 18)} ${this.t('settingsLights')}</h3>${content}</div>` : this.renderSettingsCard('lights', 'light', 'settingsLights', 'settingsLightsHint', content);
  }

  private renderNotificationsSection(compact = false): TemplateResult {
    const content = html`
      <div class="field"><span>${this.t('notificationService')}</span>${this.renderSinglePicker(this.entities.notify, this.draft.notifications.service, this.t('noNotification'), (id) => this.updateDraft((draft) => { draft.notifications.service = id; }))}</div>
      ${this.draft.notifications.service ? html`<label class="field"><span>${this.t('notificationTargets')} <em>${this.t('optional')}</em></span><input .value=${this.draft.notifications.targets.join(', ')} @input=${(event: Event) => this.updateDraft((draft) => { draft.notifications.targets = inputValue(event).split(',').map((value) => value.trim()).filter(Boolean); })}><small>${this.t('notificationTargetsHint')}</small></label>` : nothing}
      ${this.draft.notifications.service ? this.renderTestButton('notifications', 'testNotification') : nothing}
    `;
    return compact ? html`<div class="compact-section subsection"><h3>${icon('heart', 18)} ${this.t('settingsNotifications')}</h3>${content}</div>` : this.renderSettingsCard('notifications', 'heart', 'settingsNotifications', 'settingsNotificationsHint', content);
  }

  private renderVisionSection(compact = false): TemplateResult {
    const providerNames: Record<VisionProvider, TranslationKey> = { disabled: 'aiOff', gemini: 'gemini', openai: 'openai', local: 'localCompatible' };
    const consentDestination = this.draft.ai.provider === 'local' && this.draft.ai.baseUrl
      ? this.draft.ai.baseUrl
      : this.t(providerNames[this.draft.ai.provider]);
    const content = html`
      <div class="field"><span>${this.t('aiProvider')}</span>${this.renderChoiceCards([
        ['disabled', 'aiOff', 'moon'], ['gemini', 'gemini', 'sparkle'], ['openai', 'openai', 'sparkle'], ['local', 'localCompatible', 'lock'],
      ], this.draft.ai.provider, (value) => {
        this.updateDraft((draft) => {
          const provider = value as VisionProvider;
          const previousProvider = draft.ai.provider;
          draft.ai.provider = provider;
          if (provider !== 'local') draft.ai.baseUrl = null;
          if (provider !== previousProvider) {
            draft.ai.cloudImageConsent = false;
            draft.ai.model = provider === 'gemini' ? 'gemini-3.1-flash-lite' : provider === 'openai' ? 'gpt-5.6-luna' : provider === 'local' ? 'qwen2.5vl:3b' : null;
          }
        });
      })}</div>
      ${this.draft.ai.provider !== 'disabled' ? html`
        <div class="field-grid two">
          <label class="field"><span>${this.t('model')}</span><input .value=${this.draft.ai.model ?? ''} placeholder=${this.t('modelPlaceholder')} @input=${(event: Event) => this.updateDraft((draft) => { draft.ai.model = inputValue(event) || null; })}></label>
          <label class="field"><span>${this.t('imageDetail')}</span>${this.renderChoiceRow([
            ['low', 'detailLow'], ['auto', 'detailAuto'], ['high', 'detailHigh'],
          ], this.draft.ai.detail, (value) => this.updateDraft((draft) => { draft.ai.detail = value as AppSettings['ai']['detail']; }))}</label>
        </div>
        ${this.draft.ai.provider === 'local' ? html`<label class="field"><span>${this.t('baseUrl')}</span><input type="url" .value=${this.draft.ai.baseUrl ?? ''} placeholder=${this.t('baseUrlPlaceholder')} @input=${(event: Event) => this.updateDraft((draft) => { const nextUrl = inputValue(event) || null; if (normalizeHttpBaseUrl(nextUrl) !== normalizeHttpBaseUrl(draft.ai.baseUrl)) draft.ai.cloudImageConsent = false; draft.ai.baseUrl = nextUrl; })}><small>${this.t('localProvider')}</small></label>` : nothing}
        <label class="field"><span>${this.t('apiKey')} ${this.draft.ai.provider === 'local' ? html`<em>${this.t('optional')}</em>` : nothing}</span><input type="password" autocomplete="new-password" .value=${this.draft.ai.apiKey ?? ''} placeholder=${this.t('apiKeyPlaceholder')} @input=${(event: Event) => this.setSecretValue('ai_api_key', inputValue(event))}>${this.renderSecretNote(this.draft.ai.apiKeyConfigured, 'ai_api_key')}</label>
        ${this.aiEndpointChanged() && !this.draft.ai.apiKey?.trim() && !this.pendingSecretClears.includes('ai_api_key') ? html`
          <div class="credential-warning" role="status">${icon('lock', 16)}<span>${this.t('aiCredentialChanged')}</span></div>
        ` : nothing}
        <label class="consent-box"><input type="checkbox" .checked=${this.draft.ai.cloudImageConsent} @change=${(event: Event) => this.updateDraft((draft) => { draft.ai.cloudImageConsent = inputChecked(event); })}><span><strong>${this.t('cloudConsent', { provider: consentDestination })}</strong><small>${this.t(this.draft.ai.provider === 'local' ? 'compatibleConsentHint' : 'cloudConsentHint')}</small></span></label>
        ${this.renderTestButton('vision')}
      ` : nothing}
      ${this.draft.ai.provider === 'disabled' && this.draft.ai.apiKeyConfigured ? html`
        <div class="stored-secret-control"><strong>${this.t('inactiveSecret')}</strong>${this.renderSecretNote(true, 'ai_api_key')}</div>
      ` : nothing}
    `;
    return compact ? html`<div class="compact-section">${content}</div>` : this.renderSettingsCard('vision', 'sparkle', 'settingsVision', 'settingsVisionHint', content);
  }

  private renderRetentionSection(compact = false): TemplateResult {
    const content = html`
      <div class="privacy-banner">${icon('lock', 21)}<div><strong>${this.t('privacyPromise')}</strong><small>${this.t('settingsRetentionHint')}</small></div></div>
      <div class="field"><span>${this.t('keepImages')}</span>${this.renderChoiceCards([
        ['forever', 'forever', 'history'], ['days', 'customDays', 'clock'],
      ], this.draft.retention.mode, (value) => {
        this.updateDraft((draft) => { draft.retention.mode = value as AppSettings['retention']['mode']; draft.retention.days = value === 'days' ? draft.retention.days ?? 30 : null; });
        if (value === 'days') void this.estimateRetention(); else this.retentionEstimate = null;
      })}</div>
      ${this.draft.retention.mode === 'days' ? html`
        <label class="field narrow-field"><span>${this.t('retentionDays')}</span><input type="number" min="1" max="3650" .value=${String(this.draft.retention.days ?? 30)} @input=${(event: Event) => { this.updateDraft((draft) => { draft.retention.days = Number(inputValue(event)); }); void this.estimateRetention(); }}></label>
        ${this.retentionEstimate ? html`<p class="retention-impact">${this.t('retentionImpact', { frames: this.retentionEstimate.frames, size: formatBytes(this.retentionEstimate.bytes, this.language) })}</p>` : this.retentionEstimateError ? html`<p class="field-error">${this.t('retentionEstimateError')}</p>` : nothing}
      ` : nothing}
      <div class="safety-note">${icon('heart', 20)}<p>${this.t('notMedical')}</p></div>
      ${compact ? html`<label class="consent-box safety-confirm"><input type="checkbox" .checked=${this.safetyConfirmed} @change=${(event: Event) => { this.safetyConfirmed = inputChecked(event); this.inlineError = ''; }}><span><strong>${this.t('safetyConfirm')}</strong><small>${this.t('adminOnly')}</small></span></label>` : nothing}
    `;
    return compact ? html`<div class="compact-section">${content}</div>` : this.renderSettingsCard('privacy', 'lock', 'settingsRetention', 'settingsRetentionHint', content);
  }

  private renderHistoryTransferSection(): TemplateResult {
    const outgoing = this.historyTransfer?.outgoing;
    const content = html`
      <div class="privacy-banner">${icon('history', 21)}<div><strong>${this.t('historyTransferLocal')}</strong><small>${this.t('historyTransferPrivacy')}</small></div></div>
      ${this.historyTransfer?.status === 'pending' && outgoing ? html`
        <div class="credential-warning" role="status">${icon('lock', 17)}<span>${this.t('historyTransferPending')}</span></div>
        <div class="transfer-summary">
          <strong>${outgoing.filename}</strong>
          <small>${this.t('historyTransferSummary', {
            frames: outgoing.counts.frames,
            images: outgoing.counts.storedImages,
            size: formatBytes(outgoing.bytes, this.language),
          })}</small>
        </div>
        <div class="transfer-actions">
          <button type="button" class="button secondary" @click=${() => this.downloadHistoryExport(outgoing)}>${this.t('downloadAgain')}</button>
          <button type="button" class="button tertiary" ?disabled=${this.transferBusy !== ''} @click=${() => this.cancelHistoryExport()}>${this.transferBusy === 'cancel' ? this.t('cancellingTransfer') : this.t('cancelTransfer')}</button>
        </div>
        <div class="transfer-block retire-block">
          <div><strong>${this.t('retireSource')}</strong><p>${this.t('retireSourceHint')}</p></div>
          <label class="field"><span>${this.t('historyReceipt')}</span><input type="file" accept=".json,application/json" @change=${(event: Event) => { this.receiptFile = (event.currentTarget as HTMLInputElement).files?.[0] ?? null; this.inlineError = ''; }}><small>${this.receiptFile ? `${this.receiptFile.name} · ${formatBytes(this.receiptFile.size, this.language)}` : this.t('historyReceiptHint')}</small></label>
          <label class="consent-box destructive-confirm"><input type="checkbox" .checked=${this.retireHistoryConfirmed} @change=${(event: Event) => { this.retireHistoryConfirmed = inputChecked(event); this.inlineError = ''; }}><span><strong>${this.t('retireHistoryConfirm')}</strong><small>${this.t('retireHistoryConfirmHint')}</small></span></label>
          <button type="button" class="button danger" ?disabled=${this.transferBusy !== '' || !this.receiptFile} @click=${() => this.retireExportedHistory()}>${this.transferBusy === 'retire' ? this.t('retiringHistory') : this.t('retireHistoryButton')}</button>
        </div>
      ` : this.historyTransfer?.status === 'retired' ? html`
        <div class="credential-warning" role="status">${icon('check', 17)}<span>${this.t('historyRetiredState')}</span></div>
      ` : html`
        <div class="transfer-block">
          <div><strong>${this.t('exportHistory')}</strong><p>${this.t('exportHistoryHint')}</p></div>
          <button type="button" class="button secondary" ?disabled=${this.transferBusy !== ''} @click=${() => this.prepareHistoryExport()}>${this.transferBusy === 'export' ? this.t('preparingExport') : this.t('prepareExport')}</button>
        </div>
      `}
      <div class="transfer-divider"></div>
      <div class="transfer-block import-block">
        <div><strong>${this.t('importHistory')}</strong><p>${this.t('importHistoryHint')}</p></div>
        <label class="field"><span>${this.t('historyArchive')}</span><input type="file" accept=".zip,application/zip" @change=${(event: Event) => { this.transferFile = (event.currentTarget as HTMLInputElement).files?.[0] ?? null; this.inlineError = ''; }}><small>${this.transferFile ? `${this.transferFile.name} · ${formatBytes(this.transferFile.size, this.language)}` : this.t('historyArchiveHint')}</small></label>
        <label class="consent-box danger-confirm"><input type="checkbox" .checked=${this.replaceHistoryConfirmed} @change=${(event: Event) => { this.replaceHistoryConfirmed = inputChecked(event); this.inlineError = ''; }}><span><strong>${this.t('replaceHistoryConfirm')}</strong><small>${this.t('replaceHistoryConfirmHint')}</small></span></label>
        <button type="button" class="button primary" ?disabled=${this.transferBusy !== '' || !this.transferFile} @click=${() => this.importHistory()}>${this.transferBusy === 'import' ? this.t('importingHistory') : this.t('importHistoryButton')}</button>
      </div>
      ${this.importReceipt ? html`
        <div class="transfer-result" role="status">${icon('check', 18)}<div><strong>${this.t('historyImportVerified')}</strong><small>${this.t('historyImportReceiptHint')}</small></div><button type="button" class="button compact tertiary" @click=${() => this.downloadImportReceipt()}>${this.t('downloadReceipt')}</button></div>
      ` : nothing}
    `;
    return this.renderSettingsCard('transfer', 'history', 'settingsHistoryTransfer', 'settingsHistoryTransferHint', content);
  }

  private renderSettingsCard(anchor: string, itemIcon: Parameters<typeof icon>[0], title: TranslationKey, hint: TranslationKey, content: TemplateResult): TemplateResult {
    return html`
      <section class="settings-card" id=${`settings-${anchor}`}>
        <div class="settings-card-heading"><span>${icon(itemIcon, 20)}</span><div><h2>${this.t(title)}</h2><p>${this.t(hint)}</p></div></div>
        <div class="settings-card-body">${content}</div>
      </section>
    `;
  }

  private renderChoiceRow(options: Array<[string, TranslationKey]>, selected: string, onSelect: (value: string) => void): TemplateResult {
    return html`<div class="choice-row" role="radiogroup">${options.map(([value, label]) => html`<button type="button" class=${selected === value ? 'active' : ''} role="radio" aria-checked=${selected === value} @click=${() => onSelect(value)}>${this.t(label)}</button>`)}</div>`;
  }

  private renderChoiceCards(options: Array<[string, TranslationKey, Parameters<typeof icon>[0]]>, selected: string, onSelect: (value: string) => void): TemplateResult {
    return html`<div class="choice-cards" role="radiogroup">${options.map(([value, label, itemIcon]) => html`<button type="button" class=${selected === value ? 'active' : ''} role="radio" aria-checked=${selected === value} @click=${() => onSelect(value)}><span>${icon(itemIcon, 18)}</span><strong>${this.t(label)}</strong>${selected === value ? icon('check', 15) : nothing}</button>`)}</div>`;
  }

  private renderSinglePicker(entities: HomeAssistantEntity[], selected: string | null, emptyLabel: string, onSelect: (value: string | null) => void): TemplateResult {
    const current = entities.find((entity) => entity.entityId === selected);
    return html`
      <details class="entity-picker">
        <summary><span>${current?.name ?? emptyLabel}<small>${current?.entityId ?? ''}</small></span>${icon('chevron', 16)}</summary>
        <div class="picker-menu" role="listbox">
          <button type="button" class=${selected == null ? 'active' : ''} @click=${(event: Event) => { onSelect(null); (event.currentTarget as HTMLElement).closest('details')?.removeAttribute('open'); }}>${emptyLabel}${selected == null ? icon('check', 15) : nothing}</button>
          ${entities.length ? entities.map((entity) => html`<button type="button" class=${selected === entity.entityId ? 'active' : ''} ?disabled=${!entity.available} @click=${(event: Event) => { onSelect(entity.entityId); (event.currentTarget as HTMLElement).closest('details')?.removeAttribute('open'); }}><span><strong>${entity.name}</strong><small>${entity.entityId}${entity.available ? '' : ` · ${this.t('unavailable')}`}</small></span>${selected === entity.entityId ? icon('check', 15) : nothing}</button>`) : html`<p>${this.t('emptyEntities')}</p>`}
        </div>
      </details>
    `;
  }

  private renderMultiPicker(entities: HomeAssistantEntity[], selected: string[], onChange: (value: string[]) => void): TemplateResult {
    return html`
      <details class="entity-picker multi-picker">
        <summary><span>${selected.length ? this.t('selectedCount', { count: selected.length }) : this.t('noLights')}<small>${selected.map((id) => entities.find((entity) => entity.entityId === id)?.name ?? id).join(', ')}</small></span>${icon('chevron', 16)}</summary>
        <div class="picker-menu checkbox-menu">
          ${entities.length ? entities.map((entity) => html`<label class=${entity.available ? '' : 'disabled'}><input type="checkbox" .checked=${selected.includes(entity.entityId)} ?disabled=${!entity.available} @change=${(event: Event) => { const next = inputChecked(event) ? [...selected, entity.entityId] : selected.filter((id) => id !== entity.entityId); onChange(next); }}><span><strong>${entity.name}</strong><small>${entity.entityId}</small></span></label>`) : html`<p>${this.t('emptyEntities')}</p>`}
        </div>
      </details>
    `;
  }

  private renderSecretNote(configured: boolean, name: SecretName): TemplateResult | typeof nothing {
    if (!configured) return nothing;
    const removing = this.pendingSecretClears.includes(name);
    return html`<span class=${`secret-note ${removing ? 'removing' : ''}`}>${icon('lock', 14)} ${this.t(removing ? 'secretRemovedOnSave' : 'secretStored')}<button type="button" @click=${() => this.markSecretForRemoval(name)}>${this.t(removing ? 'cancel' : 'removeStoredSecret')}</button></span>`;
  }

  private renderTestButton(kind: TestKind, label: TranslationKey = 'test'): TemplateResult {
    const state = this.tests[kind];
    return html`<div class="test-row"><button type="button" class="button tertiary compact" ?disabled=${state?.busy} @click=${() => this.testConnection(kind)}>${state?.busy ? html`<span class="spinner"></span> ${this.t('testing')}` : this.t(label)}</button>${state && !state.busy ? html`<span class=${state.ok ? 'test-ok' : 'test-error'}>${state.ok ? icon('check', 14) : '!'} ${state.message || this.t(state.ok ? 'testSuccess' : 'testFailed')}</span>` : nothing}</div>`;
  }

  render(): TemplateResult {
    return html`
      <a class="skip-link" href="#main">${this.t('skipContent')}</a>
      ${this.loading ? this.renderLoading() : this.fatalError ? this.renderFatalError() : this.onboarding ? this.renderOnboarding() : html`
        <div class="app-shell">${this.renderHeader()}${this.renderHealthBanner()}${this.page === 'dashboard' ? this.renderDashboard() : this.page === 'history' ? this.renderHistory() : this.renderSettings()}</div>
      `}
      <div class="toast-region" aria-live="polite" aria-atomic="true">${this.toast ? html`<div class=${`toast ${this.toast.tone}`}>${this.toast.tone === 'success' ? icon('check', 17) : this.toast.tone === 'error' ? '!' : icon('heart', 17)}<span>${this.toast.message}</span><button aria-label=${this.t('dismiss')} @click=${() => { this.toast = null; }}>&times;</button></div>` : nothing}</div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'baby-monitor-app': BabyMonitorApp;
  }
}
