import { LitElement, html, nothing, svg, type TemplateResult } from 'lit';
import { customElement, state } from 'lit/decorators.js';

import { ApiError, api } from './api';
import { icon, type IconName } from './icons';
import { connectWebRtcVideo } from './webrtc';
import {
  buildRhythmModel,
  localDateKey,
  rhythmArcPath,
  rhythmMarkerPosition,
  rhythmPosition,
  rhythmSvgPoint,
  rhythmTrackPath,
  shiftDateKey,
  type RhythmSegment,
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
import { formatTrendDate, medianMinutes } from './trend-format';
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
  type SleepEventDetails,
  type SleepKind,
  type SleepPause,
  type SleepPlan,
  type SleepPredictionTarget,
  type VisionProvider,
  type VisionStatistics,
} from './types';

type EntityDomain = 'camera' | 'binary_sensor' | 'light' | 'notify';
type TestKind = 'home_assistant' | 'camera' | 'cry' | 'lights' | 'notifications' | 'vision';
type HistoryKind = 'sleep' | 'cry' | 'frames';
type TestState = { busy: boolean; ok?: boolean; message?: string };
type Toast = { tone: 'success' | 'error' | 'info'; message: string };
type HistoryPageState = { total: number; nextOffset: number; loading: boolean; error: string };
type LiveTransport = 'off' | 'connecting' | 'webrtc' | 'mjpeg';
type TemporalTarget =
  | 'manual-start'
  | 'manual-end'
  | 'edit-start'
  | 'edit-end'
  | `manual-pause-start-${number}`
  | `manual-pause-end-${number}`
  | `edit-pause-start-${number}`
  | `edit-pause-end-${number}`;
type TemporalPickerState = { target: TemporalTarget; value: string; title: string };
type FrameReviewState = {
  point: 'start' | 'middle' | 'end' | '';
  frames: FrameRecord[];
  index: number;
  loading: boolean;
  error: string;
  requestedAt: string;
};
type FrameReviewContext = 'edit' | 'inferred-awake';

const EMPTY_DETAILS = (): SleepEventDetails => ({ tags: [], pauses: [] });

const DETAIL_GROUPS = [
  {
    key: 'start',
    es: 'Inicio',
    en: 'Falling asleep',
    options: [
      ['long_to_fall_asleep', '⏰', 'Mucho tiempo para dormirse', 'Took a long time to fall asleep'],
      ['upset', '☹️', 'Molesto', 'Upset'],
    ],
  },
  {
    key: 'method',
    es: 'Cómo',
    en: 'How',
    options: [
      ['in_bed', '🛏️', 'En la cama', 'In bed'],
      ['breastfeeding', '🤱', 'Lactancia', 'Breastfeeding'],
      ['held', '🫶', 'Cargado o sostenido', 'Held'],
      ['beside_parent', '🤍', 'A mi lado', 'Beside me'],
      ['bottle', '🍼', 'Toma de biberón', 'Bottle'],
      ['stroller', '👶', 'Cochecito', 'Stroller'],
      ['car', '🚗', 'Automóvil', 'Car'],
      ['swing', '🌙', 'Columpio', 'Swing'],
    ],
  },
  {
    key: 'end',
    es: 'Fin',
    en: 'Wake-up',
    options: [
      ['woken_by_parent', '🔔', 'Desperté al bebé', 'Woken by parent'],
      ['woke_alone', '🌅', 'Se despertó solo', 'Woke independently'],
    ],
  },
  {
    key: 'mood',
    es: 'Estado de ánimo al despertar',
    en: 'Mood on waking',
    options: [
      ['woke_grumpy', '🙁', 'Mal humor', 'Grumpy'],
      ['woke_neutral', '😐', 'Neutral', 'Neutral'],
      ['woke_happy', '🙂', 'Buen humor', 'Happy'],
    ],
  },
] as const;

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
  @state() private page: AppPage = 'sleep';
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
  @state() private sleepPlan: SleepPlan | null = null;
  @state() private health: HealthStatus | null = null;
  @state() private sleepEvents: SleepEvent[] = [];
  @state() private cryEvents: CryEvent[] = [];
  @state() private frames: FrameRecord[] = [];
  @state() private historyPages = emptyHistoryPages();
  @state() private refreshingData = false;
  @state() private liveView = false;
  @state() private liveTransport: LiveTransport = 'off';
  @state() private cameraBusy: 'snapshot' | 'label' | '' = '';
  @state() private sleepBusy: 'start' | 'stop' | 'add' | '' = '';
  @state() private manualOpen = false;
  @state() private editingSleep: SleepEvent | null = null;
  @state() private editSleepBusy = false;
  @state() private editSleepForm = {
    startedAt: '', endedAt: '', kind: 'nap' as SleepKind, notes: '', details: EMPTY_DETAILS(),
  };
  @state() private temporalPicker: TemporalPickerState | null = null;
  @state() private selectedPrediction: SleepPredictionTarget | null = null;
  @state() private frameReview: FrameReviewState = {
    point: '', frames: [], index: 0, loading: false, error: '', requestedAt: '',
  };
  private frameReviewContext: FrameReviewContext = 'edit';
  private editSleepFrameBounds: { startedAt: string; endedAt: string } | null = null;
  private manualFrameReviewBounds: {
    startedAt: string;
    endedAt: string;
    locationId: string | null;
  } | null = null;
  private editSleepStartChanged = false;
  private editSleepEndChanged = false;
  @state() private manualEndTouched = false;
  @state() private rhythmDate = localDateKey(new Date());
  @state() private rhythmMode: RhythmMode = new Date().getHours() >= 19 || new Date().getHours() < 9 ? 'night' : 'day';
  @state() private statsTab: 'summary' | 'naps' | 'awake' | 'night' | 'pacifier' | 'head' | 'clothing' | 'mouth' = 'summary';
  @state() private visionStatistics: VisionStatistics | null = null;
  @state() private visionStatisticsLoading = false;
  @state() private manualForm = {
    startedAt: localDateTime(new Date(Date.now() - 60 * 60_000)),
    endedAt: localDateTime(new Date()),
    kind: 'nap' as SleepKind,
    notes: '',
    details: EMPTY_DETAILS(),
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
  private liveConnectTimer?: number;
  private liveDisconnectTimer?: number;
  private livePeer?: RTCPeerConnection;
  private liveGeneration = 0;
  private operationalRequest = 0;
  private frameReviewRequest = 0;
  private readonly handleKeyDown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') {
      if (this.temporalPicker) { this.temporalPicker = null; return; }
      if (this.selectedPrediction) { this.selectedPrediction = null; return; }
      if (this.manualOpen) this.closeManualForm();
      if (this.editingSleep) this.editingSleep = null;
    }
  };

  connectedCallback(): void {
    super.connectedCallback();
    window.addEventListener('keydown', this.handleKeyDown);
    void this.load();
  }

  protected firstUpdated(): void {
    if (window.parent === window) return;
    window.parent.postMessage(
      {
        type: 'baby-monitor:ready',
        loadId: new URLSearchParams(window.location.search).get('ui'),
      },
      window.location.origin,
    );
  }

  disconnectedCallback(): void {
    if (this.pollTimer) window.clearInterval(this.pollTimer);
    if (this.toastTimer) window.clearTimeout(this.toastTimer);
    this.stopLiveView();
    window.removeEventListener('keydown', this.handleKeyDown);
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
        if (!this.onboarding && (this.page === 'sleep' || this.page === 'camera')) void this.loadOperationalData(false);
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
      api.getSleep(500, 0),
      api.getCryEvents(HISTORY_PAGE_LIMITS.cry, 0),
      api.getFrames(HISTORY_PAGE_LIMITS.frames, 0),
      api.getPredictions(),
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
    if (results[4].status === 'fulfilled') this.sleepPlan = results[4].value;
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
    this.stopLiveView();
    this.page = page;
    this.inlineError = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (page === 'camera' || page === 'data') void this.loadOperationalData(true);
    if (page === 'data') void this.loadVisionStatistics();
    if (page === 'settings') {
      this.draft = structuredClone(this.settings);
      this.cameraSource = this.draft.camera.entityId ? 'entity' : this.draft.camera.streamUrlConfigured ? 'stream' : 'entity';
      void this.loadEntities();
      void this.loadHistoryTransfer();
    }
  }

  private clearLiveTimers(): void {
    if (this.liveConnectTimer) window.clearTimeout(this.liveConnectTimer);
    if (this.liveDisconnectTimer) window.clearTimeout(this.liveDisconnectTimer);
    this.liveConnectTimer = undefined;
    this.liveDisconnectTimer = undefined;
  }

  private closeLivePeer(): void {
    const video = this.querySelector<HTMLVideoElement>('.camera-live-video');
    if (video?.srcObject && globalThis.MediaStream && video.srcObject instanceof MediaStream) {
      video.srcObject.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    }
    this.livePeer?.close();
    this.livePeer = undefined;
  }

  private stopLiveView(): void {
    this.liveGeneration += 1;
    this.clearLiveTimers();
    this.closeLivePeer();
    this.liveView = false;
    this.liveTransport = 'off';
  }

  private useMjpegFallback(generation: number): void {
    if (!this.liveView || generation !== this.liveGeneration || this.liveTransport === 'mjpeg') return;
    this.clearLiveTimers();
    this.closeLivePeer();
    this.liveTransport = 'mjpeg';
  }

  private handleWebRtcPlaying(): void {
    if (!this.liveView || this.liveTransport === 'mjpeg') return;
    if (this.liveConnectTimer) window.clearTimeout(this.liveConnectTimer);
    this.liveConnectTimer = undefined;
    this.liveTransport = 'webrtc';
  }

  private handleWebRtcState(generation: number): void {
    const state = this.livePeer?.connectionState;
    if (!this.liveView || generation !== this.liveGeneration) return;
    if (state === 'failed' || state === 'closed') {
      this.useMjpegFallback(generation);
      return;
    }
    if (state === 'disconnected' && !this.liveDisconnectTimer) {
      this.liveDisconnectTimer = window.setTimeout(() => {
        this.liveDisconnectTimer = undefined;
        if (this.livePeer?.connectionState === 'disconnected') this.useMjpegFallback(generation);
      }, 2_000);
    } else if (state === 'connected' && this.liveDisconnectTimer) {
      window.clearTimeout(this.liveDisconnectTimer);
      this.liveDisconnectTimer = undefined;
    }
  }

  private async toggleLiveView(): Promise<void> {
    if (this.liveView) {
      this.stopLiveView();
      return;
    }
    const generation = ++this.liveGeneration;
    this.liveView = true;
    this.liveTransport = 'connecting';
    await this.updateComplete;
    const video = this.querySelector<HTMLVideoElement>('.camera-live-video');
    if (!video) {
      this.useMjpegFallback(generation);
      return;
    }
    this.liveConnectTimer = window.setTimeout(() => this.useMjpegFallback(generation), 8_000);
    try {
      const peer = await connectWebRtcVideo(video, (offer) => api.negotiateWebRtc(offer));
      if (!this.liveView || generation !== this.liveGeneration) {
        peer.close();
        return;
      }
      this.livePeer = peer;
      peer.onconnectionstatechange = () => this.handleWebRtcState(generation);
      this.handleWebRtcState(generation);
    } catch {
      this.useMjpegFallback(generation);
    }
  }

  private async loadVisionStatistics(): Promise<void> {
    if (this.visionStatisticsLoading) return;
    this.visionStatisticsLoading = true;
    try {
      const start = this.sleepEvents.length
        ? new Date(Math.min(...this.sleepEvents.map((event) => new Date(event.startedAt).getTime())))
        : new Date(Date.now() - 90 * 86_400_000);
      this.visionStatistics = await api.getVisionStatistics(start.toISOString(), new Date().toISOString());
    } catch (error) {
      this.inlineError = error instanceof Error ? error.message : 'No se pudieron cargar las estadísticas visuales.';
    } finally {
      this.visionStatisticsLoading = false;
    }
  }

  private manualKindForNow(now: Date): SleepKind {
    const minutes = now.getHours() * 60 + now.getMinutes();
    if (minutes < 9 * 60) return this.summary.state === 'awake' ? 'awake' : 'night';
    if (minutes >= 20 * 60) return 'night';
    return 'nap';
  }

  private suggestedEnd(startValue: string, kind: SleepKind): string {
    const start = new Date(startValue);
    if (!Number.isFinite(start.getTime())) return '';
    let minutes = kind === 'awake' ? 15 : kind === 'night' ? 10 * 60 : this.sleepPlan?.averageNapMinutes ?? 45;
    if (kind === 'night') {
      const dateKey = localDateKey(start);
      const plan = this.sleepPlan?.plans.find((item) => item.date === dateKey);
      const plannedEnd = plan?.nightEndAt ? new Date(plan.nightEndAt) : null;
      if (plannedEnd && plannedEnd > start) minutes = Math.round((plannedEnd.getTime() - start.getTime()) / 60_000);
      else {
        const morning = new Date(start);
        morning.setDate(morning.getDate() + (start.getHours() >= 8 ? 1 : 0));
        morning.setHours(7, 30, 0, 0);
        if (morning > start) minutes = Math.round((morning.getTime() - start.getTime()) / 60_000);
      }
      minutes = Math.max(6 * 60, Math.min(12 * 60, minutes));
    }
    if (kind === 'nap' && start.getHours() >= 17) minutes = Math.min(75, minutes);
    return localDateTime(new Date(start.getTime() + Math.max(15, minutes) * 60_000));
  }

  private detailsFromEvent(event: SleepEvent): { notes: string; details: SleepEventDetails } {
    const details = structuredClone(event.details ?? EMPTY_DETAILS());
    const note = event.notes ?? '';
    if (details.tags.length || !note.includes('Etiquetas:')) return { notes: note, details };
    const aliases = new Map<string, string>();
    DETAIL_GROUPS.forEach((group) => group.options.forEach(([tag, , es, en]) => {
      aliases.set(es.toLocaleLowerCase('es'), tag);
      aliases.set(en.toLocaleLowerCase('en'), tag);
    }));
    const [clean, rawTags = ''] = note.split(/\n?Etiquetas:\s*/i, 2);
    details.tags = rawTags.split(',').map((item) => aliases.get(item.trim().toLocaleLowerCase()) ?? '')
      .filter(Boolean);
    return { notes: clean.trim(), details };
  }

  private openManualForm(): void {
    this.inlineError = '';
    this.frameReviewRequest += 1;
    this.frameReviewContext = 'edit';
    this.manualFrameReviewBounds = null;
    this.frameReview = { point: '', frames: [], index: 0, loading: false, error: '', requestedAt: '' };
    const now = new Date();
    now.setSeconds(0, 0);
    this.manualForm = {
      startedAt: localDateTime(now),
      endedAt: '',
      kind: this.manualKindForNow(now),
      notes: '',
      details: EMPTY_DETAILS(),
    };
    this.manualEndTouched = false;
    this.manualOpen = true;
  }

  private closeManualForm(): void {
    this.inlineError = '';
    this.temporalPicker = null;
    this.frameReviewRequest += 1;
    this.frameReviewContext = 'edit';
    this.manualFrameReviewBounds = null;
    this.frameReview = { point: '', frames: [], index: 0, loading: false, error: '', requestedAt: '' };
    this.manualOpen = false;
  }

  private openSleepEditor(event: SleepEvent): void {
    const restored = this.detailsFromEvent(event);
    this.frameReviewContext = 'edit';
    this.manualFrameReviewBounds = null;
    this.editingSleep = event;
    this.editSleepForm = {
      startedAt: localDateTime(new Date(event.startedAt)),
      endedAt: event.endedAt ? localDateTime(new Date(event.endedAt)) : '',
      kind: event.kind,
      notes: restored.notes,
      details: restored.details,
    };
    // The minute-precision picker is a presentation affordance. Preserve the
    // detector's exact seconds for evidence queries (and unrelated edits), or
    // the capture that closed a segment can fall just outside the rounded end.
    this.editSleepFrameBounds = {
      startedAt: event.startedAt,
      endedAt: event.endedAt ?? new Date().toISOString(),
    };
    this.editSleepStartChanged = false;
    this.editSleepEndChanged = false;
    this.frameReview = { point: '', frames: [], index: 0, loading: false, error: '', requestedAt: '' };
    void this.loadFrameReview('start');
  }

  private async saveSleepEditor(): Promise<void> {
    if (!this.editingSleep) return;
    const start = new Date(this.editSleepForm.startedAt);
    const end = this.editSleepForm.endedAt ? new Date(this.editSleepForm.endedAt) : null;
    if (!Number.isFinite(start.getTime()) || (end && (!Number.isFinite(end.getTime()) || end <= start))) {
      this.inlineError = this.t('invalidSleepRange');
      return;
    }
    if (this.editSleepForm.kind === 'awake' && !end) {
      this.inlineError = this.language === 'es'
        ? 'Los despertares necesitan una hora de fin.'
        : 'Awake periods need an end time.';
      return;
    }
    this.editSleepBusy = true;
    try {
      await api.patchSleep(this.editingSleep.id, {
        startedAt: this.editSleepStartChanged ? asIso(this.editSleepForm.startedAt) : this.editingSleep.startedAt,
        endedAt: this.editSleepForm.endedAt
          ? this.editSleepEndChanged ? asIso(this.editSleepForm.endedAt) : this.editingSleep.endedAt
          : null,
        kind: this.editSleepForm.kind, notes: this.editSleepForm.notes, details: this.editSleepForm.details,
      });
      this.editingSleep = null;
      await this.loadOperationalData(false);
      this.showToast(this.language === 'es' ? 'Sueño actualizado' : 'Sleep updated');
    } catch (error) {
      this.inlineError = error instanceof Error
        ? error.message
        : this.language === 'es' ? 'No se pudo guardar.' : 'Could not save.';
    }
    finally { this.editSleepBusy = false; }
  }

  private async deleteSleepEditor(): Promise<void> {
    if (!this.editingSleep) return;
    const automatic = this.editingSleep.source !== 'manual';
    const confirmation = this.language === 'es'
      ? automatic
        ? '¿Eliminar este segmento detectado automáticamente? Las capturas y sus análisis se conservarán.'
        : '¿Eliminar este registro de sueño? Las capturas se conservarán.'
      : automatic
        ? 'Delete this automatically detected segment? Captures and model analyses will be preserved.'
        : 'Delete this sleep record? Captures will be preserved.';
    if (!window.confirm(confirmation)) return;
    this.editSleepBusy = true;
    try {
      await api.deleteSleep(this.editingSleep.id);
      this.editingSleep = null;
      await this.loadOperationalData(false);
      this.showToast(this.language === 'es'
        ? 'Segmento eliminado; las capturas se conservan'
        : 'Segment deleted; captures preserved');
    } catch (error) {
      this.inlineError = error instanceof Error
        ? error.message
        : this.language === 'es' ? 'No se pudo eliminar.' : 'Could not delete.';
    }
    finally { this.editSleepBusy = false; }
  }

  private async openTemporalPicker(target: TemporalTarget, value: string, title: string): Promise<void> {
    const fallback = localDateTime(new Date());
    const parsed = value ? new Date(value) : new Date(fallback);
    this.temporalPicker = {
      target,
      value: Number.isFinite(parsed.getTime()) ? localDateTime(parsed) : fallback,
      title,
    };
    await this.updateComplete;
    this.querySelectorAll('.temporal-option.active').forEach((element) => {
      element.scrollIntoView({ block: 'center' });
    });
  }

  private updateTemporalDate(days: number): void {
    if (!this.temporalPicker) return;
    const date = new Date(this.temporalPicker.value);
    date.setDate(date.getDate() + days);
    this.temporalPicker = { ...this.temporalPicker, value: localDateTime(date) };
  }

  private updateTemporalClock(part: 'hour' | 'minute', value: number): void {
    if (!this.temporalPicker) return;
    const date = new Date(this.temporalPicker.value);
    if (part === 'hour') date.setHours(value);
    else date.setMinutes(value);
    date.setSeconds(0, 0);
    this.temporalPicker = { ...this.temporalPicker, value: localDateTime(date) };
  }

  private adjustTemporal(minutes: number): void {
    if (!this.temporalPicker) return;
    const date = new Date(this.temporalPicker.value);
    date.setMinutes(date.getMinutes() + minutes);
    this.temporalPicker = { ...this.temporalPicker, value: localDateTime(date) };
  }

  private applyTemporalPicker(): void {
    if (!this.temporalPicker) return;
    const { target, value } = this.temporalPicker;
    let reloadFrameReview = false;
    if (target === 'manual-start') this.manualForm = { ...this.manualForm, startedAt: value };
    else if (target === 'manual-end') {
      this.manualForm = { ...this.manualForm, endedAt: value };
      this.manualEndTouched = true;
    } else if (target === 'edit-start') {
      this.editSleepForm = { ...this.editSleepForm, startedAt: value };
      this.editSleepFrameBounds = {
        startedAt: asIso(value),
        endedAt: this.editSleepFrameBounds?.endedAt ?? this.editingSleep?.endedAt ?? new Date().toISOString(),
      };
      this.editSleepStartChanged = true;
      reloadFrameReview = true;
    } else if (target === 'edit-end') {
      this.editSleepForm = { ...this.editSleepForm, endedAt: value };
      this.editSleepFrameBounds = {
        startedAt: this.editSleepFrameBounds?.startedAt ?? this.editingSleep?.startedAt ?? asIso(this.editSleepForm.startedAt),
        endedAt: asIso(value),
      };
      this.editSleepEndChanged = true;
      reloadFrameReview = true;
    }
    else {
      const match = target.match(/^(manual|edit)-pause-(start|end)-(\d+)$/);
      if (match) {
        const [, scope, edge, rawIndex] = match;
        const index = Number(rawIndex);
        const form = scope === 'manual' ? this.manualForm : this.editSleepForm;
        const pauses = form.details.pauses.map((pause, pauseIndex) => pauseIndex === index
          ? { ...pause, [edge === 'start' ? 'startedAt' : 'endedAt']: asIso(value) }
          : pause);
        if (scope === 'manual') {
          this.manualForm = { ...this.manualForm, details: { ...this.manualForm.details, pauses } };
        } else {
          this.editSleepForm = { ...this.editSleepForm, details: { ...this.editSleepForm.details, pauses } };
        }
      }
    }
    this.temporalPicker = null;
    if (reloadFrameReview) void this.loadFrameReview('start');
  }

  private setManualKind(kind: SleepKind): void {
    const suggestion = this.suggestedEnd(this.manualForm.startedAt, kind);
    this.manualForm = {
      ...this.manualForm,
      kind,
      endedAt: this.manualEndTouched ? this.manualForm.endedAt : '',
      details: kind === 'awake' ? EMPTY_DETAILS() : this.manualForm.details,
    };
    if (kind === 'awake' && !this.manualForm.endedAt) {
      this.manualForm = { ...this.manualForm, endedAt: suggestion };
    }
  }

  private toggleDetailTag(scope: 'manual' | 'edit', tag: string): void {
    const form = scope === 'manual' ? this.manualForm : this.editSleepForm;
    const tags = form.details.tags.includes(tag)
      ? form.details.tags.filter((item) => item !== tag)
      : [...form.details.tags, tag];
    if (scope === 'manual') {
      this.manualForm = { ...this.manualForm, details: { ...this.manualForm.details, tags } };
    } else {
      this.editSleepForm = { ...this.editSleepForm, details: { ...this.editSleepForm.details, tags } };
    }
  }

  private addPause(scope: 'manual' | 'edit'): void {
    const form = scope === 'manual' ? this.manualForm : this.editSleepForm;
    const start = new Date(form.startedAt);
    const endValue = form.endedAt || (scope === 'manual' ? this.suggestedEnd(form.startedAt, form.kind) : '');
    const end = new Date(endValue);
    if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime()) || end <= start) {
      this.inlineError = 'Elige primero un inicio y un fin para añadir la pausa.';
      return;
    }
    if (end.getTime() - start.getTime() < 20 * 60_000) {
      this.inlineError = 'El tramo es demasiado corto para añadir una pausa.';
      return;
    }
    const middle = new Date((start.getTime() + end.getTime()) / 2);
    const pause: SleepPause = {
      startedAt: new Date(middle.getTime() - 5 * 60_000).toISOString(),
      endedAt: new Date(middle.getTime() + 5 * 60_000).toISOString(),
    };
    const pauses = [...form.details.pauses, pause];
    if (scope === 'manual') {
      this.manualForm = { ...this.manualForm, details: { ...this.manualForm.details, pauses } };
    } else {
      this.editSleepForm = { ...this.editSleepForm, details: { ...this.editSleepForm.details, pauses } };
    }
    this.inlineError = '';
  }

  private removePause(scope: 'manual' | 'edit', index: number): void {
    if (scope === 'manual') {
      this.manualForm = {
        ...this.manualForm,
        details: { ...this.manualForm.details, pauses: this.manualForm.details.pauses.filter((_, item) => item !== index) },
      };
    } else {
      this.editSleepForm = {
        ...this.editSleepForm,
        details: { ...this.editSleepForm.details, pauses: this.editSleepForm.details.pauses.filter((_, item) => item !== index) },
      };
    }
  }

  private reviewPointDate(point: 'start' | 'middle' | 'end'): Date | null {
    const range = this.frameReviewRange();
    if (!range) return null;
    const { start, end } = range;
    if (point === 'start') return start;
    if (point === 'end') return end;
    return new Date((start.getTime() + end.getTime()) / 2);
  }

  private frameReviewRange(): { start: Date; end: Date } | null {
    const manualBounds = this.frameReviewContext === 'inferred-awake'
      ? this.manualFrameReviewBounds
      : null;
    const start = new Date(manualBounds?.startedAt ?? this.editSleepFrameBounds?.startedAt ?? this.editSleepForm.startedAt);
    const configuredEnd = manualBounds?.endedAt
      ? new Date(manualBounds.endedAt)
      : this.editSleepFrameBounds?.endedAt
        ? new Date(this.editSleepFrameBounds.endedAt)
        : this.editSleepForm.endedAt ? new Date(this.editSleepForm.endedAt) : new Date();
    if (!Number.isFinite(start.getTime()) || !Number.isFinite(configuredEnd.getTime()) || configuredEnd <= start) {
      return null;
    }
    return { start, end: configuredEnd };
  }

  private frameIndexForPoint(frames: FrameRecord[], point: 'start' | 'middle' | 'end'): number {
    if (!frames.length) return 0;
    if (point === 'start') return 0;
    if (point === 'end') return frames.length - 1;
    const requested = this.reviewPointDate(point);
    if (!requested) return 0;
    return frames.reduce((closest, frame, index) => {
      const currentDistance = Math.abs(new Date(frame.capturedAt).getTime() - requested.getTime());
      const closestDistance = Math.abs(new Date(frames[closest].capturedAt).getTime() - requested.getTime());
      return currentDistance < closestDistance ? index : closest;
    }, 0);
  }

  private selectFrameReviewPoint(point: 'start' | 'middle' | 'end'): void {
    if (!this.frameReview.frames.length) {
      void this.loadFrameReview(point);
      return;
    }
    const requested = this.reviewPointDate(point);
    this.frameReview = {
      ...this.frameReview,
      point,
      index: this.frameIndexForPoint(this.frameReview.frames, point),
      requestedAt: requested?.toISOString() ?? '',
    };
  }

  private async loadFrameReview(point: 'start' | 'middle' | 'end'): Promise<void> {
    const requestId = ++this.frameReviewRequest;
    const context = this.frameReviewContext;
    const range = this.frameReviewRange();
    const requested = this.reviewPointDate(point);
    if (!range || !requested) {
      this.frameReview = {
        point,
        frames: [],
        index: 0,
        loading: false,
        error: this.language === 'es'
          ? 'Elige un inicio y un fin válidos para revisar las capturas.'
          : 'Choose a valid start and end to review captures.',
        requestedAt: '',
      };
      return;
    }
    const editingId = context === 'edit' ? this.editingSleep?.id : undefined;
    // Keep the original ISO strings here. Converting through JavaScript Date
    // truncates SQLite's microseconds to milliseconds and can drop a capture
    // whose timestamp is exactly equal to the segment boundary.
    const rangeStart = context === 'inferred-awake'
      ? this.manualFrameReviewBounds?.startedAt ?? range.start.toISOString()
      : this.editSleepFrameBounds?.startedAt ?? range.start.toISOString();
    const rangeEnd = context === 'inferred-awake'
      ? this.manualFrameReviewBounds?.endedAt ?? range.end.toISOString()
      : this.editSleepFrameBounds?.endedAt ?? range.end.toISOString();
    const locationId = context === 'inferred-awake'
      ? this.manualFrameReviewBounds?.locationId ?? undefined
      : this.editingSleep?.locationId;
    this.frameReview = { point, frames: [], index: 0, loading: true, error: '', requestedAt: requested.toISOString() };
    try {
      const frames = await api.getFramesBetween(
        rangeStart,
        rangeEnd,
        locationId,
      );
      if (
        this.frameReviewContext !== context
        || (context === 'edit' && this.editingSleep?.id !== editingId)
        || requestId !== this.frameReviewRequest
      ) return;
      this.frameReview = {
        point,
        frames,
        index: this.frameIndexForPoint(frames, point),
        loading: false,
        error: '',
        requestedAt: requested.toISOString(),
      };
    } catch (error) {
      this.frameReview = {
        point, frames: [], index: 0, loading: false,
        error: error instanceof Error ? error.message : 'No se pudo cargar el frame.',
        requestedAt: requested.toISOString(),
      };
    }
  }

  private stepFrameReview(direction: number): void {
    const index = Math.max(0, Math.min(this.frameReview.frames.length - 1, this.frameReview.index + direction));
    this.frameReview = { ...this.frameReview, index };
  }

  private selectFrameReviewIndex(index: number): void {
    const bounded = Math.max(0, Math.min(this.frameReview.frames.length - 1, index));
    this.frameReview = { ...this.frameReview, index: bounded };
  }

  private isHomeAssistantContext(): boolean {
    const path = window.location.pathname;
    return window.self !== window.top || path.startsWith('/baby-monitor-proxy/') || path.includes('/api/hassio_ingress/');
  }

  private returnToHomeAssistant(): void {
    const home = new URL('/', window.location.origin).href;
    try {
      if (window.top && window.top !== window) {
        window.top.location.assign(home);
        return;
      }
    } catch {
      // A Home Assistant iframe can be cross-origin. The click still has a
      // safe same-origin fallback below.
    }
    window.location.assign(home);
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
        this.page = 'sleep';
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
      // Labeling the new frame can open or close a sleep event. Refresh the
      // complete operational state immediately instead of leaving the card and
      // rhythm stale until the next 30-second poll.
      await this.loadOperationalData(false);
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
      await this.loadOperationalData(false);
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
    const end = this.manualForm.endedAt ? new Date(this.manualForm.endedAt) : null;
    if (Number.isNaN(start.getTime()) || (end && (Number.isNaN(end.getTime()) || end <= start))) {
      this.inlineError = this.t('invalidSleepRange');
      return;
    }
    if (this.manualForm.kind === 'awake' && !end) {
      this.inlineError = this.language === 'es'
        ? 'Los despertares necesitan una hora de fin.'
        : 'Awake periods need an end time.';
      return;
    }
    this.sleepBusy = 'add';
    try {
      await api.addManualSleep({
        startedAt: asIso(this.manualForm.startedAt),
        endedAt: this.manualForm.endedAt ? asIso(this.manualForm.endedAt) : null,
        kind: this.manualForm.kind,
        notes: this.manualForm.notes,
        details: this.manualForm.details,
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
    const nav: Array<{ page?: AppPage; label: TranslationKey; itemIcon: IconName; add?: boolean }> = [
      { page: 'sleep', label: 'navDashboard', itemIcon: 'home' },
      { page: 'data', label: 'navStatistics', itemIcon: 'history' },
      { label: 'navAdd', itemIcon: 'plus', add: true },
      { page: 'camera', label: 'navCamera', itemIcon: 'camera' },
      { page: 'settings', label: 'navSettings', itemIcon: 'settings' },
    ];
    const inHomeAssistant = this.isHomeAssistantContext();
    return html`
      <header class="app-header">
        <div class="brand-cluster">
          ${inHomeAssistant ? html`
            <button class="ha-exit" type="button" aria-label=${this.t('returnHomeAssistant')} title=${this.t('returnHomeAssistant')} @click=${() => this.returnToHomeAssistant()}>
              ${icon('chevron', 19)}<span>Home Assistant</span>
            </button>
          ` : nothing}
          <a class="brand" href="#main" @click=${(event: Event) => event.preventDefault()}>
            <span class="brand-mark">${icon('baby', 22)}</span>
            <span><strong>${this.t('brand')}</strong><small>${this.t('brandSuffix')}</small></span>
          </a>
        </div>
        <nav class="primary-nav" aria-label="Primary">
          ${nav.map(({ page, label, itemIcon, add }) => html`
            <button
              class=${`${page && this.page === page ? 'active' : ''}${add ? ' nav-add' : ''}`}
              @click=${() => add ? this.openManualForm() : page && this.setPage(page)}
              aria-current=${page && this.page === page ? 'page' : nothing}
              aria-label=${this.t(label)}
              aria-haspopup=${add ? 'dialog' : nothing}
              aria-expanded=${add ? String(this.manualOpen) : nothing}
            >
              ${icon(itemIcon, add ? 22 : 18)}<span>${this.t(label)}</span>
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
    return html`
      <main class="page dashboard-page" id="main">
        ${this.renderDailyRhythm()}

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
            <button class="text-button" @click=${() => this.setPage('data')}>${this.t('viewRhythm')} ${icon('chevron', 15)}</button>
          </div>
          ${this.renderSleepList(this.sleepEvents.slice(0, 4))}
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
    const liveStatus = this.liveTransport === 'webrtc'
      ? 'liveLowLatency'
      : this.liveTransport === 'mjpeg'
        ? 'liveFallback'
        : 'liveConnecting';
    return html`
      <article class="camera-card">
        <div class="camera-visual">
          ${frame?.imageUrl
            ? html`<img class="camera-live-poster" src=${frame.imageUrl} alt=${this.t('imageAlt')}>`
            : html`<div class="camera-placeholder">${icon('camera', 34)}<span>${this.t('cameraRefreshing')}</span></div>`}
          ${this.liveView
            ? this.liveTransport === 'mjpeg'
              ? html`<img class="camera-live-stream" src=${api.liveCameraUrl()} alt=${this.t('imageAlt')} @error=${() => { this.stopLiveView(); this.showToast(this.t('liveUnavailable'), 'error'); }}>`
              : html`<video class=${`camera-live-video ${this.liveTransport === 'webrtc' ? 'ready' : ''}`} autoplay muted playsinline aria-label=${this.t('imageAlt')} @playing=${() => this.handleWebRtcPlaying()}></video>`
            : nothing}
          <div class="camera-overlay">
            <span class=${this.liveView ? 'live-badge active' : 'live-badge'}>${this.liveView ? html`<i></i> LIVE` : this.t('snapshot')}</span>
            ${this.liveView
              ? html`<small>${this.t(liveStatus)}</small>`
              : frame ? html`<small>${formatRelative(frame.capturedAt, this.language)}</small>` : nothing}
          </div>
        </div>
        <div class="camera-body">
          <div class="camera-heading"><div><span>${this.t('cameraTitle')}</span><strong>${this.liveView ? this.t(liveStatus) : frame ? this.t('latestCapture', { time: formatClock(frame.capturedAt, this.language) }) : this.t('noVisionLabel')}</strong></div></div>
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
            <button class=${`button compact ${this.liveView ? 'active' : 'secondary'}`} @click=${() => { void this.toggleLiveView(); }}>
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

  private renderCamera(): TemplateResult {
    return html`
      <main class="page camera-page" id="main">
        <section class="page-heading">
          <div><span class="eyebrow">${this.t('navCamera')}</span><h1>${this.t('cameraPageTitle')}</h1><p>${this.t('cameraPageIntro')}</p></div>
          <button class="icon-button" aria-label=${this.t('refresh')} ?disabled=${this.refreshingData} @click=${() => this.loadOperationalData(true)}>
            <span class=${this.refreshingData ? 'spin' : ''}>${icon('refresh', 19)}</span>
          </button>
        </section>
        <section class="camera-focus-grid">
          ${this.renderCameraCard()}
          <div class="camera-signals">
            <article class=${`signal-card ${this.summary.cryActive ? 'alert' : ''}`}>
              <span class="signal-icon">${icon('waves', 20)}</span>
              <div><span>${this.t('cryStatus')}</span><strong>${this.health?.background.errors.cry ? this.t('monitorAttention') : this.settings.cry.mode === 'disabled' ? this.t('cryDisabled') : this.summary.cryActive ? this.t('cryActive') : this.t('allQuiet')}</strong></div>
              <small>${this.summary.lastCryAt ? this.t('lastCry', { time: formatRelative(this.summary.lastCryAt, this.language) }) : '—'}</small>
            </article>
            <article class="signal-card">
              <span class="signal-icon">${icon('sparkle', 20)}</span>
              <div><span>${this.t('visionLabel')}</span><strong>${this.summary.latestFrame?.label?.description || this.t('noVisionLabel')}</strong></div>
              <small>${this.summary.latestFrame ? formatRelative(this.summary.latestFrame.capturedAt, this.language) : '—'}</small>
            </article>
          </div>
        </section>
        <section class="frame-section camera-moments">
          <div class="section-heading"><div><span class="eyebrow">${this.t('recentRhythm')}</span><h2>${this.t('imageTimeline')}</h2></div><button class="text-button" @click=${() => this.setPage('data')}>${this.t('navStatistics')} ${icon('chevron', 15)}</button></div>
          ${this.frames.length ? html`<div class="frame-grid">${this.frames.slice(0, 12).map((frame) => this.renderFrame(frame))}</div>` : html`<div class="empty-state">${icon('camera', 26)}<p>${this.t('noFrames')}</p></div>`}
        </section>
      </main>
    `;
  }

  private renderNightRibbon(): TemplateResult {
    const now = Date.now();
    const duration = 12 * 60 * 60_000;
    const start = now - duration;
    const percent = (value: number): number => Math.max(0, Math.min(100, ((value - start) / duration) * 100));
    const sleeps = this.sleepEvents.filter((event) => event.kind !== 'awake'
      && new Date(event.endedAt ?? now).getTime() >= start
      && new Date(event.startedAt).getTime() <= now);
    const cries = this.cryEvents.filter((event) => new Date(event.detectedAt).getTime() >= start);
    const frames = this.frames.filter((frame) => new Date(frame.capturedAt).getTime() >= start).slice(0, 18);
    return html`
      <section class="night-ribbon-card">
        <div class="ribbon-heading">
          <div><span>${icon('moon', 18)} ${this.t('nightRibbon')}</span><small>${this.t('nightRibbonHint')}</small></div>
          <div class="ribbon-legend"><span class="sleep-dot">${this.t('ribbonSleep')}</span><span class="cry-dot">${this.t('ribbonCry')}</span><span class="frame-dot">${this.t('ribbonFrame')}</span></div>
        </div>
        <button class="ribbon-track" @click=${() => this.setPage('data')} aria-label=${this.t('viewRhythm')}>
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
    const tomorrow = shiftDateKey(today, 1);
    if (this.rhythmDate >= today) {
      return Array.from({ length: 7 }, (_, index) => shiftDateKey(tomorrow, index - 6));
    }
    return Array.from({ length: 7 }, (_, index) => shiftDateKey(this.rhythmDate, index - 3));
  }

  private moveRhythmDate(days: number): void {
    const today = localDateKey(new Date());
    const tomorrow = shiftDateKey(today, 1);
    const next = shiftDateKey(this.rhythmDate, days);
    this.rhythmDate = next > tomorrow ? tomorrow : next;
  }

  private rhythmMarkerSegments(segments: RhythmSegment[]): RhythmSegment[] {
    const eventIds = new Set<string>();
    const candidates = segments.filter((item) => {
      const midpoint = (item.startRatio + item.endRatio) / 2;
      if (midpoint < 0.055 || midpoint > 0.945) return false;
      if (item.predicted || item.type === 'awake' || !item.event) return true;
      if (eventIds.has(item.event.id)) return false;
      eventIds.add(item.event.id);
      return true;
    }).sort((a, b) => ((a.startRatio + a.endRatio) - (b.startRatio + b.endRatio)));
    const visible: RhythmSegment[] = [];
    for (const item of candidates) {
      const midpoint = (item.startRatio + item.endRatio) / 2;
      const previous = visible.at(-1);
      const previousMidpoint = previous ? (previous.startRatio + previous.endRatio) / 2 : -1;
      if (previous && midpoint - previousMidpoint < 0.052) {
        if (item.type === 'awake' && previous.type !== 'awake') visible[visible.length - 1] = item;
        continue;
      }
      visible.push(item);
    }
    return visible;
  }

  private openRhythmSegment(segment: RhythmSegment): void {
    if (segment.prediction) {
      this.selectedPrediction = segment.prediction;
      return;
    }
    if (segment.event) {
      this.openSleepEditor(segment.event);
      return;
    }
    if (segment.type !== 'awake') return;
    this.inlineError = '';
    this.manualForm = {
      startedAt: localDateTime(segment.start),
      endedAt: localDateTime(segment.end),
      kind: 'awake',
      notes: '',
      details: EMPTY_DETAILS(),
    };
    this.frameReviewContext = 'inferred-awake';
    this.manualFrameReviewBounds = {
      startedAt: segment.evidenceStartedAt,
      endedAt: segment.evidenceEndedAt,
      locationId: segment.locationId,
    };
    this.frameReview = { point: '', frames: [], index: 0, loading: false, error: '', requestedAt: '' };
    this.manualEndTouched = true;
    this.manualOpen = true;
    void this.loadFrameReview('start');
  }

  private renderDailyRhythm(): TemplateResult {
    const today = localDateKey(new Date());
    const tomorrow = shiftDateKey(today, 1);
    const model = buildRhythmModel(this.sleepEvents, this.rhythmDate, this.rhythmMode, new Date(), this.sleepPlan);
    const selectedDate = this.rhythmDateValue(this.rhythmDate);
    const locale = this.language === 'es' ? 'es-ES' : 'en-GB';
    const titleDate = new Intl.DateTimeFormat(locale, { weekday: 'long', day: 'numeric', month: 'long' }).format(selectedDate);
    const coreDate = new Intl.DateTimeFormat(locale, { weekday: 'short', day: 'numeric' }).format(selectedDate);
    const dayKeys = this.rhythmDayKeys();
    const napCount = new Set(model.sleepSegments.filter((segment) => segment.type === 'nap').map((segment) => segment.event?.id ?? segment.id)).size;
    const nightCount = new Set(model.sleepSegments.filter((segment) => segment.type === 'night').map((segment) => segment.event?.id ?? segment.id)).size;
    const averageNap = napCount ? Math.round(model.napMinutes / napCount) : 0;
    const lastDay = dayKeys.at(-1) ?? this.rhythmDate;
    const markerSegments = this.rhythmMarkerSegments(model.segments);
    const startPosition = rhythmPosition(0);
    const endPosition = rhythmPosition(1);
    const midnightInner = model.midnightRatio == null ? null : rhythmSvgPoint(model.midnightRatio, 88);
    const midnightOuter = model.midnightRatio == null ? null : rhythmSvgPoint(model.midnightRatio, 145);
    const forecasts = [...new Map(model.predictedSegments
      .filter((segment) => segment.prediction)
      .map((segment) => [segment.prediction!.recommendedStart, segment.prediction!])).values()];
    const recordedCount = new Set(model.sleepSegments.map((segment) => segment.event?.id ?? segment.id)).size;
    const babyName = this.settings.baby.name.trim();
    const rhythmContext = `${babyName ? `${babyName} · ` : ''}${this.t(this.rhythmMode === 'night' ? 'rhythmNight' : 'rhythmDay')}`;

    return html`
      <section class=${`rhythm-visual-card ${this.rhythmMode}`} aria-label=${this.t('rhythmVisualTitle')}>
        <header class="rhythm-visual-head">
          <div>
            <span class="eyebrow rhythm-context">${rhythmContext}</span>
            <h2>${titleDate}</h2>
          </div>
          <div class="rhythm-date-nav">
            <button class="icon-button small rhythm-prev" aria-label=${this.t('rhythmPreviousDays')} @click=${() => this.moveRhythmDate(-7)}>${icon('chevron', 17)}</button>
            <button class="text-button rhythm-today" ?disabled=${this.rhythmDate === today} @click=${() => { this.rhythmDate = today; }}>${this.t('rhythmToday')}</button>
            <button class="icon-button small" aria-label=${this.t('rhythmNextDays')} ?disabled=${lastDay >= tomorrow} @click=${() => this.moveRhythmDate(7)}>${icon('chevron', 17)}</button>
            <button class="icon-button small rhythm-refresh" title=${this.t('refreshData')} aria-label=${this.t('refreshData')} ?disabled=${this.refreshingData} @click=${() => this.loadOperationalData(true)}>
              <span class=${this.refreshingData ? 'spin' : ''}>${icon('refresh', 16)}</span>
            </button>
          </div>
        </header>

        <div class="rhythm-week" aria-label=${this.t('rhythmChooseDay')}>
          ${dayKeys.map((dateKey) => {
            const date = this.rhythmDateValue(dateKey);
            const future = dateKey > tomorrow;
            const selected = dateKey === this.rhythmDate;
            const isToday = dateKey === today;
            const isTomorrow = dateKey === tomorrow;
            return html`
              <button
                class=${`rhythm-day ${selected ? 'selected' : ''} ${isToday ? 'today' : ''}`}
                ?disabled=${future}
                aria-pressed=${selected}
                @click=${() => { this.rhythmDate = dateKey; }}
              >
                <span>${new Intl.DateTimeFormat(locale, { weekday: 'short' }).format(date)}</span>
                <strong>${date.getDate()}</strong>
                <small>${isToday ? this.t('rhythmToday') : isTomorrow ? (this.language === 'es' ? 'mañana' : 'tomorrow') : new Intl.DateTimeFormat(locale, { month: 'short' }).format(date)}</small>
              </button>
            `;
          })}
        </div>

        <div class="rhythm-orbit-wrap">
          <div class="rhythm-orbit" aria-label=${this.t('rhythmRecordedSleep', { duration: formatDuration(model.totalMinutes) })}>
            <svg class="rhythm-ring" viewBox="0 0 320 320" aria-hidden="true">
              <path class="rhythm-ring-track" d=${rhythmTrackPath(122)}></path>
              <path class="rhythm-ring-inner" d=${rhythmTrackPath(82)}></path>
              ${midnightInner && midnightOuter ? svg`<line class="rhythm-midnight-line" x1=${midnightInner.x} y1=${midnightInner.y} x2=${midnightOuter.x} y2=${midnightOuter.y}></line>` : nothing}
              ${model.segments.map((segment) => svg`
                <path
                  class=${`rhythm-arc ${segment.type === 'awake' ? 'awake' : segment.type === 'night' ? 'night-sleep' : 'nap'} ${segment.predicted ? 'predicted' : ''} ${segment.event && !segment.event.endedAt ? 'ongoing' : ''}`}
                  d=${rhythmArcPath(segment.startRatio, segment.endRatio)}
                ></path>
                ${segment.event || segment.prediction || segment.type === 'awake' ? svg`
                  <path
                    class="rhythm-hit"
                    d=${rhythmArcPath(segment.startRatio, segment.endRatio)}
                    @click=${() => this.openRhythmSegment(segment)}
                  ></path>
                ` : nothing}
              `)}
            </svg>
            ${markerSegments.map((segment) => {
              const position = rhythmMarkerPosition(segment);
              const label = segment.predicted
                ? this.language === 'es' ? segment.type === 'night' ? 'Sueño previsto' : 'Siesta prevista' : 'Predicted sleep'
                : segment.type === 'awake' ? (this.language === 'es' ? 'Despertar por la noche' : 'Night waking') : this.t(segment.type === 'night' ? 'nightSleep' : 'nap');
              const detail = `${label} · ${formatClock(segment.prediction?.recommendedStart ?? segment.start.toISOString(), this.language)}${segment.predicted ? ` · ${this.language === 'es' ? 'previsto' : 'predicted'}` : `–${formatClock(segment.end.toISOString(), this.language)} · ${formatDuration(segment.minutes)}`}`;
              return html`
                <button class=${`rhythm-marker ${segment.type === 'awake' ? 'awake' : segment.type === 'night' ? 'night-sleep' : 'nap'} ${segment.predicted ? 'predicted' : ''}`} style=${`--x:${position.x}%;--y:${position.y}%`} title=${detail} aria-label=${detail} @click=${() => this.openRhythmSegment(segment)}>
                  ${icon(segment.predicted ? 'sparkle' : segment.type === 'awake' ? 'waves' : 'moon', 15)}
                </button>
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
            <div class="rhythm-endpoint start" style=${`--x:${startPosition.x}%;--y:${startPosition.y}%`}><span>${icon(this.rhythmMode === 'night' ? 'moon' : 'sun', 16)}</span><small>${this.t(this.rhythmMode === 'night' ? 'rhythmBed' : 'rhythmWake')}</small><strong>${model.bedAt && this.rhythmMode === 'night' ? formatClock(model.bedAt.toISOString(), this.language) : model.wakeAt && this.rhythmMode === 'day' ? formatClock(model.wakeAt.toISOString(), this.language) : '—'}${this.rhythmMode === 'night' ? model.bedPredicted ? ' · ~' : '' : model.wakePredicted ? ' · ~' : ''}</strong></div>
            <div class="rhythm-endpoint end" style=${`--x:${endPosition.x}%;--y:${endPosition.y}%`}><span>${icon(this.rhythmMode === 'night' ? 'sun' : 'moon', 16)}</span><small>${this.t(this.rhythmMode === 'night' ? 'rhythmWake' : 'rhythmBed')}</small><strong>${model.wakeAt && this.rhythmMode === 'night' ? formatClock(model.wakeAt.toISOString(), this.language) : model.bedAt && this.rhythmMode === 'day' ? formatClock(model.bedAt.toISOString(), this.language) : '—'}${this.rhythmMode === 'night' ? model.wakePredicted ? ' · ~' : '' : model.bedPredicted ? ' · ~' : ''}</strong></div>
          </div>
        </div>

        <div class="rhythm-summary">
          <div class="rhythm-total"><span>${icon('moon', 20)}</span><div><small>${this.t('rhythmTotal')}</small><strong>${model.totalMinutes ? formatDuration(model.totalMinutes) : this.t('rhythmNoSleep')}</strong></div><b>${recordedCount}</b></div>
          <div class="rhythm-duration-track" aria-hidden="true">
            ${model.sleepSegments.map((segment) => html`<i class=${segment.type === 'night' ? 'night-sleep' : 'nap'} style=${`--width:${model.totalMinutes ? Math.max(4, segment.minutes / model.totalMinutes * 100) : 0}%`}></i>`)}
          </div>
          <div class="rhythm-stats">
            <div><span>${this.t('rhythmNaps')}</span><strong>${napCount} · ${formatDuration(model.napMinutes)}</strong></div>
            <div><span>${this.t('rhythmNightPeriods')}</span><strong>${nightCount} · ${formatDuration(model.nightMinutes)}</strong></div>
            <div><span>${this.t('rhythmAverageNap')}</span><strong>${formatDuration(averageNap)}</strong></div>
          </div>
        </div>
        ${forecasts.length ? html`
          <div class="rhythm-forecast" aria-label=${this.language === 'es' ? 'Sueño previsto' : 'Predicted sleep'}>
            <div class="rhythm-forecast-heading"><span>${icon('sparkle', 17)}</span><div><strong>${this.language === 'es' ? 'Previsto por el modelo' : 'Model forecast'}</strong><small>${this.rhythmDate === tomorrow ? (this.language === 'es' ? 'Plan de mañana' : 'Tomorrow plan') : (this.language === 'es' ? 'Plan de hoy' : 'Today plan')}</small></div></div>
            <div class="rhythm-forecast-list">${forecasts.map((target) => html`<button type="button" @click=${() => { this.selectedPrediction = target; }}><span>${icon(target.kind === 'night' ? 'moon' : 'sun', 17)}</span><div><strong>${this.language === 'es' ? target.kind === 'night' ? 'Sueño largo' : 'Siesta' : target.label}</strong><small>${formatClock(target.windowStart, this.language)}–${formatClock(target.windowEnd, this.language)}</small></div><b>${formatClock(target.recommendedStart, this.language)}</b>${icon('chevron', 14)}</button>`)}</div>
          </div>
        ` : nothing}
        ${!model.segments.length ? html`<p class="rhythm-empty">${this.t('rhythmEmptyHint')}</p>` : nothing}
      </section>
    `;
  }

  private renderSleepList(events: SleepEvent[]): TemplateResult {
    if (!events.length) return html`<div class="empty-state compact-empty">${icon('moon', 24)}<p>${this.t('noSleepEvents')}</p></div>`;
    return html`<div class="moment-list">${events.map((event) => html`
      <article class="moment-row">
        <span class=${`moment-symbol ${event.kind === 'awake' ? 'coral' : ''} ${event.endedAt ? '' : 'active'}`}>${icon(event.kind === 'awake' || !event.endedAt ? 'waves' : 'moon', 17)}</span>
        <div class="moment-main"><strong>${event.kind === 'awake' ? (this.language === 'es' ? 'Despertar' : 'Awake') : this.t(event.kind === 'night' ? 'nightSleep' : event.kind === 'nap' ? 'nap' : 'unknownType')}</strong><small>${formatDateTime(event.startedAt, this.language)} · ${this.t('location')}: ${event.locationId}${event.notes ? ` · ${event.notes}` : ''}</small></div>
        <div class="moment-meta"><strong>${sleepDuration(event.startedAt, event.endedAt)}</strong><small>${event.endedAt ? this.t(event.source === 'vision' ? 'vision' : event.source === 'import' ? 'imported' : event.source === 'automatic' ? 'automatic' : 'manual') : this.t('ongoing')}</small></div>
      </article>
    `)}</div>`;
  }

  private renderTemporalButton(
    label: string,
    value: string,
    target: TemporalTarget,
    fallbackValue: string,
    hint = '',
  ): TemplateResult {
    const date = value ? new Date(value) : null;
    const valid = Boolean(date && Number.isFinite(date.getTime()));
    const locale = this.language === 'es' ? 'es-ES' : 'en-GB';
    const dateLabel = valid && date
      ? new Intl.DateTimeFormat(locale, { weekday: 'short', day: 'numeric', month: 'short' }).format(date)
      : this.language === 'es' ? 'Sin finalizar' : 'Ongoing';
    const timeLabel = valid && date
      ? new Intl.DateTimeFormat(locale, { hour: '2-digit', minute: '2-digit' }).format(date)
      : '—';
    return html`
      <button
        class=${`sleep-time-card ${valid ? '' : 'open-ended'}`}
        type="button"
        @click=${() => this.openTemporalPicker(target, value || fallbackValue, label)}
      >
        <span>${label}</span>
        <strong>${timeLabel}</strong>
        <small>${dateLabel}${hint ? ` · ${hint}` : ''}</small>
      </button>
    `;
  }

  private renderDetailGroups(scope: 'manual' | 'edit', details: SleepEventDetails): TemplateResult {
    return html`${DETAIL_GROUPS.map((group) => html`
      <section class="sleep-detail-group">
        <h4>${this.language === 'es' ? group.es : group.en}</h4>
        <div class=${`sleep-detail-choices ${group.key === 'method' || group.key === 'mood' ? 'wide' : ''}`}>
          ${group.options.map(([tag, emoji, es, en]) => html`
            <button
              type="button"
              class=${details.tags.includes(tag) ? 'selected' : ''}
              aria-pressed=${details.tags.includes(tag)}
              @click=${() => this.toggleDetailTag(scope, tag)}
            ><span>${emoji}</span><strong>${this.language === 'es' ? es : en}</strong></button>
          `)}
        </div>
      </section>
    `)}`;
  }

  private renderPauses(
    scope: 'manual' | 'edit',
    pauses: SleepPause[],
  ): TemplateResult {
    const label = this.language === 'es' ? 'Pausas despierto' : 'Awake pauses';
    return html`
      <section class="sleep-detail-group sleep-pause-group">
        <div class="sleep-detail-heading"><h4>${label}</h4><button type="button" @click=${() => this.addPause(scope)}>${icon('plus', 15)} ${this.language === 'es' ? 'Añadir pausa' : 'Add pause'}</button></div>
        ${pauses.length ? html`<div class="sleep-pause-list">${pauses.map((pause, index) => html`
          <article class="sleep-pause-row">
            <span>${icon('waves', 17)}</span>
            <button type="button" @click=${() => this.openTemporalPicker(
              `${scope}-pause-start-${index}` as TemporalTarget,
              localDateTime(new Date(pause.startedAt)),
              this.language === 'es' ? 'Inicio de la pausa' : 'Pause start',
            )}><small>${this.language === 'es' ? 'Inicio' : 'Start'}</small><strong>${formatClock(pause.startedAt, this.language)}</strong></button>
            <i>–</i>
            <button type="button" @click=${() => this.openTemporalPicker(
              `${scope}-pause-end-${index}` as TemporalTarget,
              localDateTime(new Date(pause.endedAt)),
              this.language === 'es' ? 'Fin de la pausa' : 'Pause end',
            )}><small>${this.language === 'es' ? 'Fin' : 'End'}</small><strong>${formatClock(pause.endedAt, this.language)}</strong></button>
            <button type="button" class="pause-remove" aria-label=${this.t('remove')} @click=${() => this.removePause(scope, index)}>&times;</button>
          </article>
        `)}</div>` : html`<p class="sleep-detail-empty">${this.language === 'es' ? 'Sin pausas dentro de este tramo.' : 'No awake pauses inside this segment.'}</p>`}
      </section>
    `;
  }

  private renderFrameReview(): TemplateResult {
    const inferredAwake = this.frameReviewContext === 'inferred-awake';
    const current = this.frameReview.frames[this.frameReview.index];
    const requested = this.frameReview.requestedAt ? new Date(this.frameReview.requestedAt) : null;
    const captured = current ? new Date(current.capturedAt) : null;
    const delta = requested && captured
      ? Math.round(Math.abs(captured.getTime() - requested.getTime()) / 60_000)
      : null;
    const humanValue = (value: string | boolean | null | string[]): string => {
      if (Array.isArray(value)) return value.length ? value.join(', ') : '—';
      if (value == null || value === 'unknown') return this.language === 'es' ? 'Sin determinar' : 'Unknown';
      if (typeof value === 'boolean') return value
        ? (this.language === 'es' ? 'Sí' : 'Yes')
        : (this.language === 'es' ? 'No' : 'No');
      const labels: Record<string, [string, string]> = {
        awake: ['Despierto', 'Awake'],
        asleep: ['Dormido', 'Asleep'],
        uncertain: ['Incierto', 'Uncertain'],
        yes: ['Sí', 'Yes'],
        no: ['No', 'No'],
        left: ['Izquierda', 'Left'],
        right: ['Derecha', 'Right'],
        back: ['Boca arriba', 'On back'],
        face_down: ['Boca abajo', 'Face down'],
      };
      return labels[value]?.[this.language === 'es' ? 0 : 1] ?? value.replaceAll('_', ' ');
    };
    const modelMetadata = current?.label ? [
      [this.language === 'es' ? 'Proveedor' : 'Provider', current.provider || '—'],
      [this.language === 'es' ? 'Modelo' : 'Model', current.model || '—'],
      [this.language === 'es' ? 'Estado' : 'State', humanValue(current.label.state)],
      [this.language === 'es' ? 'Confianza' : 'Confidence', `${Math.round(current.label.confidence * 100)}%`],
      [this.language === 'es' ? 'Bebé presente' : 'Baby present', humanValue(current.label.babyPresent)],
      [this.language === 'es' ? 'En la cuna' : 'In crib', humanValue(current.label.inCrib)],
      [this.language === 'es' ? 'Cara visible' : 'Face visible', humanValue(current.label.faceVisible)],
      [this.language === 'es' ? 'Orientación de la cabeza' : 'Head side', humanValue(current.label.headSide)],
      [this.language === 'es' ? 'Posición del cuerpo' : 'Body position', humanValue(current.label.bodyPosition)],
      [this.language === 'es' ? 'Ropa' : 'Clothing', humanValue(current.label.clothingItems)],
      [this.language === 'es' ? 'Chupete' : 'Pacifier', humanValue(current.label.pacifier)],
      [this.language === 'es' ? 'Boca abierta' : 'Mouth open', humanValue(current.label.mouthOpen)],
      [this.language === 'es' ? 'Etiquetas' : 'Tags', humanValue(current.label.tags)],
    ] : [];
    return html`
      <section class="editor-frames">
        <div class="editor-frames-heading">
          <div>
            <h4>${inferredAwake
              ? (this.language === 'es' ? 'Frames del despertar nocturno' : 'Night waking frames')
              : (this.language === 'es' ? 'Imágenes del segmento' : 'Images in this segment')}</h4>
            ${inferredAwake && this.manualFrameReviewBounds ? html`<p>${this.language === 'es'
              ? `Todas las capturas entre ${formatClock(this.manualFrameReviewBounds.startedAt, this.language)} y ${formatClock(this.manualFrameReviewBounds.endedAt, this.language)} para comprobar la detección.`
              : `Every capture from ${formatClock(this.manualFrameReviewBounds.startedAt, this.language)} to ${formatClock(this.manualFrameReviewBounds.endedAt, this.language)} so you can verify the detection.`}</p>` : nothing}
          </div>
          ${this.frameReview.frames.length ? html`<span>${this.frameReview.frames.length} ${this.language === 'es' ? 'capturas' : 'captures'}</span>` : nothing}
        </div>
        <div class="frame-point-switch">
          ${(['start', 'middle', 'end'] as const).map((point) => html`<button type="button" class=${this.frameReview.point === point ? 'active' : ''} ?disabled=${this.frameReview.loading} @click=${() => this.selectFrameReviewPoint(point)}>${icon(point === 'middle' ? 'eye' : 'clock', 17)}<span>${this.language === 'es' ? ({ start: 'Inicio', middle: 'Mitad', end: 'Fin' }[point]) : ({ start: 'Start', middle: 'Middle', end: 'End' }[point])}</span></button>`)}
        </div>
        ${this.frameReview.loading ? html`<div class="frame-review-empty"><span class="spinner"></span> ${this.language === 'es' ? 'Cargando todas las imágenes del segmento…' : 'Loading every image in this segment…'}</div>` : this.frameReview.error ? html`<div class="frame-review-empty error">${this.frameReview.error}</div>` : current ? html`
          <article class="frame-review-card">
            ${current.imageAvailable ? html`<img src=${current.imageUrl} alt=${this.language === 'es' ? 'Captura del segmento' : 'Capture in this segment'}>` : html`<div class="frame-review-missing">${icon('camera', 24)}</div>`}
            <div class="frame-review-copy">
              <div><strong>${formatDateTime(current.capturedAt, this.language)}</strong><span>${delta == null ? '' : delta === 0 ? (this.language === 'es' ? 'mismo minuto' : 'same minute') : `${delta} min`}</span></div>
              <p>${current.label?.description || (this.language === 'es' ? 'Sin lectura de IA para este frame.' : 'No AI reading for this frame.')}</p>
              ${current.label ? html`<div class="frame-labels"><span>${humanValue(current.label.state)}</span><span>${Math.round(current.label.confidence * 100)}%</span><span>${current.label.inCrib == null ? '—' : current.label.inCrib ? (this.language === 'es' ? 'en cuna' : 'in crib') : (this.language === 'es' ? 'fuera de cuna' : 'out of crib')}</span></div>` : nothing}
              ${current.label ? html`
                <details class="frame-model-details">
                  <summary>${icon('chevron', 15)}<span>${this.language === 'es' ? 'Ver análisis del modelo' : 'View model analysis'}</span></summary>
                  <div class="frame-model-metadata">
                    ${modelMetadata.map(([label, value]) => html`<div><span>${label}</span><strong>${value}</strong></div>`)}
                  </div>
                </details>
              ` : nothing}
              ${this.frameReview.frames.length > 1 ? html`<input class="frame-scrubber" type="range" min="0" max=${String(this.frameReview.frames.length - 1)} .value=${String(this.frameReview.index)} aria-label=${this.language === 'es' ? 'Recorrer capturas del segmento' : 'Browse segment captures'} @input=${(event: Event) => this.selectFrameReviewIndex(Number(inputValue(event)))}>` : nothing}
              <div class="frame-stepper"><button type="button" aria-label=${this.language === 'es' ? 'Captura anterior' : 'Previous capture'} ?disabled=${this.frameReview.index === 0} @click=${() => this.stepFrameReview(-1)}>${icon('chevron', 18)}</button><span><strong>${this.frameReview.index + 1} / ${this.frameReview.frames.length}</strong><small>${this.language === 'es' ? 'capturas del segmento' : 'segment captures'}</small></span><button type="button" aria-label=${this.language === 'es' ? 'Captura siguiente' : 'Next capture'} ?disabled=${this.frameReview.index >= this.frameReview.frames.length - 1} @click=${() => this.stepFrameReview(1)}>${icon('chevron', 18)}</button></div>
            </div>
          </article>
        ` : html`<div class="frame-review-empty">${this.language === 'es' ? 'No hay imágenes guardadas dentro de este segmento.' : 'There are no stored images inside this segment.'}</div>`}
      </section>
    `;
  }

  private renderTemporalPicker(): TemplateResult | typeof nothing {
    if (!this.temporalPicker) return nothing;
    const selected = new Date(this.temporalPicker.value);
    const locale = this.language === 'es' ? 'es-ES' : 'en-GB';
    const hours = Array.from({ length: 24 }, (_, index) => index);
    const minutes = Array.from({ length: 60 }, (_, index) => index);
    const days = Array.from({ length: 7 }, (_, index) => {
      const day = new Date(selected);
      day.setDate(day.getDate() + index - 3);
      return day;
    });
    return html`
      <div class=${`temporal-backdrop ${this.rhythmMode === 'day' ? 'theme-day' : ''}`} @click=${(event: Event) => { if (event.target === event.currentTarget) this.temporalPicker = null; }}>
        <section class="temporal-picker" role="dialog" aria-modal="true" aria-label=${this.temporalPicker.title}>
          <div class="sheet-handle"></div>
          <header><button type="button" @click=${() => { this.temporalPicker = null; }}>&times;</button><div><small>${this.temporalPicker.title}</small><strong>${new Intl.DateTimeFormat(locale, { weekday: 'long', day: 'numeric', month: 'long' }).format(selected)}</strong></div><button type="button" class="temporal-done" @click=${() => this.applyTemporalPicker()}>${this.language === 'es' ? 'Listo' : 'Done'}</button></header>
          <div class="temporal-days">${days.map((day) => {
            const active = localDateKey(day) === localDateKey(selected);
            return html`<button type="button" class=${active ? 'active' : ''} @click=${() => {
              const value = new Date(selected);
              value.setFullYear(day.getFullYear(), day.getMonth(), day.getDate());
              this.temporalPicker = this.temporalPicker ? { ...this.temporalPicker, value: localDateTime(value) } : null;
            }}><span>${new Intl.DateTimeFormat(locale, { weekday: 'narrow' }).format(day)}</span><strong>${day.getDate()}</strong></button>`;
          })}</div>
          <div class="temporal-date-jump"><button type="button" @click=${() => this.updateTemporalDate(-7)}>${icon('chevron', 16)} ${this.language === 'es' ? '7 días' : '7 days'}</button><button type="button" @click=${() => {
            const now = new Date();
            now.setHours(selected.getHours(), selected.getMinutes(), 0, 0);
            this.temporalPicker = this.temporalPicker ? { ...this.temporalPicker, value: localDateTime(now) } : null;
          }}>${this.language === 'es' ? 'Hoy' : 'Today'}</button><button type="button" @click=${() => this.updateTemporalDate(7)}>${this.language === 'es' ? '7 días' : '7 days'} ${icon('chevron', 16)}</button></div>
          <div class="temporal-clock" aria-label=${this.language === 'es' ? 'Hora y minutos' : 'Hours and minutes'}>
            <div><small>${this.language === 'es' ? 'Hora' : 'Hour'}</small><div class="temporal-column">${hours.map((hour) => html`<button type="button" class=${`temporal-option ${selected.getHours() === hour ? 'active' : ''}`} @click=${() => this.updateTemporalClock('hour', hour)}>${String(hour).padStart(2, '0')}</button>`)}</div></div>
            <b>:</b>
            <div><small>${this.language === 'es' ? 'Minuto' : 'Minute'}</small><div class="temporal-column">${minutes.map((minute) => html`<button type="button" class=${`temporal-option ${selected.getMinutes() === minute ? 'active' : ''}`} @click=${() => this.updateTemporalClock('minute', minute)}>${String(minute).padStart(2, '0')}</button>`)}</div></div>
          </div>
          <div class="temporal-quick">${[-15, -5, 5, 15].map((minutes) => html`<button type="button" @click=${() => this.adjustTemporal(minutes)}>${minutes > 0 ? '+' : ''}${minutes} min</button>`)}</div>
          <button type="button" class="button primary temporal-confirm" @click=${() => this.applyTemporalPicker()}>${this.language === 'es' ? 'Usar esta fecha y hora' : 'Use this date and time'}</button>
        </section>
      </div>
    `;
  }

  private renderPredictionDialog(): TemplateResult | typeof nothing {
    const target = this.selectedPrediction;
    if (!target) return nothing;
    const calculation = target.calculation;
    const model = this.sleepPlan?.modelDetails;
    const subject = this.settings.baby.name.trim() || (this.language === 'es' ? 'El bebé' : 'The baby');
    const spanish = this.language === 'es';
    const historyPercent = model ? Math.round(model.wakeWindows.historyWeight * 100) : 0;
    const agePercent = 100 - historyPercent;
    const ageLabels: Record<string, string> = {
      unknown: spanish ? 'edad sin configurar' : 'age not configured',
      '0-3m': spanish ? '0–3 meses' : '0–3 months',
      '4-5m': spanish ? '4–5 meses' : '4–5 months',
      '6-8m': spanish ? '6–8 meses' : '6–8 months',
      '9-11m': spanish ? '9–11 meses' : '9–11 months',
      '12-17m': spanish ? '12–17 meses' : '12–17 months',
      '18-23m': spanish ? '18–23 meses' : '18–23 months',
      '24m+': spanish ? '24 meses o más' : '24 months or older',
    };
    const ageLabel = model ? ageLabels[model.baseline.ageBand] ?? model.baseline.ageBand : '';
    const anchorLabels: Record<string, string> = spanish ? {
      last_observed_wake: 'Último despertar observado',
      typical_morning_wake: 'Despertar habitual de la mañana',
      previous_predicted_nap_end: 'Fin previsto de la siesta anterior',
      recent_bedtime_median: 'Mediana de acostarse reciente',
      age_guidance: 'Referencia por edad',
    } : {
      last_observed_wake: 'Last observed wake-up',
      typical_morning_wake: 'Typical morning wake-up',
      previous_predicted_nap_end: 'Predicted end of the previous nap',
      recent_bedtime_median: 'Recent median bedtime',
      age_guidance: 'Age guidance',
    };
    const anchorLabel = calculation ? anchorLabels[calculation.anchorType] : '';
    const learnedWake = model?.wakeWindows.medianMinutes;
    const wakeRange = model?.wakeWindows.minMinutes != null && model.wakeWindows.maxMinutes != null
      ? `${formatDuration(model.wakeWindows.minMinutes)}–${formatDuration(model.wakeWindows.maxMinutes)}`
      : '—';
    const wakeSpread = model?.wakeWindows.minMinutes != null && model.wakeWindows.maxMinutes != null
      ? model.wakeWindows.maxMinutes - model.wakeWindows.minMinutes
      : 0;
    const variabilityWarning = wakeSpread >= 180
      ? (spanish ? ' Hay mucha dispersión: conviene revisar las muestras extremas.' : ' The spread is wide, so the extreme samples are worth reviewing.')
      : '';
    const pattern = target.kind === 'nap'
      ? learnedWake != null && model
        ? spanish
          ? `La mediana reciente de ${subject} es ${formatDuration(learnedWake)} (rango observado ${wakeRange}). El historial pesa ahora un ${historyPercent}% y la referencia por edad un ${agePercent}%.${variabilityWarning}`
          : `${subject}'s recent median is ${formatDuration(learnedWake)} (observed range ${wakeRange}). Personal history currently weighs ${historyPercent}% and age guidance ${agePercent}%.${variabilityWarning}`
        : spanish
          ? `Todavía no hay tres ventanas despierto válidas. Esta hora se apoya principalmente en la referencia por edad y se irá personalizando.`
          : `There are not yet three valid wake windows. This time relies mainly on age guidance and will personalize as history grows.`
      : model
        ? spanish
          ? `${subject} suele acostarse alrededor de ${this.formatPredictionMinute(model.bedtimes.medianMinuteOfDay)} y despertarse sobre las ${this.formatPredictionMinute(model.morningWakes.medianMinuteOfDay)}. La duración nocturna agrupada tiene una mediana de ${formatDuration(model.nightDurations.finalMinutes)}.`
          : `${subject} usually goes to bed around ${this.formatPredictionMinute(model.bedtimes.medianMinuteOfDay)} and wakes around ${this.formatPredictionMinute(model.morningWakes.medianMinuteOfDay)}. Grouped night sleep has a median duration of ${formatDuration(model.nightDurations.finalMinutes)}.`
        : target.explanation;
    return html`
      <div class=${`manual-dialog-backdrop prediction-backdrop ${this.rhythmMode === 'day' ? 'theme-day' : ''}`} @click=${(event: Event) => { if (event.target === event.currentTarget) this.selectedPrediction = null; }}>
        <section class="manual-dialog prediction-dialog" role="dialog" aria-modal="true">
          <button class="dialog-close" type="button" @click=${() => { this.selectedPrediction = null; }}>&times;</button>
          <span class="prediction-dialog-icon">${icon(target.kind === 'night' ? 'moon' : 'sun', 28)}</span>
          <small>${this.language === 'es' ? 'Predicción del modelo' : 'Model prediction'}</small>
          <h2>${this.language === 'es' ? target.kind === 'night' ? 'Sueño largo previsto' : 'Siesta prevista' : target.label}</h2>
          <div class="prediction-dialog-time"><div><span>${this.language === 'es' ? 'Inicio previsto' : 'Expected start'}</span><strong>${formatClock(target.recommendedStart, this.language)}</strong></div><i>–</i><div><span>${this.language === 'es' ? 'Fin estimado' : 'Estimated end'}</span><strong>${formatClock(new Date(new Date(target.recommendedStart).getTime() + target.durationMinutes * 60_000).toISOString(), this.language)}</strong></div></div>
          <div class="prediction-dialog-grid"><div><span>${this.language === 'es' ? 'Ventana' : 'Window'}</span><strong>${formatClock(target.windowStart, this.language)}–${formatClock(target.windowEnd, this.language)}</strong></div><div><span>${this.language === 'es' ? 'Duración' : 'Duration'}</span><strong>${formatDuration(target.durationMinutes)}</strong></div><div><span>${this.language === 'es' ? 'Confianza' : 'Confidence'}</span><strong>${Math.round(target.confidence * 100)}%</strong></div></div>
          ${calculation && model ? html`
            <section class="prediction-receipt">
              <header><span>${icon('sparkle', 17)}</span><div><strong>${spanish ? `Cómo llegamos a las ${formatClock(target.recommendedStart, this.language)}` : `How we reached ${formatClock(target.recommendedStart, this.language)}`}</strong><small>${spanish ? 'Un cálculo local que puedes comprobar' : 'A local calculation you can verify'}</small></div></header>
              ${target.kind === 'nap' && calculation.anchorAt && calculation.wakeWindowMinutes != null ? html`
                <div class="prediction-equation">
                  <div><span>${spanish ? 'Partimos de' : 'Starting from'}</span><strong>${formatClock(calculation.anchorAt, this.language)}</strong><small>${anchorLabel}</small></div>
                  <b>+</b>
                  <div><span>${spanish ? 'Ventana ajustada' : 'Adjusted wake window'}</span><strong>${formatDuration(calculation.wakeWindowMinutes)}</strong><small>${calculation.startSampleCount} ${spanish ? 'muestras recientes' : 'recent samples'}</small></div>
                  <b>=</b>
                  <div class="result"><span>${spanish ? 'Inicio previsto' : 'Expected start'}</span><strong>${formatClock(calculation.baseRecommendedStart || target.recommendedStart, this.language)}</strong><small>${calculation.adjustmentMinutes ? (spanish ? 'antes de salvaguardas' : 'before safeguards') : (spanish ? 'resultado exacto' : 'exact result')}</small></div>
                </div>
                ${calculation.adjustmentMinutes ? html`<p class="prediction-adjustment">${icon('clock', 15)} ${spanish ? `La ventana original ya había pasado al actualizar el plan. Se desplazó ${formatDuration(calculation.adjustmentMinutes)} para no recomendar una hora caducada: ${formatClock(target.recommendedStart, this.language)}.` : `The original window had already passed when the plan refreshed. It was moved by ${formatDuration(calculation.adjustmentMinutes)} so it would not recommend an expired time: ${formatClock(target.recommendedStart, this.language)}.`}</p>` : nothing}
              ` : html`
                <div class="prediction-equation night-equation">
                  <div><span>${spanish ? 'Patrón usado' : 'Pattern used'}</span><strong>${model.bedtimes.count || '—'}</strong><small>${model.bedtimes.count ? (spanish ? 'noches recientes' : 'recent nights') : (spanish ? 'referencia por edad' : 'age guidance')}</small></div>
                  <b>→</b>
                  <div><span>${spanish ? 'Mediana de acostarse' : 'Median bedtime'}</span><strong>${this.formatPredictionMinute(model.bedtimes.medianMinuteOfDay)}</strong><small>${calculation.anchorType === 'age_guidance' ? (spanish ? 'valor inicial' : 'initial value') : (spanish ? 'patrón personal' : 'personal pattern')}</small></div>
                  <b>=</b>
                  <div class="result"><span>${spanish ? 'Inicio previsto' : 'Expected start'}</span><strong>${formatClock(target.recommendedStart, this.language)}</strong><small>${calculation.expectedWakeAt ? `${spanish ? 'despertar ~' : 'wake ~'} ${formatClock(calculation.expectedWakeAt, this.language)}` : ''}</small></div>
                </div>
              `}
            </section>

            <section class="prediction-learning"><span>${icon('moon', 17)}</span><div><strong>${spanish ? 'Lo que estamos aprendiendo' : 'What we are learning'}</strong><p>${pattern}</p></div></section>

            <details class="prediction-method">
              <summary>${icon('chevron', 15)}<div><strong>${spanish ? 'Ver datos y método' : 'See data and method'}</strong><small>${spanish ? 'Medianas, pesos y muestras exactas' : 'Medians, weights and exact samples'}</small></div></summary>
              <div class="prediction-method-body">
                <div class="prediction-evidence-grid">
                  ${target.kind === 'nap' ? html`
                    <div><span>${spanish ? 'Base por edad' : 'Age baseline'}</span><strong>${formatDuration(model.baseline.wakeWindowMinutes)}</strong><small>${ageLabel} · ${model.baseline.expectedNaps} ${spanish ? 'siestas esperadas' : 'expected naps'}</small></div>
                    <div><span>${spanish ? 'Mediana personal' : 'Personal median'}</span><strong>${learnedWake == null ? '—' : formatDuration(learnedWake)}</strong><small>${model.wakeWindows.count} ${spanish ? 'ventanas válidas' : 'valid windows'}</small></div>
                    <div><span>${spanish ? 'Objetivo final' : 'Final target'}</span><strong>${formatDuration(model.wakeWindows.finalMinutes)}</strong><small>${historyPercent}% ${spanish ? 'historial' : 'history'} · ${agePercent}% ${spanish ? 'edad' : 'age'}</small></div>
                    <div><span>${spanish ? 'Margen' : 'Margin'}</span><strong>±${formatDuration(this.sleepPlan?.wakeWindowMarginMinutes ?? 0)}</strong><small>${spanish ? 'según variabilidad' : 'from variability'}</small></div>
                    <div><span>${spanish ? 'Duración de siesta' : 'Nap duration'}</span><strong>${formatDuration(model.napDurations.finalMinutes)}</strong><small>${model.napDurations.count} ${spanish ? 'siestas recientes' : 'recent naps'}</small></div>
                  ` : html`
                    <div><span>${spanish ? 'Inicio nocturno' : 'Night start'}</span><strong>${this.formatPredictionMinute(model.bedtimes.medianMinuteOfDay)}</strong><small>${model.bedtimes.count} ${spanish ? 'noches recientes' : 'recent nights'}</small></div>
                    <div><span>${spanish ? 'Despertar matinal' : 'Morning wake'}</span><strong>${this.formatPredictionMinute(model.morningWakes.medianMinuteOfDay)}</strong><small>${model.morningWakes.count} ${spanish ? 'mañanas recientes' : 'recent mornings'}</small></div>
                    <div><span>${spanish ? 'Duración nocturna' : 'Night duration'}</span><strong>${formatDuration(model.nightDurations.finalMinutes)}</strong><small>${model.nightDurations.count} ${spanish ? 'noches agrupadas' : 'grouped nights'}</small></div>
                    <div><span>${spanish ? 'Duración prevista' : 'Expected duration'}</span><strong>${formatDuration(target.durationMinutes)}</strong><small>${calculation.durationSource === 'recent_night_duration' ? (spanish ? 'mediana de duración' : 'duration median') : (spanish ? 'de acostarse a despertar' : 'bedtime to wake-up')}</small></div>
                  `}
                  <div><span>${spanish ? 'Confianza global' : 'Overall confidence'}</span><strong>${Math.round(target.confidence * 100)}%</strong><small>${model.confidence.sampleCount} ${spanish ? 'ventanas recientes' : 'recent windows'}</small></div>
                </div>

                <div class="prediction-formula">
                  <strong>${spanish ? 'Cómo funciona por debajo' : 'How it works underneath'}</strong>
                  ${target.kind === 'nap' ? html`
                    <p>${learnedWake != null ? (spanish
                      ? `Ventana = redondeo de ${formatDuration(model.baseline.wakeWindowMinutes)} × ${agePercent}% + ${formatDuration(learnedWake)} × ${historyPercent}% = ${formatDuration(model.wakeWindows.finalMinutes)}.`
                      : `Wake window = rounded ${formatDuration(model.baseline.wakeWindowMinutes)} × ${agePercent}% + ${formatDuration(learnedWake)} × ${historyPercent}% = ${formatDuration(model.wakeWindows.finalMinutes)}.`) : (spanish ? 'Sin tres muestras válidas se usa directamente la referencia por edad.' : 'With fewer than three valid samples, age guidance is used directly.')}</p>
                    <p>${spanish ? `La ventana de tolerancia usa la desviación mediana de las muestras × 1,7, limitada entre 25 y 75 minutos; con menos de cuatro muestras son 35 minutos.` : `The tolerance window uses median sample deviation × 1.7, clamped between 25 and 75 minutes; with fewer than four samples it is 35 minutes.`}</p>
                  ` : html`<p>${spanish ? 'El inicio es la mediana de hasta 10 noches recientes. El fin usa la mediana de hasta 14 despertares matinales; si el intervalo resultante cae fuera de 6–13 horas, se usa la mediana de duración nocturna.' : 'Start time is the median of up to 10 recent nights. End time uses the median of up to 14 morning wake-ups; if that span falls outside 6–13 hours, median night duration is used.'}</p>`}
                  <p>${model.confidence.rule === 'recent_wake_samples'
                    ? (spanish ? `Confianza = mínimo(92%, 57% + ${model.confidence.sampleCount} muestras × 2,5 puntos) = ${Math.round(model.confidence.value * 100)}%.` : `Confidence = min(92%, 57% + ${model.confidence.sampleCount} samples × 2.5 points) = ${Math.round(model.confidence.value * 100)}%.`)
                    : (spanish ? `A falta de historial suficiente, la confianza inicial es ${Math.round(model.confidence.value * 100)}%.` : `Without enough history, initial confidence is ${Math.round(model.confidence.value * 100)}%.`)}</p>
                  <p>${spanish ? `Se revisan como máximo 80 tramos cerrados. Solo entran ventanas despierto de 15 min a 12 h y siestas de 15 min a 3 h.` : `At most 80 closed sleep periods are reviewed. Only wake windows from 15 min to 12 h and naps from 15 min to 3 h are accepted.`}</p>
                </div>

                ${this.renderPredictionSamples(target)}

                <p class="prediction-local-note">${icon('lock', 15)} ${spanish ? 'El predictor es determinista y local: no llama a Gemini ni vuelve a mirar las fotos. Usa los tramos de sueño ya guardados, así que una detección visual errónea puede influir hasta que corrijas ese tramo.' : 'The predictor is deterministic and local: it does not call Gemini or inspect images again. It uses saved sleep periods, so an incorrect visual detection can influence it until that period is corrected.'}</p>
                <p class="prediction-generated">${spanish ? 'Calculado' : 'Calculated'} · ${formatDateTime(model.generatedAt, this.language)}</p>
              </div>
            </details>
          ` : html`<p>${spanish ? 'Calculado con las ventanas despierto, las duraciones y los horarios recientes guardados en este historial.' : target.explanation}</p>`}
        </section>
      </div>
    `;
  }

  private formatPredictionMinute(value: number): string {
    if (!Number.isFinite(value)) return '—';
    const normalized = ((Math.round(value) % 1440) + 1440) % 1440;
    return `${String(Math.floor(normalized / 60)).padStart(2, '0')}:${String(normalized % 60).padStart(2, '0')}`;
  }

  private renderPredictionSamples(target: SleepPredictionTarget): TemplateResult | typeof nothing {
    const model = this.sleepPlan?.modelDetails;
    if (!model) return nothing;
    const spanish = this.language === 'es';
    if (target.kind === 'nap') {
      const wakeSamples = model.wakeWindows.samples.slice().reverse();
      const durationSamples = model.napDurations.samples.slice().reverse();
      if (!wakeSamples.length && !durationSamples.length) return nothing;
      return html`
        <section class="prediction-samples">
          <h3>${spanish ? 'Muestras exactas usadas' : 'Exact samples used'}</h3>
          ${wakeSamples.length ? html`<h4>${spanish ? 'Ventanas despierto' : 'Wake windows'}</h4><ol>${wakeSamples.map((sample) => html`<li><span>${formatDateTime(sample.previousSleepEndedAt, this.language)} → ${formatClock(sample.nextSleepStartedAt, this.language)}</span><strong>${formatDuration(sample.minutes)}</strong></li>`)}</ol>` : nothing}
          ${durationSamples.length ? html`<h4>${spanish ? 'Duraciones de siesta' : 'Nap durations'}</h4><ol>${durationSamples.map((sample) => html`<li><span>${formatDateTime(sample.startedAt, this.language)}</span><strong>${formatDuration(sample.minutes)}</strong></li>`)}</ol>` : nothing}
        </section>
      `;
    }
    const bedtimes = model.bedtimes.samples.slice().reverse();
    const wakes = model.morningWakes.samples.slice().reverse();
    const durations = model.nightDurations.samples.slice().reverse();
    if (!bedtimes.length && !wakes.length && !durations.length) return nothing;
    return html`
      <section class="prediction-samples night-samples">
        <h3>${spanish ? 'Noches exactas usadas' : 'Exact nights used'}</h3>
        ${bedtimes.length ? html`<h4>${spanish ? 'Horas de acostarse' : 'Bedtimes'}</h4><ol>${bedtimes.map((sample) => html`<li><span>${formatDateTime(sample.at, this.language)}</span><strong>${formatClock(sample.at, this.language)}</strong></li>`)}</ol>` : nothing}
        ${wakes.length ? html`<h4>${spanish ? 'Despertares matinales' : 'Morning wake-ups'}</h4><ol>${wakes.map((sample) => html`<li><span>${formatDateTime(sample.at, this.language)}</span><strong>${formatClock(sample.at, this.language)}</strong></li>`)}</ol>` : nothing}
        ${durations.length ? html`<h4>${spanish ? 'Duraciones nocturnas agrupadas' : 'Grouped night durations'}</h4><ol>${durations.map((sample) => html`<li><span>${sample.nightDate || formatDateTime(sample.startedAt, this.language)}</span><strong>${formatDuration(sample.minutes)}</strong></li>`)}</ol>` : nothing}
      </section>
    `;
  }

  private renderManualForm(): TemplateResult {
    const suggestion = this.suggestedEnd(this.manualForm.startedAt, this.manualForm.kind);
    const start = new Date(this.manualForm.startedAt);
    const effectiveEnd = this.manualForm.endedAt ? new Date(this.manualForm.endedAt) : new Date(suggestion);
    const duration = Number.isFinite(start.getTime()) && Number.isFinite(effectiveEnd.getTime())
      ? Math.max(0, Math.round((effectiveEnd.getTime() - start.getTime()) / 60_000))
      : 0;
    const title = this.manualForm.kind === 'awake'
      ? (this.language === 'es' ? 'Despertar' : 'Awake period')
      : this.manualForm.kind === 'night'
        ? (this.language === 'es' ? 'Sueño largo' : 'Night sleep')
        : (this.language === 'es' ? 'Siesta' : 'Nap');
    return html`
      <form class="manual-form legacy-sleep-form" @submit=${(event: SubmitEvent) => { event.preventDefault(); void this.addManualSleep(); }}>
        <button type="button" class="dialog-close" aria-label=${this.t('dismiss')} @click=${() => this.closeManualForm()}>&times;</button>
        <header class="sleep-form-hero">
          <span>${icon(this.manualForm.kind === 'awake' ? 'waves' : 'moon', 28)}</span>
          <small>${this.language === 'es' ? 'Nuevo dato de sueño' : 'New sleep entry'}</small>
          <h2>${title}</h2>
          <div class="sleep-type-pills" role="radiogroup">
            <button type="button" class=${this.manualForm.kind === 'nap' ? 'selected' : ''} @click=${() => this.setManualKind('nap')}>${this.language === 'es' ? 'Siesta' : 'Nap'}</button>
            <button type="button" class=${this.manualForm.kind === 'awake' ? 'selected awake' : ''} @click=${() => this.setManualKind('awake')}>${this.language === 'es' ? 'Despertar' : 'Awake'}</button>
            <button type="button" class=${this.manualForm.kind === 'night' ? 'selected' : ''} @click=${() => this.setManualKind('night')}>${this.language === 'es' ? 'Noche' : 'Night'}</button>
          </div>
        </header>
        <div class="sleep-time-editor">
          ${this.renderTemporalButton(this.language === 'es' ? 'Inicio' : 'Start', this.manualForm.startedAt, 'manual-start', this.manualForm.startedAt)}
          <i>–</i>
          ${this.renderTemporalButton(
            this.language === 'es' ? 'Fin' : 'End',
            this.manualForm.endedAt,
            'manual-end',
            suggestion,
            !this.manualForm.endedAt && duration ? `${this.language === 'es' ? 'máx. previsto' : 'suggested max'} ${formatDuration(duration)}` : '',
          )}
        </div>
        ${!this.manualForm.endedAt ? html`<button type="button" class="suggested-end" @click=${() => { this.manualForm = { ...this.manualForm, endedAt: suggestion }; this.manualEndTouched = true; }}>${icon('sparkle', 16)}<span>${this.language === 'es' ? `Usar fin sugerido · ${formatClock(new Date(suggestion).toISOString(), this.language)}` : `Use suggested end · ${formatClock(new Date(suggestion).toISOString(), this.language)}`}</span></button>` : html`<button type="button" class="suggested-end subtle" @click=${() => { this.manualForm = { ...this.manualForm, endedAt: '' }; this.manualEndTouched = false; }}>${this.language === 'es' ? 'Dejar el sueño activo, sin hora de fin' : 'Leave this sleep active without an end time'}</button>`}
        ${this.manualFrameReviewBounds ? this.renderFrameReview() : nothing}
        ${this.manualForm.kind !== 'awake' ? html`${this.renderPauses('manual', this.manualForm.details.pauses)}${this.renderDetailGroups('manual', this.manualForm.details)}` : nothing}
        <section class="sleep-detail-group"><h4>${this.language === 'es' ? 'Comentario' : 'Comment'}</h4><textarea class="sleep-comment" .value=${this.manualForm.notes} placeholder=${this.language === 'es' ? 'Aún no se han agregado comentarios' : 'No comments yet'} @input=${(event: Event) => { this.manualForm = { ...this.manualForm, notes: inputValue(event) }; }}></textarea></section>
        ${this.inlineError ? html`<div class="inline-error sleep-form-error" role="alert">${this.inlineError}</div>` : nothing}
        <div class="sleep-form-actions"><button type="button" class="button ghost" @click=${() => this.closeManualForm()}>${this.t('cancel')}</button><button class="sheet-save" aria-label=${this.t('addSleep')} ?disabled=${this.sleepBusy === 'add'}>${this.sleepBusy === 'add' ? html`<span class="spinner"></span>` : icon('check', 31)}</button></div>
      </form>
    `;
  }

  private renderManualDialog(): TemplateResult | typeof nothing {
    if (!this.manualOpen) return nothing;
    return html`
      <div class=${`manual-dialog-backdrop ${this.rhythmMode === 'day' ? 'theme-day' : ''}`} @click=${(event: Event) => { if (event.target === event.currentTarget) this.closeManualForm(); }}>
        <section class="manual-dialog" role="dialog" aria-modal="true" aria-label=${this.t('manualTitle')}>
          ${this.renderManualForm()}
        </section>
      </div>
    `;
  }

  private renderSleepEditor(): TemplateResult | typeof nothing {
    if (!this.editingSleep) return nothing;
    const start = new Date(this.editSleepForm.startedAt);
    const end = this.editSleepForm.endedAt ? new Date(this.editSleepForm.endedAt) : null;
    const minutes = end && end > start ? Math.round((end.getTime() - start.getTime()) / 60_000) : 0;
    const title = this.language === 'es'
      ? this.editSleepForm.kind === 'awake' ? 'Despertar' : this.editSleepForm.kind === 'night' ? 'Sueño largo' : 'Siesta'
      : this.editSleepForm.kind === 'awake' ? 'Awake period' : this.editSleepForm.kind === 'night' ? 'Night sleep' : 'Nap';
    return html`<div class=${`manual-dialog-backdrop ${this.rhythmMode === 'day' ? 'theme-day' : ''}`} @click=${(event: Event) => { if (event.target === event.currentTarget) this.editingSleep = null; }}><form class="manual-dialog manual-form legacy-sleep-form sleep-edit-form" @submit=${(event: SubmitEvent) => { event.preventDefault(); void this.saveSleepEditor(); }}>
      <button type="button" class="dialog-close" @click=${() => { this.editingSleep = null; }}>&times;</button>
      <header class="sleep-form-hero"><span>${icon(this.editSleepForm.kind === 'awake' ? 'waves' : 'moon', 28)}</span><small>${this.language === 'es' ? 'Editar segmento' : 'Edit segment'}</small><h2>${title}</h2></header>
      <div class="sleep-time-editor">${this.renderTemporalButton(this.language === 'es' ? 'Inicio' : 'Start', this.editSleepForm.startedAt, 'edit-start', this.editSleepForm.startedAt)}<i>–</i>${this.renderTemporalButton(this.language === 'es' ? 'Fin' : 'End', this.editSleepForm.endedAt, 'edit-end', this.editSleepForm.endedAt || this.suggestedEnd(this.editSleepForm.startedAt, this.editSleepForm.kind))}</div>
      <div class="editor-duration">${minutes ? `${formatDuration(minutes)} ${this.language === 'es' ? 'de duración' : 'duration'}` : (this.language === 'es' ? 'Sueño activo' : 'Active sleep')} · ${formatDateTime(this.editingSleep.startedAt, this.language)}</div>
      <div class="sleep-type-pills editor-types" role="radiogroup"><button type="button" class=${this.editSleepForm.kind === 'nap' ? 'selected' : ''} @click=${() => { this.editSleepForm = { ...this.editSleepForm, kind: 'nap' }; }}>${this.language === 'es' ? 'Siesta' : 'Nap'}</button><button type="button" class=${this.editSleepForm.kind === 'awake' ? 'selected awake' : ''} @click=${() => { this.editSleepForm = { ...this.editSleepForm, kind: 'awake', details: EMPTY_DETAILS() }; }}>${this.language === 'es' ? 'Despertar' : 'Awake'}</button><button type="button" class=${this.editSleepForm.kind === 'night' ? 'selected' : ''} @click=${() => { this.editSleepForm = { ...this.editSleepForm, kind: 'night' }; }}>${this.language === 'es' ? 'Sueño largo' : 'Night sleep'}</button></div>
      <div class="event-source-note">${icon(this.editingSleep.source === 'manual' ? 'heart' : 'sparkle', 16)}<span>${this.language === 'es' ? `Origen: ${this.editingSleep.source === 'manual' ? 'añadido a mano' : this.editingSleep.source === 'vision' ? 'detección de cámara' : 'histórico importado'}` : `Source: ${this.editingSleep.source}`}</span></div>
      ${this.renderFrameReview()}
      ${this.editSleepForm.kind !== 'awake' ? html`${this.renderPauses('edit', this.editSleepForm.details.pauses)}${this.renderDetailGroups('edit', this.editSleepForm.details)}` : nothing}
      <section class="sleep-detail-group"><h4>${this.language === 'es' ? 'Comentario' : 'Comment'}</h4><textarea class="sleep-comment" .value=${this.editSleepForm.notes} @input=${(event: Event) => { this.editSleepForm = { ...this.editSleepForm, notes: inputValue(event) }; }}></textarea></section>
      <details class="editor-more-options">
        <summary>${icon('chevron', 15)}<span>${this.language === 'es' ? 'Más opciones' : 'More options'}</span></summary>
        <div class="editor-delete-copy">
          <strong>${this.language === 'es' ? 'Eliminar segmento' : 'Delete segment'}</strong>
          <span>${this.language === 'es'
            ? this.editingSleep.source === 'manual'
              ? 'El registro desaparecerá, pero las capturas se conservarán.'
              : 'Se elimina esta detección; las capturas y sus análisis se conservarán. Si el sueño continúa, futuras detecciones podrán iniciar un segmento nuevo.'
            : this.editingSleep.source === 'manual'
              ? 'The record will be removed, but captures will be preserved.'
              : 'This detection is removed; captures and analyses are preserved. If sleep continues, future detections may start a new segment.'}</span>
        </div>
        <button type="button" class="editor-delete-action" ?disabled=${this.editSleepBusy} @click=${() => this.deleteSleepEditor()}>
          ${this.language === 'es' ? 'Eliminar este segmento' : 'Delete this segment'}
        </button>
      </details>
      ${this.inlineError ? html`<div class="inline-error sleep-form-error">${this.inlineError}</div>` : nothing}<div class="sleep-form-actions"><button type="button" class="button ghost" @click=${() => { this.editingSleep = null; }}>${this.language === 'es' ? 'Cancelar' : 'Cancel'}</button><button class="sheet-save" ?disabled=${this.editSleepBusy}>${this.editSleepBusy ? html`<span class="spinner"></span>` : icon('check', 31)}</button></div>
    </form></div>`;
  }

  private sleepSeries(): Array<{ date: string; total: number; naps: number; night: number; count: number; wake: string; bed: string }> {
    const rows = new Map<string, { date: string; total: number; naps: number; night: number; count: number; wake: string; bed: string }>();
    for (const event of this.sleepEvents) {
      if (!event.endedAt || event.kind === 'awake') continue;
      const start = new Date(event.startedAt);
      const end = new Date(event.endedAt);
      const date = event.kind === 'night' && start.getHours() >= 18 ? localDateKey(new Date(start.getTime() + 86_400_000)) : localDateKey(start);
      const row = rows.get(date) ?? { date, total: 0, naps: 0, night: 0, count: 0, wake: '', bed: '' };
      const pausedMinutes = (event.details?.pauses ?? []).reduce((total, pause) => {
        const pauseStart = Math.max(start.getTime(), new Date(pause.startedAt).getTime());
        const pauseEnd = Math.min(end.getTime(), new Date(pause.endedAt).getTime());
        return total + Math.max(0, Math.round((pauseEnd - pauseStart) / 60_000));
      }, 0);
      const minutes = Math.max(1, Math.round((end.getTime() - start.getTime()) / 60_000) - pausedMinutes);
      row.total += minutes;
      row.count += 1;
      if (event.kind === 'nap') row.naps += minutes;
      else {
        row.night += minutes;
        if (!row.bed || start < new Date(`${date}T${row.bed}:00`)) row.bed = `${String(start.getHours()).padStart(2, '0')}:${String(start.getMinutes()).padStart(2, '0')}`;
        row.wake = `${String(end.getHours()).padStart(2, '0')}:${String(end.getMinutes()).padStart(2, '0')}`;
      }
      rows.set(date, row);
    }
    return [...rows.values()].sort((a, b) => a.date.localeCompare(b.date));
  }

  private renderMetricBars(rows: Array<{ label: string; value: number }>): TemplateResult {
    const visible = rows.slice(-21);
    const maximum = Math.max(1, ...visible.map((row) => row.value));
    return html`<div class="legacy-chart" role="img">${visible.map((row) => html`
      <div class="legacy-chart-column" title=${`${formatTrendDate(row.label)}: ${formatDuration(row.value)}`}>
        <strong>${formatDuration(row.value)}</strong><i style=${`--height:${Math.max(2, row.value / maximum * 100)}%`}></i><small>${formatTrendDate(row.label)}</small>
      </div>`)}
    </div>`;
  }

  private renderVisualStatistic(kind: 'pacifier' | 'mouth_open' | 'head_side' | 'clothing', title: string): TemplateResult {
    const metric = this.visionStatistics?.metrics[kind];
    if (this.visionStatisticsLoading) return html`<div class="empty-state"><span class="spinner"></span><p>Cargando el histórico visual…</p></div>`;
    if (!metric?.segments.length) return html`<div class="empty-state"><p>No hay observaciones estructuradas para este periodo.</p></div>`;
    const total = metric.total_minutes;
    const label = (value: string): string => ({
      pacifier: 'Con chupete', 'no pacifier': 'Sin chupete', 'mouth open': 'Boca abierta', 'mouth closed': 'Boca cerrada',
      left: 'Izquierda', right: 'Derecha', back: 'Boca arriba', face_down: 'Boca abajo', diaper_only: 'Solo pañal',
      short_sleeve_onesie: 'Body de manga corta', long_sleeve_onesie: 'Body de manga larga', sleep_sack: 'Saco de dormir', blanket: 'Manta',
    }[value] ?? value.replaceAll('_', ' '));
    return html`
      <section class="legacy-stat-grid">
        <article class="legacy-stat-card hero-stat"><span>Tiempo observado</span><strong>${formatDuration(total)}</strong><small>${this.visionStatistics?.visible_sample_count ?? 0} imágenes válidas</small></article>
        <article class="legacy-stat-card"><h2>${title}</h2><div class="donut-legend">${metric.segments.map((segment) => html`<div><i style=${`--color:${segment.color}`}></i><span>${label(segment.label)}</span><strong>${segment.percent}%</strong><small>${formatDuration(segment.minutes)}</small></div>`)}</div></article>
      </section>
      ${kind === 'pacifier' || kind === 'mouth_open' ? html`<article class="legacy-stat-card"><h2>Evolución diaria</h2>${this.renderMetricBars((this.visionStatistics?.daily ?? []).map((day) => ({ label: day.date, value: kind === 'pacifier' ? day.pacifier_minutes : day.mouth_open_minutes })))}</article>` : nothing}
    `;
  }

  private renderHistory(): TemplateResult {
    const series = this.sleepSeries();
    const total = series.reduce((sum, row) => sum + row.total, 0);
    const nightRows = series.filter((row) => row.night > 0);
    const medianNight = medianMinutes(nightRows.map((row) => row.night));
    const medianNaps = medianMinutes(series.map((row) => row.naps));
    const tabs = [
      ['summary', 'Resumen de sueño'], ['naps', 'Siestas'], ['awake', 'Tiempo despierto'], ['night', 'Sueño nocturno'],
      ['pacifier', 'Chupete'], ['head', 'Cabeza'], ['clothing', 'Ropa'], ['mouth', 'Boca'],
    ] as const;
    return html`
      <main class="page history-page" id="main">
        <section class="page-heading">
          <div><span class="eyebrow">Histórico completo</span><h1>Tendencias</h1><p>El mismo análisis de sueño y de las imágenes de la app original.</p></div>
          <div class="heading-actions"><button class="icon-button" aria-label=${this.t('refresh')} ?disabled=${this.refreshingData} @click=${() => this.loadOperationalData(true)}><span class=${this.refreshingData ? 'spin' : ''}>${icon('refresh', 19)}</span></button></div>
        </section>
        <nav class="legacy-stats-tabs" aria-label="Estadísticas">${tabs.map(([tab, label]) => html`<button class=${this.statsTab === tab ? 'active' : ''} @click=${() => { this.statsTab = tab; if (['pacifier', 'head', 'clothing', 'mouth'].includes(tab)) void this.loadVisionStatistics(); }}>${label}</button>`)}</nav>
        ${this.statsTab === 'summary' ? html`
          <section class="legacy-stat-grid three"><article class="legacy-stat-card hero-stat"><span>Sueño total</span><strong>${formatDuration(total)}</strong><small>${series.length} días con datos</small></article><article class="legacy-stat-card hero-stat"><span>Mediana nocturna</span><strong>${formatDuration(medianNight)}</strong><small>${nightRows.length} noches con datos</small></article><article class="legacy-stat-card hero-stat"><span>Mediana diaria de siestas</span><strong>${formatDuration(medianNaps)}</strong><small>${series.length} días analizados</small></article></section>
          <article class="legacy-stat-card"><h2>Sueño diario</h2>${this.renderMetricBars(series.map((row) => ({ label: row.date, value: row.total })))}</article>
        ` : this.statsTab === 'naps' ? html`<article class="legacy-stat-card"><h2>Sueño durante el día</h2>${this.renderMetricBars(series.map((row) => ({ label: row.date, value: row.naps })))}</article>
        <section class="legacy-stat-grid"><article class="legacy-stat-card hero-stat"><span>Mediana diaria de siestas</span><strong>${formatDuration(medianNaps)}</strong></article><article class="legacy-stat-card hero-stat"><span>Días analizados</span><strong>${series.length}</strong></article></section>`
        : this.statsTab === 'awake' ? html`<article class="legacy-stat-card"><h2>Hora de despertar</h2><div class="clock-history">${series.slice(-21).map((row) => html`<div><span>${formatTrendDate(row.date)}</span><strong>${row.wake || '—'}</strong></div>`)}</div></article>`
        : this.statsTab === 'night' ? html`<article class="legacy-stat-card"><h2>Sueño nocturno</h2>${this.renderMetricBars(series.map((row) => ({ label: row.date, value: row.night })))}</article><article class="legacy-stat-card"><h2>Se durmió / se despertó</h2><div class="clock-history">${series.slice(-21).map((row) => html`<div><span>${formatTrendDate(row.date)}</span><strong>${row.bed || '—'} → ${row.wake || '—'}</strong></div>`)}</div></article>`
        : this.statsTab === 'pacifier' ? this.renderVisualStatistic('pacifier', 'Uso del chupete')
        : this.statsTab === 'head' ? this.renderVisualStatistic('head_side', 'Posición de la cabeza')
        : this.statsTab === 'clothing' ? this.renderVisualStatistic('clothing', 'Ropa detectada')
        : this.renderVisualStatistic('mouth_open', 'Boca abierta o cerrada')}
        <details class="legacy-raw-history"><summary>Ver registros detallados</summary><section class="history-grid"><article class="history-panel">${this.renderSleepList(this.sleepEvents)}${this.renderHistoryPager('sleep', this.sleepEvents.length)}</article><article class="history-panel">${this.renderCryList()}${this.renderHistoryPager('cry', this.cryEvents.length)}</article></section></details>
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
          ? html`
            <details class="camera-setup-guide">
              <summary>${icon('camera', 16)}<span>${this.t('boifunGuideTitle')}</span></summary>
              <div class="camera-setup-guide-body">
                <p>${this.t('boifunGuideIntro')}</p>
                <ol>
                  <li>${this.t('boifunGuideStepOne')}</li>
                  <li>${this.t('boifunGuideStepTwo')}</li>
                  <li>${this.t('boifunGuideStepThree')}</li>
                  <li>${this.t('boifunGuideStepFour')}</li>
                </ol>
                <small>${this.t('boifunGuideNetworkHint')}</small>
              </div>
            </details>
            <div class="field"><span>${this.t('cameraEntity')}</span>${this.renderSinglePicker(this.entities.camera, this.draft.camera.entityId, this.t('noCamera'), (id) => this.updateDraft((draft) => { draft.camera.entityId = id; }))}</div>
          `
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
        <div class=${`app-shell ${this.page === 'sleep' && this.rhythmMode === 'day' ? 'theme-day' : 'theme-night'}`}>${this.renderHeader()}${this.renderHealthBanner()}${this.page === 'sleep' ? this.renderDashboard() : this.page === 'camera' ? this.renderCamera() : this.page === 'data' ? this.renderHistory() : this.renderSettings()}</div>
        ${this.renderManualDialog()}
        ${this.renderSleepEditor()}
        ${this.renderPredictionDialog()}
        ${this.renderTemporalPicker()}
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
