import { render, type TemplateResult } from 'lit';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '../src/api';
import { BabyMonitorApp } from '../src/baby-monitor-app';
import type { RhythmSegment } from '../src/sleep-rhythm';
import type { FrameRecord, Language, SleepEvent, SleepEventDetails, SleepKind } from '../src/types';

interface SleepEditorHarness {
  language: Language;
  manualOpen: boolean;
  activeSleepOverlay: SleepEvent | null;
  activeSleepNow: number;
  editingSleep: SleepEvent | null;
  sleepBusy: 'start' | 'stop' | 'add' | '';
  manualForm: {
    startedAt: string;
    endedAt: string;
    kind: SleepKind;
    notes: string;
    details: SleepEventDetails;
  };
  frameReview: {
    frames: FrameRecord[];
    index: number;
  };
  loadOperationalData(initial?: boolean): Promise<void>;
  deleteSleepEditor(): Promise<void>;
  closeActiveSleepOverlay(): void;
  finishActiveSleep(): Promise<void>;
  openSleepEditor(event: SleepEvent): void;
  openRhythmSegment(segment: RhythmSegment): void;
  renderActiveSleepOverlay(): TemplateResult;
  renderManualDialog(): TemplateResult;
  renderSleepEditor(): TemplateResult;
}

const automaticSleep: SleepEvent = {
  id: 'night-1',
  startedAt: '2026-07-13T21:07:49.454Z',
  endedAt: '2026-07-14T04:09:29.132Z',
  kind: 'night',
  source: 'vision',
  notes: null,
  details: { tags: [], pauses: [] },
  locationId: 'madrid',
};

const firstFrame: FrameRecord = {
  id: 'frame-1',
  capturedAt: automaticSleep.startedAt,
  cameraEntityId: 'camera.baby_room',
  locationId: 'madrid',
  imageUrl: '/api/v1/frames/frame-1/image',
  imageAvailable: true,
  mimeType: 'image/jpeg',
  sizeBytes: 42_000,
  provider: 'gemini',
  model: 'gemini-3.1-flash-lite',
  label: {
    babyPresent: true,
    state: 'asleep',
    confidence: 0.97,
    description: 'Dormido boca arriba.',
    tags: ['calm'],
    inCrib: true,
    sleepSurface: 'crib',
    faceVisible: 'yes',
    headSide: 'left',
    bodyPosition: 'supine',
    clothingItems: ['sleep_sack'],
    pacifier: 'yes',
    mouthOpen: 'no',
  },
};

const segmentFrames: FrameRecord[] = Array.from({ length: 32 }, (_, index) => {
  const start = new Date(automaticSleep.startedAt).getTime();
  const end = new Date(automaticSleep.endedAt ?? automaticSleep.startedAt).getTime();
  const capturedAt = new Date(start + ((end - start) * index) / 31).toISOString();
  return {
    ...firstFrame,
    id: `frame-${index + 1}`,
    capturedAt,
    imageUrl: `/api/v1/frames/frame-${index + 1}/image`,
  };
});

function harness(): SleepEditorHarness {
  const app = new BabyMonitorApp() as unknown as SleepEditorHarness;
  app.language = 'es';
  app.loadOperationalData = vi.fn().mockResolvedValue(undefined);
  return app;
}

async function openAndRender(app: SleepEditorHarness, event: SleepEvent): Promise<HTMLElement> {
  app.openSleepEditor(event);
  await vi.waitFor(() => expect(api.getFramesBetween).toHaveBeenCalled());
  await vi.waitFor(() => {
    const container = document.createElement('div');
    render(app.renderSleepEditor(), container);
    expect(container.querySelector('.frame-review-card')).not.toBeNull();
  });
  const container = document.createElement('div');
  document.body.append(container);
  render(app.renderSleepEditor(), container);
  return container;
}

describe('sleep segment editor', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  it('loads every segment frame and uses start, middle, and end as precise jumps', async () => {
    vi.spyOn(api, 'getFramesBetween').mockResolvedValue(segmentFrames);
    const app = harness();
    const container = await openAndRender(app, automaticSleep);

    expect(api.getFramesBetween).toHaveBeenCalledWith(
      automaticSleep.startedAt,
      automaticSleep.endedAt,
      automaticSleep.locationId,
    );
    expect(container.querySelector('.frame-point-switch button.active')?.textContent).toContain('Inicio');
    expect(container.querySelector('.frame-review-card img')?.getAttribute('src')).toContain('frame-1/image');
    expect(container.querySelector('.editor-frames-heading')?.textContent).toContain('32 capturas');
    expect(container.querySelector('.frame-stepper')?.textContent).toContain('1 / 32');
    expect(container.querySelector('.frame-model-details')?.textContent).toContain('Ver análisis del modelo');
    expect(container.querySelector('.frame-model-details')?.textContent).toContain('gemini-3.1-flash-lite');
    expect(container.querySelector('.frame-model-details')?.textContent).toContain('Boca abierta');

    const pointButtons = container.querySelectorAll<HTMLButtonElement>('.frame-point-switch button');
    pointButtons[1].click();
    render(app.renderSleepEditor(), container);
    expect(container.querySelector('.frame-review-card img')?.getAttribute('src')).toContain('frame-17/image');

    container.querySelectorAll<HTMLButtonElement>('.frame-point-switch button')[2].click();
    render(app.renderSleepEditor(), container);
    expect(container.querySelector('.frame-review-card img')?.getAttribute('src')).toContain('frame-32/image');
    expect(container.querySelector('.frame-stepper')?.textContent).toContain('32 / 32');
  });

  it('hides deletion under more options for both camera and manual events', async () => {
    vi.spyOn(api, 'getFramesBetween').mockResolvedValue([firstFrame]);
    const automaticApp = harness();
    const automatic = await openAndRender(automaticApp, automaticSleep);
    expect(automatic.querySelector('.editor-delete')).toBeNull();
    expect(automatic.querySelector('.editor-more-options')?.textContent).toContain('Más opciones');
    expect(automatic.querySelector('.editor-more-options')?.textContent).toContain('capturas y sus análisis');
    expect(automatic.querySelector('.editor-delete-action')?.textContent).toContain('Eliminar este segmento');

    const manual = await openAndRender(harness(), { ...automaticSleep, id: 'manual-1', source: 'manual' });
    expect(manual.querySelector('.editor-delete')).toBeNull();
    expect(manual.querySelector('.editor-more-options')?.textContent).toContain('Más opciones');
    expect(manual.querySelector('.editor-delete-action')?.textContent).toContain('Eliminar este segmento');
  });

  it('deletes an automatic segment after warning that captures are preserved', async () => {
    vi.spyOn(api, 'getFramesBetween').mockResolvedValue([firstFrame]);
    const deleteRequest = vi.spyOn(api, 'deleteSleep').mockResolvedValue(undefined);
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const app = harness();
    await openAndRender(app, automaticSleep);

    await app.deleteSleepEditor();

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('capturas y sus análisis se conservarán'));
    expect(deleteRequest).toHaveBeenCalledWith(automaticSleep.id);
    expect(app.loadOperationalData).toHaveBeenCalledWith(false);
  });

  it('opens an inferred night interruption with every frame available for review', async () => {
    const wakingFrames: FrameRecord[] = [0, 5, 10].map((minutes, index) => ({
      ...firstFrame,
      id: `wake-frame-${index + 1}`,
      capturedAt: new Date(new Date('2026-07-14T04:09:00.000Z').getTime() + minutes * 60_000).toISOString(),
      imageUrl: `/api/v1/frames/wake-frame-${index + 1}/image`,
      label: { ...firstFrame.label!, state: index === 1 ? 'asleep' : 'awake' },
    }));
    vi.spyOn(api, 'getFramesBetween').mockResolvedValue(wakingFrames);
    const app = harness();
    const start = new Date('2026-07-14T04:09:00.000Z');
    const end = new Date('2026-07-14T04:24:00.000Z');

    app.openRhythmSegment({
      id: 'awake-gap',
      event: null,
      prediction: null,
      locationId: 'madrid',
      evidenceStartedAt: start.toISOString(),
      evidenceEndedAt: end.toISOString(),
      type: 'awake',
      start,
      end,
      startRatio: 0.7,
      endRatio: 0.73,
      minutes: 15,
      inferred: true,
      predicted: false,
    });

    expect(app.manualOpen).toBe(true);
    expect(app.manualForm.kind).toBe('awake');
    expect(new Date(app.manualForm.startedAt).toISOString()).toBe(start.toISOString());
    expect(new Date(app.manualForm.endedAt).toISOString()).toBe(end.toISOString());
    await vi.waitFor(() => expect(api.getFramesBetween).toHaveBeenCalledWith(
      start.toISOString(),
      end.toISOString(),
      'madrid',
    ));
    await vi.waitFor(() => expect(app.frameReview.frames).toHaveLength(3));

    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderManualDialog(), container);
    expect(container.querySelector('.editor-frames-heading')?.textContent).toContain('Frames del despertar nocturno');
    expect(container.querySelector('.editor-frames-heading')?.textContent).toContain('3 capturas');
    expect(container.querySelector('.frame-review-card img')?.getAttribute('src')).toContain('wake-frame-1/image');
    expect(container.querySelector('.frame-labels')?.textContent).toContain('Despierto');
    expect(container.querySelector('.frame-model-details')?.textContent).toContain('Ver análisis del modelo');

    container.querySelectorAll<HTMLButtonElement>('.frame-stepper button')[1].click();
    render(app.renderManualDialog(), container);
    expect(container.querySelector('.frame-review-card img')?.getAttribute('src')).toContain('wake-frame-2/image');
    expect(container.querySelector('.frame-labels')?.textContent).toContain('Dormido');
    expect(container.querySelector('.frame-stepper')?.textContent).toContain('2 / 3');
  });

  it('opens an unfinished manual sleep as the original floating live timer', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-14T16:08:49.000Z'));
    const ongoing: SleepEvent = {
      ...automaticSleep,
      id: 'manual-active',
      startedAt: '2026-07-14T14:01:46.000Z',
      endedAt: null,
      kind: 'nap',
      source: 'manual',
    };
    const app = harness();

    app.openRhythmSegment({
      id: ongoing.id,
      event: ongoing,
      prediction: null,
      locationId: ongoing.locationId,
      evidenceStartedAt: ongoing.startedAt,
      evidenceEndedAt: ongoing.startedAt,
      type: 'nap',
      start: new Date(ongoing.startedAt),
      end: new Date(),
      startRatio: 0.3,
      endRatio: 0.5,
      minutes: 127,
      inferred: false,
      predicted: false,
    });

    expect(app.activeSleepOverlay?.id).toBe(ongoing.id);
    expect(app.editingSleep).toBeNull();
    const container = document.createElement('div');
    render(app.renderActiveSleepOverlay(), container);
    expect(container.querySelector('.active-sleep-float')).not.toBeNull();
    expect(container.querySelector('.active-sleep-clock')?.textContent).toBe('2:07:03');
    expect(container.textContent).toContain('Siesta en curso');
    expect(container.textContent).toContain('Empezó a las');
    expect(container.querySelector('.active-sleep-stop')?.textContent).toContain('Finalizar ahora');
    expect(container.querySelector('.active-sleep-edit')?.textContent).toContain('Editar detalles');

    vi.advanceTimersByTime(1000);
    render(app.renderActiveSleepOverlay(), container);
    expect(container.querySelector('.active-sleep-clock')?.textContent).toBe('2:07:04');
    app.closeActiveSleepOverlay();
  });

  it('stops the active timer, refreshes the public app, and closes the floating window', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-14T16:08:49.000Z'));
    const ongoing: SleepEvent = {
      ...automaticSleep,
      id: 'manual-active',
      startedAt: '2026-07-14T14:01:46.000Z',
      endedAt: null,
      kind: 'night',
      source: 'manual',
    };
    const stopped = { ...ongoing, endedAt: new Date().toISOString() };
    const stopRequest = vi.spyOn(api, 'stopSleep').mockResolvedValue(stopped);
    const app = harness();

    app.openRhythmSegment({
      id: ongoing.id,
      event: ongoing,
      prediction: null,
      locationId: ongoing.locationId,
      evidenceStartedAt: ongoing.startedAt,
      evidenceEndedAt: ongoing.startedAt,
      type: 'night',
      start: new Date(ongoing.startedAt),
      end: new Date(),
      startRatio: 0.2,
      endRatio: 0.6,
      minutes: 127,
      inferred: false,
      predicted: false,
    });
    await app.finishActiveSleep();

    expect(stopRequest).toHaveBeenCalledOnce();
    expect(app.loadOperationalData).toHaveBeenCalledWith(false);
    expect(app.activeSleepOverlay).toBeNull();
    expect(app.sleepBusy).toBe('');
  });
});
