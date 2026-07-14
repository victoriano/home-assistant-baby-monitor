import { render, type TemplateResult } from 'lit';
import { describe, expect, it, vi } from 'vitest';

import { api } from '../src/api';
import { BabyMonitorApp } from '../src/baby-monitor-app';
import type { RhythmSegment } from '../src/sleep-rhythm';
import type { FrameRecord, Language, SleepEvent, SleepEventDetails, SleepKind } from '../src/types';

interface SleepEditorHarness {
  language: Language;
  manualOpen: boolean;
  manualForm: {
    startedAt: string;
    endedAt: string;
    kind: SleepKind;
    notes: string;
    details: SleepEventDetails;
  };
  openSleepEditor(event: SleepEvent): void;
  openRhythmSegment(segment: RhythmSegment): void;
  renderSleepEditor(): TemplateResult;
}

const automaticSleep: SleepEvent = {
  id: 'night-1',
  startedAt: '2026-07-13T21:07:00.000Z',
  endedAt: '2026-07-14T04:09:00.000Z',
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
    faceVisible: 'yes',
    headSide: 'left',
    bodyPosition: 'supine',
    clothingItems: ['sleep_sack'],
    pacifier: 'yes',
    mouthOpen: 'no',
  },
};

function harness(): SleepEditorHarness {
  const app = new BabyMonitorApp() as unknown as SleepEditorHarness;
  app.language = 'es';
  return app;
}

async function openAndRender(app: SleepEditorHarness, event: SleepEvent): Promise<HTMLElement> {
  app.openSleepEditor(event);
  await vi.waitFor(() => expect(api.getNearestFrames).toHaveBeenCalled());
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
  it('loads the first camera frame and exposes the complete model analysis', async () => {
    vi.spyOn(api, 'getNearestFrames').mockResolvedValue([firstFrame]);
    const container = await openAndRender(harness(), automaticSleep);

    expect(api.getNearestFrames).toHaveBeenCalledWith(automaticSleep.startedAt, 7);
    expect(container.querySelector('.frame-point-switch button.active')?.textContent).toContain('Inicio');
    expect(container.querySelector('.frame-review-card img')?.getAttribute('src')).toContain('frame-1/image');
    expect(container.querySelector('.frame-model-details')?.textContent).toContain('Ver análisis del modelo');
    expect(container.querySelector('.frame-model-details')?.textContent).toContain('gemini-3.1-flash-lite');
    expect(container.querySelector('.frame-model-details')?.textContent).toContain('Boca abierta');
  });

  it('never shows delete for camera events and hides it under more options for manual events', async () => {
    vi.spyOn(api, 'getNearestFrames').mockResolvedValue([firstFrame]);
    const automatic = await openAndRender(harness(), automaticSleep);
    expect(automatic.querySelector('.editor-delete')).toBeNull();
    expect(automatic.querySelector('.editor-more-options')).toBeNull();

    const manual = await openAndRender(harness(), { ...automaticSleep, id: 'manual-1', source: 'manual' });
    expect(manual.querySelector('.editor-delete')).toBeNull();
    expect(manual.querySelector('.editor-more-options')?.textContent).toContain('Más opciones');
    expect(manual.querySelector('.editor-delete-action')?.textContent).toContain('Eliminar este registro');
  });

  it('opens an inferred night interruption as a prefilled awake entry', () => {
    const app = harness();
    const start = new Date('2026-07-14T04:09:00.000Z');
    const end = new Date('2026-07-14T04:24:00.000Z');

    app.openRhythmSegment({
      id: 'awake-gap',
      event: null,
      prediction: null,
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
  });
});
