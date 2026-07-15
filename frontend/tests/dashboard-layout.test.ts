import { render, type TemplateResult } from 'lit';
import { describe, expect, it, vi } from 'vitest';

import { api } from '../src/api';
import { BabyMonitorApp } from '../src/baby-monitor-app';
import { cloneDefaultSettings, type AppSettings, type FrameRecord, type Language, type SleepEvent } from '../src/types';

interface DashboardHarness {
  settings: AppSettings;
  renderDashboard(): TemplateResult;
  loadOperationalData(showSpinner?: boolean): Promise<void>;
  refreshSnapshot(): Promise<FrameRecord | null>;
}

interface RhythmHarness {
  settings: AppSettings;
  language: Language;
  rhythmDate: string;
  rhythmMode: 'day' | 'night';
  sleepEvents: SleepEvent[];
  renderDailyRhythm(): TemplateResult;
}

describe('dashboard hierarchy', () => {
  it('puts the baby identity and manual refresh inside the rhythm card', () => {
    const app = new BabyMonitorApp() as unknown as DashboardHarness;
    app.settings = cloneDefaultSettings();
    app.settings.baby.name = 'Esteban';
    app.loadOperationalData = vi.fn().mockResolvedValue(undefined);

    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderDashboard(), container);

    expect(container.querySelector('.dashboard-heading')).toBeNull();
    expect(container.querySelector('.rhythm-context')?.textContent).toContain('Esteban');

    const refresh = container.querySelector('.rhythm-refresh');
    if (!(refresh instanceof HTMLButtonElement)) throw new Error('Missing rhythm refresh action');
    expect(refresh.getAttribute('aria-label')).toBe('Refresh all data');
    refresh.click();
    expect(app.loadOperationalData).toHaveBeenCalledWith(true);
  });

  it('refreshes sleep state immediately after a labeled camera snapshot', async () => {
    const app = new BabyMonitorApp() as unknown as DashboardHarness;
    app.loadOperationalData = vi.fn().mockResolvedValue(undefined);
    const frame: FrameRecord = {
      id: 'frame-after-wake',
      capturedAt: '2026-07-14T15:09:00Z',
      cameraEntityId: 'camera.nursery',
      locationId: 'granada',
      imageUrl: 'api/v1/frames/frame-after-wake/image',
      imageAvailable: true,
      mimeType: 'image/jpeg',
      sizeBytes: 123,
      label: {
        babyPresent: true,
        state: 'awake',
        confidence: 0.95,
        description: 'Baby is awake',
        tags: ['awake'],
        inCrib: true,
        faceVisible: 'yes',
        headSide: 'back',
        bodyPosition: 'back',
        clothingItems: ['short_sleeve_onesie'],
        pacifier: 'no',
        mouthOpen: 'no',
      },
      provider: 'gemini',
      model: 'gemini-test',
    };
    const refresh = vi.spyOn(api, 'refreshSnapshot').mockResolvedValue(frame);

    expect(await app.refreshSnapshot()).toEqual(frame);
    expect(app.loadOperationalData).toHaveBeenCalledWith(false);
    refresh.mockRestore();
  });

  it('shows duration above and start time below each recorded sleep segment', () => {
    const app = new BabyMonitorApp() as unknown as RhythmHarness;
    app.settings = cloneDefaultSettings();
    app.language = 'es';
    app.rhythmDate = '2026-07-10';
    app.rhythmMode = 'day';
    app.sleepEvents = [{
      id: 'nap-with-visible-labels',
      startedAt: '2026-07-10T10:00:00',
      endedAt: '2026-07-10T11:30:00',
      kind: 'nap',
      source: 'vision',
      notes: null,
      details: { tags: [], pauses: [] },
      locationId: 'granada',
    }];

    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderDailyRhythm(), container);

    expect(container.querySelector('.rhythm-marker-duration')?.textContent).toBe('1 h 30 min');
    expect(container.querySelector('.rhythm-marker-start')?.textContent).toBe('10:00');
  });
});
