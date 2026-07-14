import { render, type TemplateResult } from 'lit';
import { describe, expect, it, vi } from 'vitest';

import { api } from '../src/api';
import { BabyMonitorApp } from '../src/baby-monitor-app';
import { cloneDefaultSettings, type AppSettings, type FrameRecord } from '../src/types';

interface DashboardHarness {
  settings: AppSettings;
  renderDashboard(): TemplateResult;
  loadOperationalData(showSpinner?: boolean): Promise<void>;
  refreshSnapshot(): Promise<FrameRecord | null>;
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
});
