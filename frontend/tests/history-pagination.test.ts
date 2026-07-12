import { render, type TemplateResult } from 'lit';
import { describe, expect, it, vi } from 'vitest';

import { api } from '../src/api';
import { BabyMonitorApp } from '../src/baby-monitor-app';
import type { CryEvent, FrameRecord, SleepEvent } from '../src/types';

type HistoryKind = 'sleep' | 'cry' | 'frames';
type HistoryPageState = { total: number; nextOffset: number; loading: boolean; error: string };

interface HistoryHarness {
  sleepEvents: SleepEvent[];
  cryEvents: CryEvent[];
  frames: FrameRecord[];
  historyPages: Record<HistoryKind, HistoryPageState>;
  loadMoreHistory(kind: HistoryKind): Promise<void>;
  renderHistoryPager(kind: HistoryKind, loaded: number): TemplateResult;
}

function sleep(id: string): SleepEvent {
  return {
    id,
    startedAt: '2026-07-10T20:00:00Z',
    endedAt: '2026-07-10T21:00:00Z',
    kind: 'nap',
    source: 'manual',
    notes: null,
    locationId: 'home',
  };
}

function cry(id: string): CryEvent {
  return {
    id,
    detectedAt: '2026-07-10T22:00:00Z',
    endedAt: null,
    source: 'binary_sensor',
    confidence: null,
    locationId: 'home',
  };
}

function frame(id: string): FrameRecord {
  return {
    id,
    capturedAt: '2026-07-10T23:00:00Z',
    cameraEntityId: 'camera.nursery',
    locationId: 'home',
    imageUrl: `/api/v1/frames/${id}/image`,
    imageAvailable: true,
    mimeType: 'image/jpeg',
    sizeBytes: 123,
    label: null,
    provider: null,
    model: null,
  };
}

function harness(): HistoryHarness {
  const app = new BabyMonitorApp() as unknown as HistoryHarness;
  app.sleepEvents = [sleep('sleep-1')];
  app.cryEvents = [cry('cry-1')];
  app.frames = [frame('frame-1')];
  app.historyPages = {
    sleep: { total: 2, nextOffset: 1, loading: false, error: '' },
    cry: { total: 2, nextOffset: 1, loading: false, error: '' },
    frames: { total: 2, nextOffset: 1, loading: false, error: '' },
  };
  return app;
}

describe('history load-more interactions', () => {
  it('loads the next sleep page from the server offset when the caregiver asks for older moments', async () => {
    const app = harness();
    const request = vi.spyOn(api, 'getSleep').mockResolvedValue({
      items: [sleep('sleep-2')], limit: 30, offset: 1, total: 2,
    });
    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderHistoryPager('sleep', app.sleepEvents.length), container);

    const button = container.querySelector('button');
    if (!(button instanceof HTMLButtonElement)) throw new Error('Missing load-more button');
    button.click();

    await vi.waitFor(() => expect(app.sleepEvents.map((event) => event.id)).toEqual(['sleep-1', 'sleep-2']));
    expect(request).toHaveBeenCalledWith(30, 1);
    expect(app.historyPages.sleep).toMatchObject({ total: 2, nextOffset: 2, loading: false, error: '' });
  });

  it('keeps independent offsets for cry events and camera frames', async () => {
    const app = harness();
    const cries = vi.spyOn(api, 'getCryEvents').mockResolvedValue({
      items: [cry('cry-2')], limit: 30, offset: 1, total: 2,
    });
    const frames = vi.spyOn(api, 'getFrames').mockResolvedValue({
      items: [frame('frame-2')], limit: 24, offset: 1, total: 2,
    });

    await app.loadMoreHistory('cry');
    await app.loadMoreHistory('frames');

    expect(cries).toHaveBeenCalledWith(30, 1);
    expect(frames).toHaveBeenCalledWith(24, 1);
    expect(app.cryEvents.map((event) => event.id)).toEqual(['cry-1', 'cry-2']);
    expect(app.frames.map((item) => item.id)).toEqual(['frame-1', 'frame-2']);
    expect(app.historyPages.cry.nextOffset).toBe(2);
    expect(app.historyPages.frames.nextOffset).toBe(2);
  });

  it('keeps loaded history visible and exposes a retryable error state', async () => {
    const app = harness();
    vi.spyOn(api, 'getSleep').mockRejectedValue(new Error('History service unavailable'));

    await app.loadMoreHistory('sleep');

    expect(app.sleepEvents.map((event) => event.id)).toEqual(['sleep-1']);
    expect(app.historyPages.sleep).toMatchObject({
      nextOffset: 1,
      loading: false,
      error: 'History service unavailable',
    });
    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderHistoryPager('sleep', app.sleepEvents.length), container);
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('History service unavailable');
    expect(container.querySelector('button')?.textContent).toContain('Try again');
  });
});
