import { render, type TemplateResult } from 'lit';
import { describe, expect, it } from 'vitest';

import { BabyMonitorApp } from '../src/baby-monitor-app';
import { formatTrendDate, medianMinutes } from '../src/trend-format';
import type { Language, SleepEvent } from '../src/types';

interface TrendsHarness {
  language: Language;
  sleepEvents: SleepEvent[];
  statsTab: 'summary' | 'naps' | 'awake' | 'night' | 'pacifier' | 'head' | 'clothing' | 'mouth';
  renderHistory(): TemplateResult;
}

const sleepEvents: SleepEvent[] = [
  { id: 'night-1', startedAt: '2026-07-03T22:00:00', endedAt: '2026-07-04T06:00:00', kind: 'night', source: 'manual', notes: null, details: { tags: [], pauses: [] }, locationId: 'granada' },
  { id: 'nap-1', startedAt: '2026-07-04T14:00:00', endedAt: '2026-07-04T15:00:00', kind: 'nap', source: 'manual', notes: null, details: { tags: [], pauses: [] }, locationId: 'granada' },
  { id: 'night-2', startedAt: '2026-07-04T21:00:00', endedAt: '2026-07-05T07:00:00', kind: 'night', source: 'manual', notes: null, details: { tags: [], pauses: [] }, locationId: 'granada' },
  { id: 'night-3', startedAt: '2026-07-05T22:00:00', endedAt: '2026-07-06T07:00:00', kind: 'night', source: 'manual', notes: null, details: { tags: [], pauses: [] }, locationId: 'granada' },
  { id: 'nap-3', startedAt: '2026-07-06T13:00:00', endedAt: '2026-07-06T15:00:00', kind: 'nap', source: 'manual', notes: null, details: { tags: [], pauses: [] }, locationId: 'granada' },
];

function harness(): TrendsHarness {
  const app = new BabyMonitorApp() as unknown as TrendsHarness;
  app.language = 'es';
  app.sleepEvents = sleepEvents;
  app.statsTab = 'summary';
  return app;
}

describe('trend presentation', () => {
  it('formats European dates and calculates medians', () => {
    expect(formatTrendDate('2026-07-05')).toBe('05/07');
    expect(formatTrendDate('not-a-date')).toBe('not-a-date');
    expect(medianMinutes([480, 600, 540])).toBe(540);
    expect(medianMinutes([0, 60, 120])).toBe(60);
    expect(medianMinutes([])).toBe(0);
  });

  it('renders daily bars as durations and summary cards as daily medians', () => {
    const app = harness();
    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderHistory(), container);

    const cards = [...container.querySelectorAll('.hero-stat')];
    expect(cards[0]?.querySelector('span')?.textContent).toBe('Sueño total');
    expect(cards[0]?.querySelector('strong')?.textContent).toBe('30 h');
    expect(cards[1]?.querySelector('span')?.textContent).toBe('Mediana nocturna');
    expect(cards[1]?.querySelector('strong')?.textContent).toBe('9 h');
    expect(cards[2]?.querySelector('span')?.textContent).toBe('Mediana diaria de siestas');
    expect(cards[2]?.querySelector('strong')?.textContent).toBe('1 h');

    const firstBar = container.querySelector('.legacy-chart-column');
    expect(firstBar?.querySelector('strong')?.textContent).toBe('9 h');
    expect(firstBar?.querySelector('small')?.textContent).toBe('04/07');
    expect(container.textContent).not.toContain('07-04');
  });

  it('uses the same European dates in wake and bedtime histories', () => {
    const app = harness();
    app.statsTab = 'night';
    const container = document.createElement('div');
    render(app.renderHistory(), container);

    expect([...container.querySelectorAll('.clock-history span')].map((node) => node.textContent)).toEqual(['04/07', '05/07', '06/07']);
    expect([...container.querySelectorAll('.legacy-chart-column strong')].map((node) => node.textContent)).toEqual(['8 h', '10 h', '9 h']);
  });
});
