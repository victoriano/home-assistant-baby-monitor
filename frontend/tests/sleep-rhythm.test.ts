import { describe, expect, it } from 'vitest';

import {
  buildRhythmModel,
  localDateKey,
  rhythmArcPath,
  rhythmWindow,
  shiftDateKey,
} from '../src/sleep-rhythm';
import type { SleepEvent, SleepKind } from '../src/types';

function at(dateKey: string, hour: number, minute = 0): string {
  const [year, month, day] = dateKey.split('-').map(Number);
  return new Date(year, month - 1, day, hour, minute).toISOString();
}

function sleep(id: string, startedAt: string, endedAt: string | null, kind: SleepKind): SleepEvent {
  return { id, startedAt, endedAt, kind, source: 'import', notes: null, locationId: 'madrid' };
}

describe('sleep rhythm model', () => {
  it('uses a local calendar day and a night ending on the selected date', () => {
    const day = rhythmWindow('2026-07-10', 'day');
    const night = rhythmWindow('2026-07-10', 'night');

    expect([day.start.getHours(), day.end.getHours()]).toEqual([0, 0]);
    expect(localDateKey(day.end)).toBe('2026-07-11');
    expect([night.start.getHours(), night.end.getHours()]).toEqual([18, 12]);
    expect(localDateKey(night.start)).toBe('2026-07-09');
    expect(localDateKey(night.end)).toBe('2026-07-10');
    expect(shiftDateKey('2026-07-10', -7)).toBe('2026-07-03');
  });

  it('restores the day view with naps plus morning and bedtime boundaries', () => {
    const events = [
      sleep('night-before', at('2026-07-09', 22, 30), at('2026-07-10', 7, 15), 'night'),
      sleep('nap-1', at('2026-07-10', 10), at('2026-07-10', 10, 45), 'nap'),
      sleep('nap-2', at('2026-07-10', 15), at('2026-07-10', 16, 30), 'nap'),
      sleep('night-after', at('2026-07-10', 21, 10), at('2026-07-11', 7), 'night'),
    ];

    const model = buildRhythmModel(events, '2026-07-10', 'day', new Date(at('2026-07-10', 17)));

    expect(model.segments.map((segment) => segment.event.id)).toEqual(['nap-1', 'nap-2']);
    expect(model.totalMinutes).toBe(135);
    expect(model.napMinutes).toBe(135);
    expect(model.wakeAt?.getHours()).toBe(7);
    expect(model.bedAt?.getHours()).toBe(21);
  });

  it('joins separate night periods inside the previous-evening-to-noon window', () => {
    const events = [
      sleep('night-1', at('2026-07-09', 22), at('2026-07-10', 1), 'night'),
      sleep('night-2', at('2026-07-10', 2), at('2026-07-10', 7, 30), 'night'),
      sleep('day-nap', at('2026-07-10', 9), at('2026-07-10', 10), 'nap'),
    ];

    const model = buildRhythmModel(events, '2026-07-10', 'night', new Date(at('2026-07-10', 10)));

    expect(model.segments.map((segment) => segment.event.id)).toEqual(['night-1', 'night-2']);
    expect(model.totalMinutes).toBe(510);
    expect(model.bedAt?.getHours()).toBe(22);
    expect(model.wakeAt?.getHours()).toBe(7);
    expect(model.wakeAt?.getMinutes()).toBe(30);
  });

  it('uses the current time for an ongoing sleep', () => {
    const ongoing = sleep('ongoing', at('2026-07-10', 14), null, 'nap');
    const model = buildRhythmModel([ongoing], '2026-07-10', 'day', new Date(at('2026-07-10', 15, 20)));

    expect(model.totalMinutes).toBe(80);
    expect(model.segments[0].event.endedAt).toBeNull();
  });

  it('creates a valid SVG arc for a sleep segment', () => {
    expect(rhythmArcPath(0.25, 0.5)).toMatch(/^M .+ A 122 122 0 0 1 /);
    expect(rhythmArcPath(0.1, 0.8)).toContain(' A 122 122 0 1 1 ');
  });
});
