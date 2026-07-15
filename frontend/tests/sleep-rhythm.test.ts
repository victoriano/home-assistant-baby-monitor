import { describe, expect, it } from 'vitest';

import {
  buildRhythmModel,
  localDateKey,
  rhythmArcPath,
  rhythmPosition,
  rhythmTrackPath,
  rhythmWindow,
  shiftDateKey,
} from '../src/sleep-rhythm';
import type { SleepEvent, SleepKind, SleepPlan } from '../src/types';

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
      sleep('night-before-1', at('2026-07-09', 22, 30), at('2026-07-10', 2), 'night'),
      sleep('night-before-2', at('2026-07-10', 2, 20), at('2026-07-10', 7, 15), 'night'),
      sleep('nap-1', at('2026-07-10', 10), at('2026-07-10', 10, 45), 'nap'),
      sleep('nap-2', at('2026-07-10', 15), at('2026-07-10', 16, 30), 'nap'),
      sleep('night-after', at('2026-07-10', 21, 10), at('2026-07-11', 7), 'night'),
    ];

    const model = buildRhythmModel(events, '2026-07-10', 'day', new Date(at('2026-07-10', 17)));

    expect(model.segments.map((segment) => segment.event?.id)).toEqual(['nap-1', 'nap-2']);
    expect(model.totalMinutes).toBe(135);
    expect(model.napMinutes).toBe(135);
    expect(model.wakeAt?.getHours()).toBe(7);
    expect(model.bedAt?.getHours()).toBe(21);
  });

  it('joins separate night periods inside the previous-evening-to-noon window', () => {
    const events = [
      sleep('night-1', at('2026-07-09', 22), at('2026-07-10', 1), 'night'),
      sleep('night-2', at('2026-07-10', 1, 30), at('2026-07-10', 7, 30), 'night'),
      sleep('day-nap', at('2026-07-10', 9), at('2026-07-10', 10), 'nap'),
    ];

    const model = buildRhythmModel(events, '2026-07-10', 'night', new Date(at('2026-07-10', 10)));

    expect(model.sleepSegments.map((segment) => segment.event?.id)).toEqual(['night-1', 'night-2']);
    expect(model.wakeGaps).toHaveLength(1);
    expect(model.wakeGaps[0]?.minutes).toBe(30);
    expect(model.wakeGaps[0]?.locationId).toBe('madrid');
    expect(model.segments.map((segment) => segment.type)).toEqual(['night', 'awake', 'night']);
    expect(model.totalMinutes).toBe(540);
    expect(model.bedAt?.getHours()).toBe(22);
    expect(model.wakeAt?.getHours()).toBe(7);
    expect(model.wakeAt?.getMinutes()).toBe(30);
  });

  it('keeps long overnight interruptions and a contiguous imported morning continuation', () => {
    const events = [
      sleep('evening-night', at('2026-07-10', 19, 30), at('2026-07-10', 20, 30), 'night'),
      sleep('main-night', at('2026-07-10', 23, 5), at('2026-07-11', 8), 'night'),
      sleep('imported-morning-continuation', at('2026-07-11', 8), at('2026-07-11', 9, 15), 'nap'),
    ];

    const model = buildRhythmModel(events, '2026-07-11', 'night', new Date(at('2026-07-11', 10)));

    expect(model.sleepSegments.map((item) => item.event?.id)).toEqual([
      'evening-night',
      'main-night',
      'imported-morning-continuation',
    ]);
    expect(model.wakeGaps).toHaveLength(1);
    expect(model.wakeGaps[0]?.minutes).toBe(155);
    expect(model.totalMinutes).toBe(670);
    expect(model.nightMinutes).toBe(595);
    expect(model.napMinutes).toBe(75);
  });

  it('preserves detector microseconds at inferred waking boundaries', () => {
    const events = [
      sleep(
        'before-gap',
        '2026-07-13T21:07:33.084208Z',
        '2026-07-14T00:28:13.196542Z',
        'night',
      ),
      sleep(
        'after-gap',
        '2026-07-14T03:28:07.485531Z',
        '2026-07-14T04:09:14.776632Z',
        'night',
      ),
    ];

    const model = buildRhythmModel(events, '2026-07-14', 'night', new Date('2026-07-14T08:00:00Z'));

    expect(model.wakeGaps).toHaveLength(1);
    expect(model.wakeGaps[0]?.evidenceStartedAt).toBe(events[0].endedAt);
    expect(model.wakeGaps[0]?.evidenceEndedAt).toBe(events[1].startedAt);
  });

  it('keeps the recorded night when a later morning nap is separated by a long gap', () => {
    const events = [
      sleep(
        'granada-night',
        '2026-07-14T21:21:57.052274Z',
        '2026-07-15T01:44:18.740963Z',
        'night',
      ),
      sleep(
        'granada-morning-nap',
        '2026-07-15T08:23:16.741416Z',
        '2026-07-15T09:09:32.119608Z',
        'nap',
      ),
    ];
    const now = new Date('2026-07-15T09:30:00Z');

    const night = buildRhythmModel(events, '2026-07-15', 'night', now);
    const day = buildRhythmModel(events, '2026-07-15', 'day', now);

    expect(night.sleepSegments.map((item) => item.event?.id)).toEqual(['granada-night']);
    expect(night.totalMinutes).toBe(262);
    expect(night.nightMinutes).toBe(262);
    expect(night.napMinutes).toBe(0);
    expect(night.bedAt?.toISOString()).toBe('2026-07-14T21:21:57.052Z');
    expect(night.wakeAt?.toISOString()).toBe('2026-07-15T01:44:18.740Z');

    expect(day.sleepSegments.map((item) => item.event?.id)).toEqual(['granada-morning-nap']);
    expect(day.totalMinutes).toBe(46);
    expect(day.napMinutes).toBe(46);
    expect(day.nightMinutes).toBe(0);
  });

  it('does not promote a morning nap into an otherwise empty night', () => {
    const nap = sleep(
      'morning-nap-only',
      at('2026-07-15', 10, 23),
      at('2026-07-15', 11, 9),
      'nap',
    );

    const model = buildRhythmModel([nap], '2026-07-15', 'night', new Date(at('2026-07-15', 11, 30)));

    expect(model.sleepSegments).toEqual([]);
    expect(model.totalMinutes).toBe(0);
    expect(model.napMinutes).toBe(0);
  });

  it('uses the current time for an ongoing sleep', () => {
    const ongoing = sleep('ongoing', at('2026-07-10', 14), null, 'nap');
    const model = buildRhythmModel([ongoing], '2026-07-10', 'day', new Date(at('2026-07-10', 15, 20)));

    expect(model.totalMinutes).toBe(80);
    expect(model.segments[0]?.event?.endedAt).toBeNull();
  });

  it('creates a valid SVG arc for a sleep segment', () => {
    expect(rhythmArcPath(0.25, 0.5)).toMatch(/^M .+ A 122 122 0 0 1 /);
    expect(rhythmArcPath(0.1, 0.8)).toContain(' A 122 122 0 1 1 ');
    expect(rhythmTrackPath()).toContain(' A 122 122 0 1 1 ');
    expect(rhythmPosition(0).x).toBeLessThan(50);
    expect(rhythmPosition(0).y).toBeGreaterThan(50);
    expect(rhythmPosition(1).x).toBeGreaterThan(50);
    expect(rhythmPosition(1).y).toBeGreaterThan(50);
  });

  it('subtracts structured awake pauses and keeps a long pause inside the same night', () => {
    const event = sleep('night', at('2026-07-09', 22), at('2026-07-10', 7), 'night');
    event.details = {
      tags: ['in_bed'],
      pauses: [{ startedAt: at('2026-07-10', 2), endedAt: at('2026-07-10', 3) }],
    };
    const model = buildRhythmModel([event], '2026-07-10', 'night', new Date(at('2026-07-10', 8)));

    expect(model.totalMinutes).toBe(480);
    expect(model.sleepSegments).toHaveLength(2);
    expect(model.wakeGaps).toHaveLength(1);
    expect(model.wakeGaps[0]?.minutes).toBe(60);
    expect(model.wakeGaps[0]?.inferred).toBe(false);
    expect(model.bedAt?.getHours()).toBe(22);
    expect(model.wakeAt?.getHours()).toBe(7);
  });

  it('shows predicted naps today and the full predicted night ending tomorrow', () => {
    const target = (kind: 'nap' | 'night', start: string, durationMinutes: number) => ({
      kind, label: kind, recommendedStart: start, windowStart: start, windowEnd: start,
      durationMinutes, confidence: 0.8, explanation: 'test',
    });
    const plan: SleepPlan = {
      generatedAt: at('2026-07-10', 9), ageBand: '12-17m', confidence: 0.8, reason: 'test',
      recentSampleCount: 8, wakeWindowMinutes: 240, wakeWindowMarginMinutes: 30,
      averageNapMinutes: 60, averageNightMinutes: 600, nextSleepAt: at('2026-07-10', 12),
      windowStart: at('2026-07-10', 11, 30), windowEnd: at('2026-07-10', 12, 30), nextKind: 'nap',
      plans: [{
        date: '2026-07-10', morningWakeAt: at('2026-07-10', 7), nightStartAt: at('2026-07-10', 20),
        nightEndAt: at('2026-07-11', 7), dayNapPredictions: [target('nap', at('2026-07-10', 12), 60)],
        nightPrediction: target('night', at('2026-07-10', 20), 660), explanation: 'test',
      }],
    };
    const day = buildRhythmModel([], '2026-07-10', 'day', new Date(at('2026-07-10', 9)), plan);
    const night = buildRhythmModel([], '2026-07-11', 'night', new Date(at('2026-07-10', 9)), plan);

    expect(day.predictedSegments.map((item) => item.type)).toEqual(['nap', 'night']);
    expect(day.wakePredicted).toBe(true);
    expect(day.bedPredicted).toBe(true);
    expect(night.predictedSegments).toHaveLength(1);
    expect(night.visualStart.getHours()).toBe(20);
    expect(night.visualEnd.getHours()).toBe(7);
  });
});
