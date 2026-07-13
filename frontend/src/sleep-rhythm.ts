import type { SleepEvent } from './types';

export type RhythmMode = 'day' | 'night';

export interface RhythmWindow {
  start: Date;
  end: Date;
}

export interface RhythmSegment {
  event: SleepEvent;
  start: Date;
  end: Date;
  startRatio: number;
  endRatio: number;
  minutes: number;
}

export interface RhythmModel {
  mode: RhythmMode;
  dateKey: string;
  window: RhythmWindow;
  segments: RhythmSegment[];
  totalMinutes: number;
  napMinutes: number;
  nightMinutes: number;
  wakeAt: Date | null;
  bedAt: Date | null;
}

function localDate(dateKey: string, hour = 0): Date {
  const [year, month, day] = dateKey.split('-').map(Number);
  return new Date(year, month - 1, day, hour, 0, 0, 0);
}

export function localDateKey(date: Date): string {
  const year = String(date.getFullYear()).padStart(4, '0');
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

export function shiftDateKey(dateKey: string, days: number): string {
  const date = localDate(dateKey, 12);
  date.setDate(date.getDate() + days);
  return localDateKey(date);
}

export function rhythmWindow(dateKey: string, mode: RhythmMode): RhythmWindow {
  if (mode === 'night') {
    const end = localDate(dateKey, 12);
    const start = localDate(dateKey, 18);
    start.setDate(start.getDate() - 1);
    return { start, end };
  }
  const start = localDate(dateKey);
  const end = localDate(dateKey);
  end.setDate(end.getDate() + 1);
  return { start, end };
}

function eventEnd(event: SleepEvent, now: Date): Date {
  return event.endedAt ? new Date(event.endedAt) : now;
}

function overlappingSegments(events: SleepEvent[], window: RhythmWindow, now: Date, mode: RhythmMode): RhythmSegment[] {
  const windowMs = window.end.getTime() - window.start.getTime();
  return events
    .filter((event) => mode === 'night' ? event.kind === 'night' : event.kind !== 'night')
    .map((event): RhythmSegment | null => {
      const rawStart = new Date(event.startedAt);
      const rawEnd = eventEnd(event, now);
      if (!Number.isFinite(rawStart.getTime()) || !Number.isFinite(rawEnd.getTime())) return null;
      const start = new Date(Math.max(rawStart.getTime(), window.start.getTime()));
      const end = new Date(Math.min(rawEnd.getTime(), window.end.getTime()));
      if (end <= start) return null;
      return {
        event,
        start,
        end,
        startRatio: (start.getTime() - window.start.getTime()) / windowMs,
        endRatio: (end.getTime() - window.start.getTime()) / windowMs,
        minutes: Math.max(1, Math.round((end.getTime() - start.getTime()) / 60_000)),
      };
    })
    .filter((segment): segment is RhythmSegment => Boolean(segment))
    .sort((a, b) => a.start.getTime() - b.start.getTime());
}

function nightBoundaries(events: SleepEvent[], dateKey: string, now: Date): { wakeAt: Date | null; bedAt: Date | null } {
  const day = rhythmWindow(dateKey, 'day');
  const nightEvents = events
    .filter((event) => event.kind === 'night')
    .map((event) => ({ start: new Date(event.startedAt), end: eventEnd(event, now) }))
    .filter(({ start, end }) => Number.isFinite(start.getTime()) && Number.isFinite(end.getTime()));

  const wakeAt = nightEvents
    .map(({ end }) => end)
    .filter((end) => end >= day.start && end <= new Date(day.start.getTime() + 12 * 60 * 60_000))
    .sort((a, b) => b.getTime() - a.getTime())[0] ?? null;
  const bedAt = nightEvents
    .map(({ start }) => start)
    .filter((start) => start >= new Date(day.start.getTime() + 18 * 60 * 60_000) && start < day.end)
    .sort((a, b) => a.getTime() - b.getTime())[0] ?? null;
  return { wakeAt, bedAt };
}

export function buildRhythmModel(
  events: SleepEvent[],
  dateKey: string,
  mode: RhythmMode,
  now = new Date(),
): RhythmModel {
  const window = rhythmWindow(dateKey, mode);
  const segments = overlappingSegments(events, window, now, mode);
  const totalMinutes = segments.reduce((total, segment) => total + segment.minutes, 0);
  const napMinutes = segments
    .filter((segment) => segment.event.kind !== 'night')
    .reduce((total, segment) => total + segment.minutes, 0);
  const nightMinutes = segments
    .filter((segment) => segment.event.kind === 'night')
    .reduce((total, segment) => total + segment.minutes, 0);

  if (mode === 'night') {
    return {
      mode,
      dateKey,
      window,
      segments,
      totalMinutes,
      napMinutes,
      nightMinutes,
      bedAt: segments[0]?.start ?? null,
      wakeAt: segments.at(-1)?.end ?? null,
    };
  }

  return {
    mode,
    dateKey,
    window,
    segments,
    totalMinutes,
    napMinutes,
    nightMinutes,
    ...nightBoundaries(events, dateKey, now),
  };
}

function polarPoint(ratio: number, radius: number): { x: number; y: number } {
  const angle = ratio * Math.PI * 2 - Math.PI / 2;
  return { x: 160 + radius * Math.cos(angle), y: 160 + radius * Math.sin(angle) };
}

export function rhythmArcPath(startRatio: number, endRatio: number, radius = 122): string {
  const start = Math.max(0, Math.min(1, startRatio));
  const end = Math.max(start, Math.min(0.999999, endRatio));
  const from = polarPoint(start, radius);
  const to = polarPoint(end, radius);
  const largeArc = end - start > 0.5 ? 1 : 0;
  return `M ${from.x.toFixed(3)} ${from.y.toFixed(3)} A ${radius} ${radius} 0 ${largeArc} 1 ${to.x.toFixed(3)} ${to.y.toFixed(3)}`;
}

export function rhythmMarkerPosition(segment: RhythmSegment): { x: number; y: number } {
  const point = polarPoint((segment.startRatio + segment.endRatio) / 2, 122);
  return { x: (point.x / 320) * 100, y: (point.y / 320) * 100 };
}
