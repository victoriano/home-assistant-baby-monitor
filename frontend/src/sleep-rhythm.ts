import type { SleepEvent } from './types';

export type RhythmMode = 'day' | 'night';
export type RhythmSegmentType = 'nap' | 'night' | 'awake';

export interface RhythmWindow { start: Date; end: Date }

export interface RhythmSegment {
  id: string;
  event: SleepEvent | null;
  type: RhythmSegmentType;
  start: Date;
  end: Date;
  startRatio: number;
  endRatio: number;
  minutes: number;
  inferred: boolean;
}

export interface RhythmModel {
  mode: RhythmMode;
  dateKey: string;
  window: RhythmWindow;
  segments: RhythmSegment[];
  sleepSegments: RhythmSegment[];
  wakeGaps: RhythmSegment[];
  totalMinutes: number;
  napMinutes: number;
  nightMinutes: number;
  wakeAt: Date | null;
  bedAt: Date | null;
  visualStart: Date;
  visualEnd: Date;
  midnightRatio: number | null;
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

function eventEnd(event: SleepEvent, now: Date): Date { return event.endedAt ? new Date(event.endedAt) : now; }
function minutesBetween(start: Date, end: Date): number { return Math.max(0, Math.round((end.getTime() - start.getTime()) / 60_000)); }

function sleepInWindow(events: SleepEvent[], window: RhythmWindow, now: Date): RhythmSegment[] {
  return events.map((event): RhythmSegment | null => {
    if (event.kind !== 'nap' && event.kind !== 'night') return null;
    const rawStart = new Date(event.startedAt);
    const rawEnd = eventEnd(event, now);
    if (!Number.isFinite(rawStart.getTime()) || !Number.isFinite(rawEnd.getTime())) return null;
    const start = new Date(Math.max(rawStart.getTime(), window.start.getTime()));
    const end = new Date(Math.min(rawEnd.getTime(), window.end.getTime()));
    if (end <= start) return null;
    return { id: event.id, event, type: event.kind, start, end, startRatio: 0, endRatio: 0, minutes: Math.max(1, minutesBetween(start, end)), inferred: false };
  }).filter((segment): segment is RhythmSegment => Boolean(segment)).sort((a, b) => a.start.getTime() - b.start.getTime());
}

function nightCluster(events: SleepEvent[], dateKey: string, now: Date): { sleep: RhythmSegment[]; window: RhythmWindow } {
  const window = rhythmWindow(dateKey, 'night');
  const all = sleepInWindow(events, window, now);
  if (!all.length) return { sleep: [], window };
  const eveningStart = localDate(shiftDateKey(dateKey, -1), 20);
  const morningCutoff = localDate(dateKey, 9);
  let startIndex = all.findIndex((event) => event.start >= eveningStart);
  if (startIndex < 0) startIndex = 0;
  while (startIndex < all.length - 1) {
    const current = all[startIndex];
    const next = all[startIndex + 1];
    const manualBoundary = current.event?.source === 'manual' && current.event.notes?.startsWith('manual-boundary');
    if (minutesBetween(current.end, next.start) > 40 && current.end < morningCutoff && !manualBoundary) { startIndex += 1; continue; }
    break;
  }
  const sleep: RhythmSegment[] = [];
  for (let index = startIndex; index < all.length; index += 1) {
    const event = all[index];
    const previous = sleep.at(-1);
    if (previous) {
      const crossedMorning = previous.end >= morningCutoff || event.start >= morningCutoff;
      if (minutesBetween(previous.end, event.start) > 40 && crossedMorning) break;
    }
    sleep.push(event);
  }
  return { sleep, window };
}

function awakeGaps(sleep: RhythmSegment[]): RhythmSegment[] {
  return sleep.slice(0, -1).map((event, index): RhythmSegment | null => {
    const next = sleep[index + 1];
    const minutes = minutesBetween(event.end, next.start);
    if (minutes < 10 || minutes > 240) return null;
    return { id: `awake-${event.end.toISOString()}-${next.start.toISOString()}`, event: null, type: 'awake', start: event.end, end: next.start, startRatio: 0, endRatio: 0, minutes, inferred: true };
  }).filter((segment): segment is RhythmSegment => Boolean(segment));
}

function applyGeometry(segments: RhythmSegment[], start: Date, end: Date): void {
  const span = Math.max(1, end.getTime() - start.getTime());
  for (const segment of segments) {
    segment.startRatio = Math.max(0, Math.min(1, (segment.start.getTime() - start.getTime()) / span));
    segment.endRatio = Math.max(segment.startRatio, Math.min(1, (segment.end.getTime() - start.getTime()) / span));
  }
}

export function buildRhythmModel(events: SleepEvent[], dateKey: string, mode: RhythmMode, now = new Date()): RhythmModel {
  if (mode === 'night') {
    const { sleep, window } = nightCluster(events, dateKey, now);
    const gaps = awakeGaps(sleep);
    const visualStart = sleep[0]?.start ?? window.start;
    const visualEnd = sleep.at(-1)?.end ?? window.end;
    const segments = [...sleep, ...gaps].sort((a, b) => a.start.getTime() - b.start.getTime());
    applyGeometry(segments, visualStart, visualEnd);
    const midnight = localDate(dateKey);
    const midnightRatio = midnight >= visualStart && midnight <= visualEnd ? (midnight.getTime() - visualStart.getTime()) / Math.max(1, visualEnd.getTime() - visualStart.getTime()) : null;
    return {
      mode, dateKey, window, segments, sleepSegments: sleep, wakeGaps: gaps,
      totalMinutes: sleep.reduce((sum, item) => sum + item.minutes, 0),
      napMinutes: sleep.filter((item) => item.type === 'nap').reduce((sum, item) => sum + item.minutes, 0),
      nightMinutes: sleep.filter((item) => item.type === 'night').reduce((sum, item) => sum + item.minutes, 0),
      bedAt: sleep[0]?.start ?? null, wakeAt: sleep.at(-1)?.end ?? null, visualStart, visualEnd, midnightRatio,
    };
  }

  const window = rhythmWindow(dateKey, 'day');
  const all = sleepInWindow(events, window, now);
  const night = nightCluster(events, dateKey, now).sleep;
  const sameInterval = (candidate: RhythmSegment): boolean => night.some((item) => item.event?.id === candidate.event?.id || (item.start.getTime() === candidate.start.getTime() && item.end.getTime() === candidate.end.getTime()));
  const sleep = all.filter((item) => item.type === 'nap' && !sameInterval(item));
  const wakeAt = night.at(-1)?.end ?? null;
  const bedAt = all.find((item) => item.start.getHours() >= 18 && item.minutes >= 90)?.start ?? null;
  const visualStart = wakeAt ?? localDate(dateKey, 7);
  let visualEnd = bedAt ?? localDate(dateKey, 22);
  if (!bedAt) visualEnd.setMinutes(30);
  if (visualEnd <= visualStart) visualEnd = new Date(visualStart.getTime() + 14 * 60 * 60_000);
  applyGeometry(sleep, visualStart, visualEnd);
  return {
    mode, dateKey, window, segments: sleep, sleepSegments: sleep, wakeGaps: [],
    totalMinutes: sleep.reduce((sum, item) => sum + item.minutes, 0),
    napMinutes: sleep.reduce((sum, item) => sum + item.minutes, 0), nightMinutes: 0,
    wakeAt, bedAt, visualStart, visualEnd, midnightRatio: null,
  };
}

function polarPoint(ratio: number, radius: number): { x: number; y: number } {
  const angle = ((225 + Math.max(0, Math.min(1, ratio)) * 270) * Math.PI) / 180;
  return { x: 160 + radius * Math.cos(angle), y: 160 + radius * Math.sin(angle) };
}

export function rhythmArcPath(startRatio: number, endRatio: number, radius = 122): string {
  const start = Math.max(0, Math.min(1, startRatio));
  const end = Math.max(start, Math.min(0.999999, endRatio));
  const from = polarPoint(start, radius);
  const to = polarPoint(end, radius);
  const largeArc = (end - start) * 270 > 180 ? 1 : 0;
  return `M ${from.x.toFixed(3)} ${from.y.toFixed(3)} A ${radius} ${radius} 0 ${largeArc} 1 ${to.x.toFixed(3)} ${to.y.toFixed(3)}`;
}

export function rhythmMarkerPosition(segment: RhythmSegment): { x: number; y: number } {
  const point = polarPoint((segment.startRatio + segment.endRatio) / 2, 122);
  return { x: (point.x / 320) * 100, y: (point.y / 320) * 100 };
}
