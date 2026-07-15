import type { SleepDayPlan, SleepEvent, SleepPlan, SleepPredictionTarget } from './types';

export type RhythmMode = 'day' | 'night';
export type RhythmSegmentType = 'nap' | 'night' | 'awake';

export interface RhythmWindow { start: Date; end: Date }

export interface RhythmSegment {
  id: string;
  event: SleepEvent | null;
  prediction: SleepPredictionTarget | null;
  type: RhythmSegmentType;
  start: Date;
  end: Date;
  startRatio: number;
  endRatio: number;
  minutes: number;
  inferred: boolean;
  predicted: boolean;
}

export interface RhythmModel {
  mode: RhythmMode;
  dateKey: string;
  window: RhythmWindow;
  segments: RhythmSegment[];
  sleepSegments: RhythmSegment[];
  wakeGaps: RhythmSegment[];
  predictedSegments: RhythmSegment[];
  totalMinutes: number;
  napMinutes: number;
  nightMinutes: number;
  wakeAt: Date | null;
  bedAt: Date | null;
  wakePredicted: boolean;
  bedPredicted: boolean;
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

function eventEnd(event: SleepEvent, now: Date): Date {
  return event.endedAt ? new Date(event.endedAt) : now;
}

function minutesBetween(start: Date, end: Date): number {
  return Math.max(0, Math.round((end.getTime() - start.getTime()) / 60_000));
}

function segment(
  id: string,
  event: SleepEvent | null,
  prediction: SleepPredictionTarget | null,
  type: RhythmSegmentType,
  start: Date,
  end: Date,
  options: { inferred?: boolean; predicted?: boolean; minutes?: number } = {},
): RhythmSegment {
  return {
    id,
    event,
    prediction,
    type,
    start,
    end,
    startRatio: 0,
    endRatio: 0,
    minutes: options.minutes ?? Math.max(1, minutesBetween(start, end)),
    inferred: options.inferred ?? false,
    predicted: options.predicted ?? false,
  };
}

function actualInWindow(
  events: SleepEvent[],
  window: RhythmWindow,
  now: Date,
): { sleep: RhythmSegment[]; awake: RhythmSegment[] } {
  const sleep: RhythmSegment[] = [];
  const awake: RhythmSegment[] = [];
  for (const event of events) {
    const rawStart = new Date(event.startedAt);
    const rawEnd = eventEnd(event, now);
    if (!Number.isFinite(rawStart.getTime()) || !Number.isFinite(rawEnd.getTime())) continue;
    const clippedStart = new Date(Math.max(rawStart.getTime(), window.start.getTime()));
    const clippedEnd = new Date(Math.min(rawEnd.getTime(), window.end.getTime()));
    if (clippedEnd <= clippedStart) continue;
    if (event.kind === 'awake') {
      awake.push(segment(event.id, event, null, 'awake', clippedStart, clippedEnd));
      continue;
    }
    if (event.kind !== 'nap' && event.kind !== 'night') continue;
    const sleepType: RhythmSegmentType = event.kind === 'nap' ? 'nap' : 'night';

    const pauses = (event.details?.pauses ?? [])
      .map((pause) => ({ start: new Date(pause.startedAt), end: new Date(pause.endedAt) }))
      .filter((pause) => pause.end > clippedStart && pause.start < clippedEnd)
      .sort((a, b) => a.start.getTime() - b.start.getTime());
    let cursor = clippedStart;
    pauses.forEach((pause, index) => {
      const pauseStart = new Date(Math.max(cursor.getTime(), pause.start.getTime(), clippedStart.getTime()));
      const pauseEnd = new Date(Math.min(pause.end.getTime(), clippedEnd.getTime()));
      if (pauseStart > cursor) {
        sleep.push(segment(`${event.id}-sleep-${index}`, event, null, sleepType, cursor, pauseStart));
      }
      if (pauseEnd > pauseStart) {
        awake.push(segment(`${event.id}-pause-${index}`, event, null, 'awake', pauseStart, pauseEnd));
        cursor = pauseEnd;
      }
    });
    if (cursor < clippedEnd) {
      sleep.push(segment(`${event.id}-sleep-end`, event, null, sleepType, cursor, clippedEnd));
    }
  }
  sleep.sort((a, b) => a.start.getTime() - b.start.getTime());
  awake.sort((a, b) => a.start.getTime() - b.start.getTime());
  return { sleep, awake };
}

function nightCluster(
  events: SleepEvent[],
  dateKey: string,
  now: Date,
): { sleep: RhythmSegment[]; awake: RhythmSegment[]; window: RhythmWindow } {
  const window = rhythmWindow(dateKey, 'night');
  const actual = actualInWindow(events, window, now);
  const all = actual.sleep;
  const nightIndexes = all.flatMap((item, index) => (item.type === 'night' ? [index] : []));
  if (!nightIndexes.length) return { sleep: [], awake: [], window };

  // Anchor the view to a recorded night event. Naps share this window because
  // it ends at noon; they may extend an already-running night (legacy imports
  // can split an uninterrupted morning continuation at 08:00), but a later nap
  // must never become the night by itself. Separate night clusters more than
  // four hours apart are treated as distinct, and the latest cluster wins.
  let startIndex = nightIndexes[0];
  for (const nightIndex of nightIndexes.slice(1)) {
    const previous = all[nightIndex - 1];
    const item = all[nightIndex];
    const sameSleepWithPause = Boolean(previous?.event && previous.event.id === item.event?.id);
    if (previous && minutesBetween(previous.end, item.start) > 240 && !sameSleepWithPause) {
      startIndex = nightIndex;
    }
  }
  const sleep: RhythmSegment[] = [];
  for (let index = startIndex; index < all.length; index += 1) {
    const item = all[index];
    const previous = sleep.at(-1);
    if (previous) {
      const sameSleepWithPause = Boolean(previous.event && previous.event.id === item.event?.id);
      const gap = minutesBetween(previous.end, item.start);
      if (!sameSleepWithPause && item.type === 'nap' && gap > 40) break;
      if (!sameSleepWithPause && item.type === 'night' && gap > 240) break;
    }
    sleep.push(item);
  }
  const start = sleep[0]?.start;
  const end = sleep.at(-1)?.end;
  const awake = start && end
    ? actual.awake.filter((item) => item.end > start && item.start < end)
    : [];
  return { sleep, awake, window };
}

function awakeGaps(sleep: RhythmSegment[], explicitAwake: RhythmSegment[]): RhythmSegment[] {
  const inferred = sleep.slice(0, -1).map((item, index): RhythmSegment | null => {
    const next = sleep[index + 1];
    const minutes = minutesBetween(item.end, next.start);
    if (minutes < 10 || minutes > 240) return null;
    const represented = explicitAwake.some((awake) => awake.end > item.end && awake.start < next.start);
    if (represented) return null;
    return segment(
      `awake-${item.end.toISOString()}-${next.start.toISOString()}`,
      null,
      null,
      'awake',
      item.end,
      next.start,
      { inferred: true },
    );
  }).filter((item): item is RhythmSegment => Boolean(item));
  return [...explicitAwake, ...inferred].sort((a, b) => a.start.getTime() - b.start.getTime());
}

function applyGeometry(segments: RhythmSegment[], start: Date, end: Date): void {
  const span = Math.max(1, end.getTime() - start.getTime());
  for (const item of segments) {
    item.startRatio = Math.max(0, Math.min(1, (item.start.getTime() - start.getTime()) / span));
    item.endRatio = Math.max(item.startRatio, Math.min(1, (item.end.getTime() - start.getTime()) / span));
  }
}

function planForDate(plan: SleepPlan | null, dateKey: string): SleepDayPlan | null {
  return plan?.plans.find((item) => item.date === dateKey) ?? null;
}

function targetDates(target: SleepPredictionTarget): { start: Date; end: Date } | null {
  const start = new Date(target.recommendedStart);
  const end = new Date(start.getTime() + target.durationMinutes * 60_000);
  return Number.isFinite(start.getTime()) && end > start ? { start, end } : null;
}

function targetIsVisible(target: SleepPredictionTarget, dateKey: string, now: Date): boolean {
  if (dateKey > localDateKey(now)) return true;
  if (dateKey < localDateKey(now)) return false;
  const windowEnd = new Date(target.windowEnd);
  return !Number.isFinite(windowEnd.getTime()) || windowEnd > now;
}

function overlapsRecorded(target: SleepPredictionTarget, sleep: RhythmSegment[]): boolean {
  const dates = targetDates(target);
  if (!dates) return false;
  return sleep.some((item) => item.end > dates.start && item.start < dates.end);
}

function predictedNapSegments(
  dayPlan: SleepDayPlan | null,
  dateKey: string,
  actual: RhythmSegment[],
  now: Date,
): RhythmSegment[] {
  if (!dayPlan) return [];
  return dayPlan.dayNapPredictions.flatMap((target, index) => {
    const dates = targetDates(target);
    if (!dates || !targetIsVisible(target, dateKey, now) || overlapsRecorded(target, actual)) return [];
    return [segment(
      `prediction-${dateKey}-nap-${index}`,
      null,
      target,
      'nap',
      dates.start,
      dates.end,
      { predicted: true, minutes: target.durationMinutes },
    )];
  });
}

export function buildRhythmModel(
  events: SleepEvent[],
  dateKey: string,
  mode: RhythmMode,
  now = new Date(),
  plan: SleepPlan | null = null,
): RhythmModel {
  if (mode === 'night') {
    const cluster = nightCluster(events, dateKey, now);
    const dayPlan = planForDate(plan, shiftDateKey(dateKey, -1));
    const target = dayPlan?.nightPrediction ?? null;
    const targetRange = target ? targetDates(target) : null;
    const showPrediction = Boolean(
      target
      && targetRange
      && dateKey >= localDateKey(now)
      && (!cluster.sleep.length || cluster.sleep.some((item) => !item.event?.endedAt)),
    );
    const predicted = showPrediction && target && targetRange
      ? [segment(
        `prediction-${dateKey}-night`,
        null,
        target,
        'night',
        cluster.sleep.length ? cluster.sleep.at(-1)!.end : targetRange.start,
        targetRange.end,
        { predicted: true, minutes: target.durationMinutes },
      )].filter((item) => item.end > item.start)
      : [];
    const actualWake = awakeGaps(cluster.sleep, cluster.awake);
    const visualStart = cluster.sleep[0]?.start ?? predicted[0]?.start ?? cluster.window.start;
    const visualEnd = predicted.at(-1)?.end ?? cluster.sleep.at(-1)?.end ?? cluster.window.end;
    const segments = [...cluster.sleep, ...actualWake, ...predicted]
      .sort((a, b) => a.start.getTime() - b.start.getTime());
    applyGeometry(segments, visualStart, visualEnd);
    const midnight = localDate(dateKey);
    const midnightRatio = midnight >= visualStart && midnight <= visualEnd
      ? (midnight.getTime() - visualStart.getTime()) / Math.max(1, visualEnd.getTime() - visualStart.getTime())
      : null;
    return {
      mode,
      dateKey,
      window: cluster.window,
      segments,
      sleepSegments: cluster.sleep,
      wakeGaps: actualWake,
      predictedSegments: predicted,
      totalMinutes: cluster.sleep.reduce((sum, item) => sum + item.minutes, 0),
      napMinutes: cluster.sleep.filter((item) => item.type === 'nap').reduce((sum, item) => sum + item.minutes, 0),
      nightMinutes: cluster.sleep.filter((item) => item.type === 'night').reduce((sum, item) => sum + item.minutes, 0),
      bedAt: cluster.sleep[0]?.start ?? targetRange?.start ?? null,
      wakeAt: cluster.sleep.at(-1)?.end ?? targetRange?.end ?? null,
      bedPredicted: !cluster.sleep.length && Boolean(targetRange),
      wakePredicted: Boolean(predicted.length) || (!cluster.sleep.length && Boolean(targetRange)),
      visualStart,
      visualEnd,
      midnightRatio,
    };
  }

  const window = rhythmWindow(dateKey, 'day');
  const all = actualInWindow(events, window, now).sleep;
  const night = nightCluster(events, dateKey, now).sleep;
  const sameInterval = (candidate: RhythmSegment): boolean => night.some((item) => (
    item.event?.id === candidate.event?.id
    || (item.start.getTime() === candidate.start.getTime() && item.end.getTime() === candidate.end.getTime())
  ));
  const sleep = all.filter((item) => item.type === 'nap' && !sameInterval(item));
  const previousNightWake = night.at(-1)?.end ?? null;
  const actualBed = all.find((item) => item.type === 'night' && item.start.getHours() >= 18)?.start ?? null;
  const dayPlan = planForDate(plan, dateKey);
  const predictedNaps = predictedNapSegments(dayPlan, dateKey, sleep, now);
  const nightTarget = dayPlan?.nightPrediction ?? null;
  const nightRange = nightTarget ? targetDates(nightTarget) : null;
  const showBedPrediction = Boolean(
    nightTarget && nightRange && !actualBed && targetIsVisible(nightTarget, dateKey, now),
  );
  const predictedBed = showBedPrediction && nightTarget && nightRange
    ? segment(
      `prediction-${dateKey}-bedtime`,
      null,
      nightTarget,
      'night',
      new Date(nightRange.start.getTime() - 10 * 60_000),
      nightRange.start,
      { predicted: true, minutes: nightTarget.durationMinutes },
    )
    : null;
  const predicted = [...predictedNaps, ...(predictedBed ? [predictedBed] : [])];
  const plannedWake = dayPlan?.morningWakeAt ? new Date(dayPlan.morningWakeAt) : null;
  const wakeAt = previousNightWake ?? (plannedWake && Number.isFinite(plannedWake.getTime()) ? plannedWake : null);
  const bedAt = actualBed ?? nightRange?.start ?? null;
  const visualStart = wakeAt ?? localDate(dateKey, 7);
  let visualEnd = bedAt ?? localDate(dateKey, 22);
  if (!bedAt) visualEnd.setMinutes(30);
  if (visualEnd <= visualStart) visualEnd = new Date(visualStart.getTime() + 14 * 60 * 60_000);
  const segments = [...sleep, ...predicted].sort((a, b) => a.start.getTime() - b.start.getTime());
  applyGeometry(segments, visualStart, visualEnd);
  return {
    mode,
    dateKey,
    window,
    segments,
    sleepSegments: sleep,
    wakeGaps: [],
    predictedSegments: predicted,
    totalMinutes: sleep.reduce((sum, item) => sum + item.minutes, 0),
    napMinutes: sleep.reduce((sum, item) => sum + item.minutes, 0),
    nightMinutes: 0,
    wakeAt,
    bedAt,
    wakePredicted: !previousNightWake && Boolean(wakeAt),
    bedPredicted: !actualBed && Boolean(bedAt),
    visualStart,
    visualEnd,
    midnightRatio: null,
  };
}

function polarPoint(ratio: number, radius: number): { x: number; y: number } {
  const angle = ((225 + Math.max(0, Math.min(1, ratio)) * 270) * Math.PI) / 180;
  return {
    x: 160 + radius * Math.sin(angle),
    y: 160 - radius * Math.cos(angle),
  };
}

export function rhythmArcPath(startRatio: number, endRatio: number, radius = 122): string {
  const start = Math.max(0, Math.min(1, startRatio));
  const end = Math.max(start + 0.0005, Math.min(1, endRatio));
  const from = polarPoint(start, radius);
  const to = polarPoint(end, radius);
  const largeArc = (end - start) * 270 > 180 ? 1 : 0;
  return `M ${from.x.toFixed(3)} ${from.y.toFixed(3)} A ${radius} ${radius} 0 ${largeArc} 1 ${to.x.toFixed(3)} ${to.y.toFixed(3)}`;
}

export function rhythmTrackPath(radius = 122): string {
  return rhythmArcPath(0, 1, radius);
}

export function rhythmSvgPoint(ratio: number, radius: number): { x: number; y: number } {
  return polarPoint(ratio, radius);
}

export function rhythmPosition(ratio: number, radius = 122): { x: number; y: number } {
  const point = polarPoint(ratio, radius);
  return {
    x: 6 + (point.x / 320) * 88,
    y: 6 + (point.y / 320) * 88,
  };
}

export function rhythmMarkerPosition(item: RhythmSegment): { x: number; y: number } {
  return rhythmPosition((item.startRatio + item.endRatio) / 2);
}
