import { render, type TemplateResult } from 'lit';
import { describe, expect, it } from 'vitest';

import { apiTesting } from '../src/api';
import { BabyMonitorApp } from '../src/baby-monitor-app';
import { cloneDefaultSettings, type AppSettings, type Language, type SleepPlan, type SleepPredictionTarget } from '../src/types';

interface PredictionHarness {
  language: Language;
  settings: AppSettings;
  sleepPlan: SleepPlan | null;
  selectedPrediction: SleepPredictionTarget | null;
  renderPredictionDialog(): TemplateResult;
}

function predictionPlan(): SleepPlan {
  return apiTesting.normalizeSleepPlan({
    generatedAt: '2026-07-15T12:00:00+02:00',
    ageBand: '12-17m', confidence: 0.87, recentSampleCount: 12,
    wakeWindowMinutes: 222, wakeWindowMarginMinutes: 41,
    averageNapMinutes: 55, averageNightMinutes: 620,
    modelDetails: {
      generatedAt: '2026-07-15T12:00:00+02:00', lookbackClosedSleepCount: 80,
      baseline: { ageBand: '12-17m', birthDateKnown: true, wakeWindowMinutes: 240, expectedNaps: 2 },
      wakeWindows: {
        count: 12, medianMinutes: 214, minMinutes: 175, maxMinutes: 285,
        valuesMinutes: [190, 214, 225], finalMinutes: 222,
        medianAbsoluteDeviationMinutes: 24, historyWeight: 0.7,
        samples: [{
          previousSleepId: 'night-1', previousSleepKind: 'night',
          previousSleepEndedAt: '2026-07-15T07:30:00+02:00',
          nextSleepId: 'nap-1', nextSleepKind: 'nap',
          nextSleepStartedAt: '2026-07-15T11:04:00+02:00', minutes: 214,
        }],
      },
      napDurations: {
        count: 6, medianMinutes: 55, minMinutes: 35, maxMinutes: 90,
        valuesMinutes: [45, 55, 70], finalMinutes: 55,
        samples: [{ eventId: 'nap-1', startedAt: '2026-07-15T11:04:00+02:00', endedAt: '2026-07-15T11:59:00+02:00', minutes: 55, source: 'vision' }],
      },
      bedtimes: { count: 5, medianMinuteOfDay: 1250, usedFallback: false, samples: [] },
      morningWakes: { count: 5, medianMinuteOfDay: 450, usedFallback: false, samples: [] },
      nightDurations: { count: 5, medianMinutes: 620, minMinutes: 570, maxMinutes: 690, valuesMinutes: [570, 620, 690], finalMinutes: 620, samples: [] },
      confidence: { value: 0.87, sampleCount: 12, rule: 'recent_wake_samples' },
    },
    plans: [{
      date: '2026-07-15', morningWakeAt: '2026-07-15T07:30:00+02:00',
      nightStartAt: '2026-07-15T20:50:00+02:00', nightEndAt: '2026-07-16T07:30:00+02:00',
      dayNapPredictions: [{
        kind: 'nap', label: 'Nap 2', recommendedStart: '2026-07-15T15:41:00+02:00',
        windowStart: '2026-07-15T15:00:00+02:00', windowEnd: '2026-07-15T16:22:00+02:00',
        durationMinutes: 55, confidence: 0.87, explanation: 'Recent rhythm',
        calculation: {
          method: 'wake_window', anchorAt: '2026-07-15T11:59:00+02:00',
          anchorType: 'last_observed_wake', baseRecommendedStart: '2026-07-15T15:41:00+02:00',
          adjustmentMinutes: 0, adjustmentReason: null, wakeWindowMinutes: 222,
          startSampleCount: 12, durationSampleCount: 6, plannedNapNumber: 2,
        },
      }],
      nightPrediction: {
        kind: 'night', label: 'Night sleep', recommendedStart: '2026-07-15T20:50:00+02:00',
        windowStart: '2026-07-15T20:20:00+02:00', windowEnd: '2026-07-15T21:20:00+02:00',
        durationMinutes: 640, confidence: 0.87, explanation: 'Recent bedtime',
      },
    }],
  });
}

describe('prediction detail', () => {
  it('shows a verifiable calculation receipt and exact model inputs', () => {
    const plan = predictionPlan();
    const app = new BabyMonitorApp() as unknown as PredictionHarness;
    app.language = 'es';
    app.settings = cloneDefaultSettings();
    app.settings.baby.name = 'Esteban';
    app.sleepPlan = plan;
    app.selectedPrediction = plan.plans[0].dayNapPredictions[0];

    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderPredictionDialog(), container);

    const receipt = container.querySelector('.prediction-receipt')?.textContent ?? '';
    expect(receipt).toContain('Cómo llegamos a las');
    expect(receipt).toContain('Último despertar observado');
    expect(receipt).toContain('3 h 42 min');
    expect(container.querySelector('.prediction-learning')?.textContent).toContain('70%');
    expect(container.querySelector('.prediction-formula')?.textContent).toContain('4 h');
    expect(container.querySelector('.prediction-formula')?.textContent).toContain('mínimo(92%');
    expect(container.querySelector('.prediction-samples')?.textContent).toContain('3 h 34 min');
    expect(container.querySelector('.prediction-local-note')?.textContent).toContain('no llama a Gemini');
  });
});
