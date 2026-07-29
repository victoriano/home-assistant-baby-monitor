import { render, type TemplateResult } from 'lit';
import { describe, expect, it } from 'vitest';

import { apiTesting } from '../src/api';
import { BabyMonitorApp } from '../src/baby-monitor-app';
import type { FrameRecord, Language } from '../src/types';

interface FrameDetailHarness {
  language: Language;
  selectedFrame: FrameRecord | null;
  renderFrame(frame: FrameRecord): TemplateResult;
  renderFrameDetail(): TemplateResult;
}

function labeledFrame(): FrameRecord {
  return apiTesting.normalizeFrame({
    id: 'frame-audit',
    captured_at: '2026-07-29T14:04:14Z',
    camera_entity_id: 'camera.boifun_granada',
    location_id: 'granada',
    image_url: '/api/v1/frames/frame-audit/image',
    image_available: true,
    mime_type: 'image/jpeg',
    size_bytes: 48_200,
    provider: 'yolo',
    model: 'baby-monitor-yolo-v1',
    label: {
      baby_present: true,
      state: 'asleep',
      confidence: 0.94731,
      description: 'Bebé dormido boca arriba con chupete.',
      tags: ['calm', 'night'],
      in_crib: true,
      sleep_surface: 'crib',
      face_visible: 'yes',
      head_side: 'back',
      body_position: 'supine',
      clothing_items: ['sleep_sack'],
      pacifier: 'yes',
      mouth_open: 'no',
      attention_score: 0.73,
    },
  });
}

describe('camera frame detail', () => {
  it('opens every camera card and exposes normalized plus original model metadata', () => {
    const app = new BabyMonitorApp() as unknown as FrameDetailHarness;
    app.language = 'es';
    const frame = labeledFrame();

    expect(frame.label?.raw?.attention_score).toBe(0.73);

    const cardHost = document.createElement('div');
    render(app.renderFrame(frame), cardHost);
    const trigger = cardHost.querySelector('.frame-card-trigger');
    if (!(trigger instanceof HTMLButtonElement)) throw new Error('Missing frame detail trigger');

    expect(trigger.getAttribute('aria-haspopup')).toBe('dialog');
    trigger.click();
    expect(app.selectedFrame?.id).toBe('frame-audit');

    const dialogHost = document.createElement('div');
    render(app.renderFrameDetail(), dialogHost);
    const dialog = dialogHost.querySelector('[role="dialog"]');
    expect(dialog?.getAttribute('aria-modal')).toBe('true');

    const metadata = dialogHost.querySelector('.frame-detail-metadata')?.textContent ?? '';
    expect(metadata).toContain('Bebé presente');
    expect(metadata).toContain('Estado');
    expect(metadata).toContain('Dormido');
    expect(metadata).toContain('Confianza');
    expect(metadata).toContain('94,731% · 0,94731');
    expect(metadata).toContain('Chupete');
    expect(metadata).toContain('Attention score');
    expect(metadata).toContain('0,73');

    const technicalKeys = [...dialogHost.querySelectorAll('.frame-detail-metadata code')]
      .map((node) => node.textContent);
    expect(technicalKeys).toEqual(expect.arrayContaining([
      'baby_present',
      'state',
      'confidence',
      'description',
      'tags',
      'in_crib',
      'sleep_surface',
      'face_visible',
      'head_side',
      'body_position',
      'clothing_items',
      'pacifier',
      'mouth_open',
      'attention_score',
    ]));

    const raw = dialogHost.querySelector('.frame-detail-technical pre')?.textContent ?? '';
    expect(raw).toContain('"attention_score": 0.73');
    expect(dialogHost.textContent).toContain('baby-monitor-yolo-v1');

    const close = dialogHost.querySelector('.frame-detail-close');
    if (!(close instanceof HTMLButtonElement)) throw new Error('Missing frame detail close action');
    close.click();
    expect(app.selectedFrame).toBeNull();
  });

  it('explains when a saved capture has no model label', () => {
    const app = new BabyMonitorApp() as unknown as FrameDetailHarness;
    app.language = 'es';
    app.selectedFrame = {
      ...labeledFrame(),
      label: null,
      provider: null,
      model: null,
    };

    const container = document.createElement('div');
    render(app.renderFrameDetail(), container);

    expect(container.querySelector('.frame-detail-empty')?.textContent)
      .toContain('el modelo no dejó metadatos');
    expect(container.querySelector('.frame-detail-metadata')).toBeNull();
  });
});
