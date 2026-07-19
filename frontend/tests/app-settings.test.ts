import { render, type TemplateResult } from 'lit';
import { describe, expect, it, vi } from 'vitest';

import { api } from '../src/api';
import { BabyMonitorApp } from '../src/baby-monitor-app';
import {
  cloneDefaultSettings,
  settingsToPayload,
  type AppSettings,
  type HistoryTransferStatus,
  type SecretName,
} from '../src/types';

interface SettingsHarness {
  draft: AppSettings;
  settings: AppSettings;
  language: 'en' | 'es';
  birthDatePickerOpen: boolean;
  birthDatePickerMonth: string;
  pendingSecretClears: SecretName[];
  historyTransfer: HistoryTransferStatus | null;
  entities: {
    camera: never[];
    binary_sensor: never[];
    light: never[];
    notify: Array<{ entityId: string; name: string; available: boolean; attributes: Record<string, unknown> }>;
    person: Array<{ entityId: string; name: string; available: boolean; attributes: Record<string, unknown> }>;
  };
  renderCameraSection(compact?: boolean): TemplateResult;
  renderHomeAssistantSection(compact?: boolean): TemplateResult;
  renderNotificationsSection(compact?: boolean): TemplateResult;
  renderProfileSection(compact?: boolean): TemplateResult;
  renderVisionSection(compact?: boolean): TemplateResult;
  renderHistoryTransferSection(): TemplateResult;
  homeAssistantValidationError(): string;
  validationError(stage: number | 'all'): string;
}

function harness(settings = cloneDefaultSettings()): SettingsHarness {
  const app = new BabyMonitorApp() as unknown as SettingsHarness;
  app.settings = structuredClone(settings);
  app.draft = structuredClone(settings);
  app.pendingSecretClears = [];
  app.historyTransfer = null;
  return app;
}

function buttonNamed(name: string): HTMLButtonElement {
  const button = [...document.querySelectorAll('button')]
    .find((candidate) => candidate.textContent?.includes(name));
  if (!(button instanceof HTMLButtonElement)) throw new Error(`Missing button: ${name}`);
  return button;
}

function renderSettings(template: TemplateResult): void {
  const container = document.createElement('div');
  document.body.append(container);
  render(template, container);
}

describe('settings safety interactions', () => {
  it('uses a compact custom birth-date picker and stores an exact calendar date', () => {
    const settings = cloneDefaultSettings();
    settings.baby.birthDate = '2025-07-05';
    const app = harness(settings);
    app.language = 'es';

    renderSettings(app.renderProfileSection(true));

    const trigger = document.querySelector('.profile-date-trigger');
    if (!(trigger instanceof HTMLButtonElement)) throw new Error('Missing birth-date trigger');
    expect(document.querySelector('input[type="date"]')).toBeNull();
    expect(trigger.textContent).toContain('05/07/2025');

    trigger.click();
    expect(app.birthDatePickerOpen).toBe(true);
    expect(app.birthDatePickerMonth).toBe('2025-07-01');

    document.body.replaceChildren();
    renderSettings(app.renderProfileSection(true));
    const selected = document.querySelector('button[data-date="2025-07-05"]');
    const replacement = document.querySelector('button[data-date="2025-07-12"]');
    expect(selected?.classList.contains('selected')).toBe(true);
    if (!(replacement instanceof HTMLButtonElement)) throw new Error('Missing replacement calendar date');
    replacement.click();

    expect(app.draft.baby.birthDate).toBe('2025-07-12');
    expect(app.birthDatePickerOpen).toBe(false);

    document.body.replaceChildren();
    renderSettings(app.renderProfileSection(true));
    const updatedTrigger = document.querySelector('.profile-date-trigger');
    if (!(updatedTrigger instanceof HTMLButtonElement)) throw new Error('Missing updated birth-date trigger');
    expect(updatedTrigger.textContent).toContain('12/07/2025');
    updatedTrigger.click();

    document.body.replaceChildren();
    renderSettings(app.renderProfileSection(true));
    buttonNamed('Borrar fecha').click();
    expect(app.draft.baby.birthDate).toBeNull();
  });

  it('clears a local endpoint and cloud consent when the AI provider changes', () => {
    const settings = cloneDefaultSettings();
    settings.ai.provider = 'local';
    settings.ai.baseUrl = 'http://ollama.local:11434/v1';
    settings.ai.model = 'qwen-vl';
    settings.ai.apiKeyConfigured = true;
    settings.ai.cloudImageConsent = true;
    const app = harness(settings);

    renderSettings(app.renderVisionSection(true));
    buttonNamed('OpenAI').click();

    expect(app.draft.ai.provider).toBe('openai');
    expect(app.draft.ai.baseUrl).toBeNull();
    expect(app.draft.ai.cloudImageConsent).toBe(false);
    expect(app.draft.ai.model).toBe('gpt-5.6-luna');
  });

  it('describes OpenAI-compatible endpoints honestly and resets consent when their URL changes', () => {
    const settings = cloneDefaultSettings();
    settings.ai.provider = 'local';
    settings.ai.baseUrl = 'https://vision.example.test/v1';
    settings.ai.model = 'vision-model';
    settings.ai.cloudImageConsent = true;
    const app = harness(settings);

    renderSettings(app.renderVisionSection(true));
    expect(document.body.textContent).toContain('OpenAI-compatible endpoint');
    expect(document.body.textContent).toContain('self-hosted or remote');
    expect(document.body.textContent).toContain('https://vision.example.test/v1');
    expect(document.body.textContent).not.toContain('Local / OpenAI-compatible');

    const endpoint = document.querySelector('input[type="url"]');
    if (!(endpoint instanceof HTMLInputElement)) throw new Error('Missing compatible endpoint input');
    endpoint.value = 'https://different.example.test/v1';
    endpoint.dispatchEvent(new Event('input', { bubbles: true }));

    expect(app.draft.ai.baseUrl).toBe('https://different.example.test/v1');
    expect(app.draft.ai.cloudImageConsent).toBe(false);
    expect(app.validationError(2)).toContain('Confirm image sharing');
  });

  it('requires explicit image consent for an OpenAI-compatible endpoint and persists it only after checking', () => {
    const settings = cloneDefaultSettings();
    settings.ai.provider = 'local';
    settings.ai.baseUrl = 'http://ollama.local:11434/v1';
    settings.ai.model = 'vision-model';
    const app = harness(settings);

    renderSettings(app.renderVisionSection(true));
    const consent = document.querySelector('.consent-box input[type="checkbox"]');
    if (!(consent instanceof HTMLInputElement)) throw new Error('Missing endpoint consent');
    expect(settingsToPayload(app.draft).ai.cloud_image_consent).toBe(false);
    expect(app.validationError(2)).toContain('Confirm image sharing');

    consent.checked = true;
    consent.dispatchEvent(new Event('change', { bubbles: true }));
    expect(app.draft.ai.cloudImageConsent).toBe(true);
    expect(settingsToPayload(app.draft).ai.cloud_image_consent).toBe(true);
    expect(app.validationError(2)).toBe('');
  });

  it('does not silently bind a stored API key to a different compatible endpoint', () => {
    const settings = cloneDefaultSettings();
    settings.ai.provider = 'local';
    settings.ai.baseUrl = 'https://first.example.test/v1';
    settings.ai.model = 'vision-model';
    settings.ai.apiKeyConfigured = true;
    settings.ai.cloudImageConsent = true;
    const app = harness(settings);

    renderSettings(app.renderVisionSection(true));
    const endpoint = document.querySelector('input[type="url"]');
    if (!(endpoint instanceof HTMLInputElement)) throw new Error('Missing compatible endpoint input');
    endpoint.value = 'https://second.example.test/v1';
    endpoint.dispatchEvent(new Event('input', { bubbles: true }));

    expect(app.validationError(2)).toContain('Re-enter the API key');
    expect(settingsToPayload(app.draft).secrets.ai_api_key).toBeUndefined();
  });

  it('lets an inactive camera secret be explicitly removed', () => {
    const settings = cloneDefaultSettings();
    settings.camera.enabled = false;
    settings.camera.streamUrlConfigured = true;
    const app = harness(settings);

    renderSettings(app.renderCameraSection(true));
    buttonNamed('Remove stored secret').click();

    expect(app.pendingSecretClears).toContain('camera_stream_url');
    expect(app.draft.camera.streamUrl).toBeUndefined();
  });

  it('shows the Boifun Baby 6T ONVIF setup inside camera settings', () => {
    const settings = cloneDefaultSettings();
    settings.camera.enabled = true;
    const app = harness(settings);

    renderSettings(app.renderCameraSection(true));

    expect(document.body.textContent).toContain('Boifun Baby 6T setup');
    expect(document.body.textContent).toContain('ONVIF settings');
    expect(document.body.textContent).toContain('port 8000');
    expect(document.body.textContent).toContain('account admin');
    expect(document.body.textContent).toContain('Reserve the camera’s IP');
    expect(document.body.textContent).not.toContain('191290');
  });

  it('renders and validates the standalone Home Assistant connection', () => {
    const settings = cloneDefaultSettings();
    settings.homeAssistant.mode = 'standalone';
    const app = harness(settings);

    renderSettings(app.renderHomeAssistantSection(true));
    expect(document.querySelector('input[type="url"]')).not.toBeNull();
    expect(document.querySelector('input[type="password"]')).not.toBeNull();
    expect(buttonNamed('Test connection')).toBeTruthy();
    expect(app.homeAssistantValidationError()).toContain('Home Assistant URL');

    app.draft.homeAssistant.baseUrl = 'https://ha.example.test';
    app.draft.homeAssistant.accessToken = 'new-token';
    expect(app.homeAssistantValidationError()).toBe('');
  });

  it('sends a test notification through the notifications contract', async () => {
    const settings = cloneDefaultSettings();
    settings.notifications.recipients = [{
      personEntityId: 'person.parent',
      name: 'Parent',
      notifyService: 'notify.mobile_app_parent',
      targets: [],
      enabled: true,
      language: 'en',
      events: ['cry_started'],
    }];
    const app = harness(settings);
    const test = vi.spyOn(api, 'testSettings').mockResolvedValue({ ok: true, message: 'Notification sent' });

    renderSettings(app.renderNotificationsSection(true));
    buttonNamed('Send test notification').click();

    await vi.waitFor(() => expect(test).toHaveBeenCalledWith('notifications', app.draft));
  });

  it('selects Home Assistant people and gives each caregiver independent alert toggles', () => {
    const app = harness();
    app.entities = {
      camera: [], binary_sensor: [], light: [],
      notify: [{ entityId: 'notify.mobile_app_victorianos_iphone', name: "Victoriano's iPhone", available: true, attributes: {} }],
      person: [
        { entityId: 'person.victoriano', name: 'Victoriano', available: true, attributes: {} },
        { entityId: 'person.marta', name: 'Marta', available: true, attributes: {} },
      ],
    };

    renderSettings(app.renderNotificationsSection(true));
    const labels = [...document.querySelectorAll('.people-picker label')];
    const victoriano = labels.find((label) => label.textContent?.includes('Victoriano'));
    const checkbox = victoriano?.querySelector('input');
    if (!(checkbox instanceof HTMLInputElement)) throw new Error('Missing Victoriano person option');
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event('change', { bubbles: true }));

    expect(app.draft.notifications.recipients).toHaveLength(1);
    expect(app.draft.notifications.recipients[0].personEntityId).toBe('person.victoriano');
    expect(app.draft.notifications.recipients[0].notifyService).toBe('notify.mobile_app_victorianos_iphone');
    expect(app.draft.notifications.recipients[0].events).toEqual(['cry_started']);

    document.body.replaceChildren();
    renderSettings(app.renderNotificationsSection(true));
    const predicted = [...document.querySelectorAll('.subscription-row')]
      .find((row) => row.textContent?.includes('Sleep is approaching'));
    const predictedToggle = predicted?.querySelector('input');
    if (!(predictedToggle instanceof HTMLInputElement)) throw new Error('Missing predicted sleep toggle');
    predictedToggle.checked = true;
    predictedToggle.dispatchEvent(new Event('change', { bubbles: true }));
    expect(app.draft.notifications.recipients[0].events).toContain('sleep_predicted_soon');
  });

  it('shows the portable CSV and image export as read-only while transfer is pending', () => {
    const app = harness();
    app.historyTransfer = {
      status: 'pending',
      writable: false,
      datasetId: 'dataset-1',
      generation: 1,
      lastImport: null,
      outgoing: {
        archiveId: 'archive-1',
        filename: 'baby-monitor-history.zip',
        createdAt: '2026-07-13T00:00:00Z',
        manifestSha256: 'hash',
        bytes: 1024,
        counts: { frames: 10, storedImages: 9, sleepEvents: 2, cryEvents: 1 },
        downloadUrl: 'api/v1/history-transfer/exports/archive-1',
      },
    };

    renderSettings(app.renderHistoryTransferSection());

    expect(document.body.textContent).toContain('read-only');
    expect(document.body.textContent).toContain('10 records');
    expect(buttonNamed('Download again')).toBeTruthy();
    expect(buttonNamed('Cancel transfer')).toBeTruthy();
    expect(document.body.textContent).toContain('CSV');
    expect(document.body.textContent).toContain('Verified import receipt');
    expect(document.body.textContent).toContain('delete this source history');
  });

  it('explains that a retired source becomes active again by importing a newer ZIP', () => {
    const app = harness();
    app.historyTransfer = {
      status: 'retired',
      writable: false,
      datasetId: 'dataset-1',
      generation: 1,
      lastImport: null,
      outgoing: null,
    };

    renderSettings(app.renderHistoryTransferSection());

    expect(document.body.textContent).toContain('safely retired');
    expect(document.body.textContent).toContain('Import a newer ZIP');
    expect(document.body.textContent).not.toContain('Prepare and download ZIP');
  });
});
