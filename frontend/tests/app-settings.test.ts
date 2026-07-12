import { render, type TemplateResult } from 'lit';
import { describe, expect, it, vi } from 'vitest';

import { api } from '../src/api';
import { BabyMonitorApp } from '../src/baby-monitor-app';
import { cloneDefaultSettings, settingsToPayload, type AppSettings, type SecretName } from '../src/types';

interface SettingsHarness {
  draft: AppSettings;
  settings: AppSettings;
  pendingSecretClears: SecretName[];
  renderCameraSection(compact?: boolean): TemplateResult;
  renderHomeAssistantSection(compact?: boolean): TemplateResult;
  renderNotificationsSection(compact?: boolean): TemplateResult;
  renderVisionSection(compact?: boolean): TemplateResult;
  homeAssistantValidationError(): string;
  validationError(stage: number | 'all'): string;
}

function harness(settings = cloneDefaultSettings()): SettingsHarness {
  const app = new BabyMonitorApp() as unknown as SettingsHarness;
  app.settings = structuredClone(settings);
  app.draft = structuredClone(settings);
  app.pendingSecretClears = [];
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
    settings.notifications.service = 'notify.mobile_app_parent';
    const app = harness(settings);
    const test = vi.spyOn(api, 'testSettings').mockResolvedValue({ ok: true, message: 'Notification sent' });

    renderSettings(app.renderNotificationsSection(true));
    buttonNamed('Send test notification').click();

    await vi.waitFor(() => expect(test).toHaveBeenCalledWith('notifications', app.draft));
  });
});
