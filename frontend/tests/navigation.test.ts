import { render, type TemplateResult } from 'lit';
import { describe, expect, it } from 'vitest';

import { BabyMonitorApp } from '../src/baby-monitor-app';
import type { AppPage } from '../src/types';

interface NavigationHarness {
  page: AppPage;
  manualOpen: boolean;
  renderHeader(): TemplateResult;
  renderManualDialog(): TemplateResult;
}

function harness(): NavigationHarness {
  const app = new BabyMonitorApp() as unknown as NavigationHarness;
  app.page = 'dashboard';
  app.manualOpen = false;
  return app;
}

describe('floating navigation', () => {
  it('exposes the five caregiver destinations in the requested order', () => {
    const app = harness();
    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderHeader(), container);

    const buttons = [...container.querySelectorAll('.primary-nav button')];
    expect(buttons.map((button) => button.getAttribute('aria-label'))).toEqual([
      'Now', 'Camera', 'Add', 'History', 'Settings',
    ]);
    expect(buttons[0]?.getAttribute('aria-current')).toBe('page');
  });

  it('opens the manual sleep dialog from the raised centre action', async () => {
    const app = harness();
    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderHeader(), container);

    const add = container.querySelector('.nav-add');
    if (!(add instanceof HTMLButtonElement)) throw new Error('Missing central add action');
    add.click();
    expect(app.manualOpen).toBe(true);

    render(app.renderManualDialog(), container);
    expect(container.querySelector('[role="dialog"]')?.getAttribute('aria-label')).toBe('Add a past sleep');
    expect(container.querySelector('form')).not.toBeNull();
  });
});
