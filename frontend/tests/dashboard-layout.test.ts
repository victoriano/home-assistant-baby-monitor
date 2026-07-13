import { render, type TemplateResult } from 'lit';
import { describe, expect, it, vi } from 'vitest';

import { BabyMonitorApp } from '../src/baby-monitor-app';
import { cloneDefaultSettings, type AppSettings } from '../src/types';

interface DashboardHarness {
  settings: AppSettings;
  renderDashboard(): TemplateResult;
  loadOperationalData(showSpinner?: boolean): Promise<void>;
}

describe('dashboard hierarchy', () => {
  it('puts the baby identity and manual refresh inside the rhythm card', () => {
    const app = new BabyMonitorApp() as unknown as DashboardHarness;
    app.settings = cloneDefaultSettings();
    app.settings.baby.name = 'Esteban';
    app.loadOperationalData = vi.fn().mockResolvedValue(undefined);

    const container = document.createElement('div');
    document.body.append(container);
    render(app.renderDashboard(), container);

    expect(container.querySelector('.dashboard-heading')).toBeNull();
    expect(container.querySelector('.rhythm-context')?.textContent).toContain('Esteban');

    const refresh = container.querySelector('.rhythm-refresh');
    if (!(refresh instanceof HTMLButtonElement)) throw new Error('Missing rhythm refresh action');
    expect(refresh.getAttribute('aria-label')).toBe('Refresh all data');
    refresh.click();
    expect(app.loadOperationalData).toHaveBeenCalledWith(true);
  });
});
