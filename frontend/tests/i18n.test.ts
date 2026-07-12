import { describe, expect, it } from 'vitest';

import { formatDuration, translate } from '../src/i18n';

describe('translations', () => {
  it('interpolates Spanish interface copy', () => {
    expect(translate('es', 'selectedCount', { count: 3 })).toBe('3 seleccionadas');
    expect(translate('es', 'cloudConsent', { provider: 'OpenAI' })).toContain('OpenAI');
    expect(translate('es', 'localCompatible')).toBe('Endpoint compatible con OpenAI');
    expect(translate('es', 'historyShowing', { shown: 30, total: 82 })).toBe('Mostrando 30 de 82');
    expect(translate('es', 'homeAssistantTokenHint')).toContain('Solo escritura');
    expect(translate('en', 'requiredApiKeyAgain')).toContain('Re-enter');
  });

  it('formats long sleep durations compactly', () => {
    expect(formatDuration(59)).toBe('59 min');
    expect(formatDuration(60)).toBe('1 h');
    expect(formatDuration(545)).toBe('9 h 5 min');
  });
});
