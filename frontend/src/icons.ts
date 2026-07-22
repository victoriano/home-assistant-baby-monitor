import { svg, type SVGTemplateResult } from 'lit';

export type IconName =
  | 'baby'
  | 'calendar'
  | 'camera'
  | 'check'
  | 'chevron'
  | 'clock'
  | 'eye'
  | 'heart'
  | 'history'
  | 'home'
  | 'light'
  | 'lock'
  | 'moon'
  | 'play'
  | 'plus'
  | 'refresh'
  | 'settings'
  | 'sparkle'
  | 'stop'
  | 'sun'
  | 'waves';

function path(name: IconName): SVGTemplateResult {
  switch (name) {
    case 'baby':
      return svg`<path d="M9 12h6"/><path d="M10 16c.6.7 1.3 1 2 1s1.4-.3 2-1"/><path d="M12 2a3 3 0 0 0-3 3c0 .4.1.8.2 1.1A7 7 0 1 0 18.9 9"/><circle cx="9" cy="10" r=".6" fill="currentColor" stroke="none"/><circle cx="15" cy="10" r=".6" fill="currentColor" stroke="none"/>`;
    case 'calendar':
      return svg`<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M8 3v4M16 3v4M3 10h18"/><path d="M8 14h.01M12 14h.01M16 14h.01M8 18h.01M12 18h.01"/>`;
    case 'camera':
      return svg`<path d="M14.5 5 13 3h-2L9.5 5H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-4.5Z"/><circle cx="12" cy="12" r="4"/>`;
    case 'check':
      return svg`<path d="m5 12 4 4L19 6"/>`;
    case 'chevron':
      return svg`<path d="m9 18 6-6-6-6"/>`;
    case 'clock':
      return svg`<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>`;
    case 'eye':
      return svg`<path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/>`;
    case 'heart':
      return svg`<path d="M20.8 5.7a5.4 5.4 0 0 0-7.6 0L12 6.9l-1.2-1.2a5.4 5.4 0 0 0-7.6 7.6L12 22l8.8-8.7a5.4 5.4 0 0 0 0-7.6Z"/><path d="M7 12h2l1-2 2 5 1.5-3H17"/>`;
    case 'history':
      return svg`<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>`;
    case 'home':
      return svg`<path d="m3 11 9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/>`;
    case 'light':
      return svg`<path d="M9 18h6"/><path d="M10 22h4"/><path d="M8.2 15.2A7 7 0 1 1 15.8 15c-.7.5-.8 1.2-.8 2H9c0-.8-.1-1.3-.8-1.8Z"/>`;
    case 'lock':
      return svg`<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/><path d="M12 14v3"/>`;
    case 'moon':
      return svg`<path d="M20 15.2A8.7 8.7 0 0 1 8.8 4a9 9 0 1 0 11.2 11.2Z"/>`;
    case 'play':
      return svg`<path d="m8 5 11 7-11 7V5Z" fill="currentColor" stroke="none"/>`;
    case 'plus':
      return svg`<path d="M12 5v14M5 12h14"/>`;
    case 'refresh':
      return svg`<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/>`;
    case 'settings':
      return svg`<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.6v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/>`;
    case 'sparkle':
      return svg`<path d="m12 3 1.3 3.7L17 8l-3.7 1.3L12 13l-1.3-3.7L7 8l3.7-1.3L12 3Z"/><path d="m18.5 13 .8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2Z"/><path d="m5 13 .7 1.8 1.8.7-1.8.7L5 18l-.7-1.8-1.8-.7 1.8-.7L5 13Z"/>`;
    case 'stop':
      return svg`<rect x="6" y="6" width="12" height="12" rx="2"/>`;
    case 'sun':
      return svg`<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>`;
    case 'waves':
      return svg`<path d="M3 8c2-2 4-2 6 0s4 2 6 0 4-2 6 0"/><path d="M3 12c2-2 4-2 6 0s4 2 6 0 4-2 6 0"/><path d="M3 16c2-2 4-2 6 0s4 2 6 0 4-2 6 0"/>`;
  }
}

export function icon(name: IconName, size = 20): SVGTemplateResult {
  return svg`<svg
    class="icon icon-${name}"
    width=${size}
    height=${size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.8"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >${path(name)}</svg>`;
}
