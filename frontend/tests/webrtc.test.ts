import { render, type TemplateResult } from 'lit';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { BabyMonitorApp } from '../src/baby-monitor-app';
import { cloneDefaultSettings, type AppSettings, type DashboardSummary } from '../src/types';
import { connectWebRtcVideo } from '../src/webrtc';

interface CameraHarness {
  settings: AppSettings;
  summary: DashboardSummary;
  liveView: boolean;
  liveTransport: 'off' | 'connecting' | 'webrtc' | 'mjpeg';
  renderCameraCard(): TemplateResult;
}

class FakeMediaStream {
  readonly tracks: unknown[] = [];

  addTrack(track: unknown): void {
    this.tracks.push(track);
  }
}

class FakePeerConnection extends EventTarget {
  iceGatheringState: RTCIceGatheringState = 'complete';
  connectionState: RTCPeerConnectionState = 'new';
  localDescription: RTCSessionDescriptionInit | null = null;
  ontrack: ((event: RTCTrackEvent) => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;
  readonly transceivers: string[] = [];
  closed = false;
  remoteDescription: RTCSessionDescriptionInit | null = null;

  addTransceiver(kind: string): RTCRtpTransceiver {
    this.transceivers.push(kind);
    return {} as RTCRtpTransceiver;
  }

  async createOffer(): Promise<RTCSessionDescriptionInit> {
    return { type: 'offer', sdp: 'v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n' };
  }

  async setLocalDescription(description: RTCSessionDescriptionInit): Promise<void> {
    this.localDescription = description;
  }

  async setRemoteDescription(description: RTCSessionDescriptionInit): Promise<void> {
    this.remoteDescription = description;
  }

  close(): void {
    this.closed = true;
  }
}

describe('WebRTC live view', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('negotiates receive-only video and audio and attaches the remote answer', async () => {
    const peer = new FakePeerConnection();
    vi.stubGlobal('MediaStream', FakeMediaStream);
    vi.stubGlobal('RTCPeerConnection', function FakePeerConstructor() { return peer; });
    const video = document.createElement('video');
    video.play = vi.fn().mockResolvedValue(undefined);
    const negotiate = vi.fn().mockResolvedValue('v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n');

    const connected = await connectWebRtcVideo(video, negotiate);

    expect(connected).toBe(peer);
    expect(peer.transceivers).toEqual(['video', 'audio']);
    expect(negotiate).toHaveBeenCalledWith(expect.stringContaining('m=video'));
    expect(peer.remoteDescription?.type).toBe('answer');
  });

  it('keeps the latest frame visible until WebRTC is actually playing', () => {
    const app = new BabyMonitorApp() as unknown as CameraHarness;
    app.settings = cloneDefaultSettings();
    app.settings.camera.enabled = true;
    app.summary = {
      state: 'awake',
      stateSince: null,
      currentSleep: null,
      prediction: { nextSleepAt: null, windowStart: null, windowEnd: null, confidence: null, reason: null },
      sleepTodayMinutes: 0,
      lastCryAt: null,
      cryActive: false,
      latestFrame: {
        id: 'frame-1',
        capturedAt: '2026-07-13T18:33:19Z',
        cameraEntityId: 'camera.nursery',
        locationId: 'granada',
        imageUrl: '/api/v1/frames/frame-1/image',
        imageAvailable: true,
        mimeType: 'image/jpeg',
        sizeBytes: 123,
        label: null,
        provider: null,
        model: null,
      },
      recentSleep: [],
      recentCry: [],
      updatedAt: null,
    };
    app.liveView = true;
    app.liveTransport = 'connecting';
    const container = document.createElement('div');
    render(app.renderCameraCard(), container);

    expect(container.querySelector<HTMLImageElement>('.camera-live-poster')?.src).toContain('/api/v1/frames/frame-1/image');
    expect(container.querySelector('.camera-live-video')?.classList.contains('ready')).toBe(false);

    app.liveTransport = 'webrtc';
    render(app.renderCameraCard(), container);
    expect(container.querySelector('.camera-live-video')?.classList.contains('ready')).toBe(true);
  });
});
