export type WebRtcNegotiator = (offer: string) => Promise<string>;

function waitForIceGathering(peer: RTCPeerConnection, timeoutMs = 2_500): Promise<void> {
  if (peer.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    const finish = (): void => {
      window.clearTimeout(timeout);
      peer.removeEventListener('icegatheringstatechange', changed);
      resolve();
    };
    const changed = (): void => {
      if (peer.iceGatheringState === 'complete') finish();
    };
    const timeout = window.setTimeout(finish, timeoutMs);
    peer.addEventListener('icegatheringstatechange', changed);
  });
}

export async function connectWebRtcVideo(
  video: HTMLVideoElement,
  negotiate: WebRtcNegotiator,
): Promise<RTCPeerConnection> {
  if (!globalThis.RTCPeerConnection || !globalThis.MediaStream) {
    throw new Error('WebRTC is not available in this browser');
  }
  const peer = new RTCPeerConnection({ bundlePolicy: 'max-bundle' });
  const stream = new MediaStream();
  video.srcObject = stream;
  peer.ontrack = (event) => {
    const remote = event.streams[0];
    if (remote) {
      video.srcObject = remote;
      return;
    }
    stream.addTrack(event.track);
  };
  try {
    peer.addTransceiver('video', { direction: 'recvonly' });
    peer.addTransceiver('audio', { direction: 'recvonly' });
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    await waitForIceGathering(peer);
    const localSdp = peer.localDescription?.sdp;
    if (!localSdp) throw new Error('WebRTC did not create an SDP offer');
    const answer = await negotiate(localSdp);
    await peer.setRemoteDescription({ type: 'answer', sdp: answer });
    void video.play().catch(() => undefined);
    return peer;
  } catch (error) {
    peer.close();
    video.srcObject = null;
    throw error;
  }
}
