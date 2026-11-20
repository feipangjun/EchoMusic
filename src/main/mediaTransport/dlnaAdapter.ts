/**
 * dlnaAdapter.ts —— 宿主 DLNA 远程媒体后端（design §5"宿主统一输出会话 + 内部协议适配器"）。
 *
 * 契约（对齐 outputs/sessionManager 的 OutputBackendLike，测试注入 mock，不依赖 .node）：
 * - 把宿主输出命令翻译成 AVTransport / RenderingControl SOAP action（经 echo-upnp
 *   原生 loadDevice / action / clearDeviceCache）；
 * - 以 didlLite.buildDidlLite 生成 CurrentURIMetaData，复用输出会话门（sessionLogic 的
 *   EpochGate + advanceGate + acceptsEvent）过滤旧纪元事件；
 * - LastChange（genaReceiver.parseLastChange）驱动的只是"上报分类"，命令面完全走 SOAP action；
 * - 命令失败走 sessionLogic.degradeCapability 降级，不改变会话门（routeEpoch/trackGeneration）。
 *
 * 本文件不触碰插件 API；renderer 的 player:* 契约不变。
 */
import { EventEmitter } from 'events';
import type { OutputBackendLike } from '../outputs/sessionManager';
import {
  type OutputCapability,
  type OutputSessionSnapshot,
  type OutputTrackMeta,
} from '../../shared/playbackOutput';
import {
  acceptsEvent,
  advanceGate,
  classifyRemoteStop,
  degradeCapability,
  REMOTE_END_TOLERANCE_SEC,
  type CommandResult,
  type EpochGate,
} from '../outputs/sessionLogic';
import { buildDidlLite, formatUpnpDuration } from './didlLite';
import { parseLastChange } from './genaReceiver';

/** echo-upnp 原生桥接面（测试注入 mock，不依赖 .node）。 */
export interface UpnpNativeLike {
  loadDevice(descriptionUrl: string): Promise<UpnpDeviceSnapshot>;
  action(
    descriptionUrl: string,
    serviceId: string,
    actionName: string,
    args: Record<string, string>,
  ): Promise<Record<string, string>>;
  clearDeviceCache(): Promise<void>;
}

export interface UpnpServiceEndpoint {
  serviceId: string;
  serviceType: string;
  controlUrl: string;
  eventSubUrl: string;
}

export interface UpnpDeviceSnapshot {
  descriptionUrl: string;
  deviceType: string;
  friendlyName: string;
  manufacturer?: string;
  manufacturerUrl?: string;
  modelName?: string;
  modelDescription?: string;
  modelNumber?: string;
  modelUrl?: string;
  serialNumber?: string;
  udn?: string;
  upc?: string;
  presentationUrl?: string;
  services: UpnpServiceEndpoint[];
}

export interface DlnaAdapterDeps {
  descriptionUrl: string;
  targetId: string;
  displayName: string;
  routeEpoch: number;
  trackGeneration: number;
  native: UpnpNativeLike;
  bus: EventEmitter;
  now?: () => number;
  /** 设备描述里的 serviceId。缺省保持旧的服务类型 URN，供既有单测使用。 */
  avTransportServiceId?: string;
  renderingControlServiceId?: string;
}

export interface DlnaCommand {
  instanceId?: string;
  speed?: string;
  unit?: string;
  target?: string;
  channel?: string;
  desiredVolume?: number;
  currentUri?: string;
  currentUriMetaData?: string;
}

export const AV_TRANSPORT_SERVICE = 'urn:schemas-upnp-org:service:AVTransport:1';
export const RENDERING_CONTROL_SERVICE = 'urn:schemas-upnp-org:service:RenderingControl:1';

const STOPPED_STATES = new Set(['STOPPED', 'NO_MEDIA_PRESENT', 'IDLE']);
export const DLNA_POLL_INTERVAL_MS = 2_000;
const DLNA_TAIL_POLL_INTERVAL_MS = 300;
const TAIL_POLL_WINDOW_SEC = 5;
const TERMINAL_CONFIRM_MS = 3_000;
const RECENT_SAMPLE_MS = 5_000;
const RESTART_WINDOW_SEC = 5;
const POSITION_JITTER_SEC = 0.25;

type TransportObservation = {
  transportState?: string;
  transportStatus?: string;
  positionSec?: number | null;
  durationSec?: number | null;
  trackUri?: string | null;
};

type PlaybackSample = {
  positionSec: number;
  durationSec: number;
  observedAt: number;
  ownUri: boolean;
};

export interface DlnaPlaybackEnded extends EpochGate {
  positionSec: number;
  durationSec: number | null;
  reason: 'reported-stop' | 'position-stuck' | 'position-wrapped';
}

export { formatUpnpDuration } from './didlLite';

function parseUpnpDuration(value: string | undefined): number | null {
  const match = /^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$/.exec(value ?? '');
  if (!match) return null;
  const [hours, minutes, seconds] = match.slice(1).map(Number);
  return minutes < 60 && seconds < 60 ? hours * 3600 + minutes * 60 + seconds : null;
}

export class DlnaBackend implements OutputBackendLike {
  readonly protocol = 'dlna' as const;
  readonly semantics = 'remote-media' as const;
  private readonly deps: DlnaAdapterDeps;
  private readonly bus: EventEmitter;
  private readonly now: () => number;
  private gate: EpochGate;
  private currentCapabilities: OutputCapability = {
    pause: true,
    seek: 'by-time',
    volume: true,
    mute: false,
    rate: false,
    position: 'approximate',
    nextUri: false,
    gapless: false,
    dsp: false,
    spectrum: false,
  };
  private currentUri = '';
  private previousUri: { value: string; expiresAt: number } | null = null;
  private durationSec: number | null = null;
  private volume: number | null = null;
  private trackMeta: OutputTrackMeta = {};
  private lastKnownState = 'STOPPED';
  private lastKnownPositionSec = 0;
  private positionUpdatedAt = 0;
  private transportRevision = 0;
  private activeTransportCommand: number | null = null;
  private endDetectionEnabled = false;
  private seekTarget: {
    positionSec: number;
    previousPositionSec: number;
    observedAt: number;
  } | null = null;
  private lastPlayingSample: PlaybackSample | null = null;
  private lastAdvancingSample: PlaybackSample | null = null;
  private terminalSample: PlaybackSample | null = null;
  private wrapCandidate: Omit<PlaybackSample, 'ownUri'> | null = null;
  private progressObserved = false;
  private endEmitted = false;
  private awaitingReplayProgress = false;
  private takenOver = false;
  private endStopFlight: Promise<unknown> | null = null;
  private readonly avTransportId: string;
  private readonly renderingId: string;
  private sourceMime = 'audio/*';

  constructor(deps: DlnaAdapterDeps) {
    this.deps = deps;
    this.bus = deps.bus;
    this.now = deps.now ?? (() => Date.now());
    this.gate = { routeEpoch: deps.routeEpoch, trackGeneration: deps.trackGeneration };
    this.avTransportId = deps.avTransportServiceId || AV_TRANSPORT_SERVICE;
    this.renderingId = deps.renderingControlServiceId || RENDERING_CONTROL_SERVICE;
  }

  /** 宿主在 load 前声明探测到的 MIME。不改变音源内容。 */
  setSourceHint(hint: { mime?: string | null; relayed?: boolean }): void {
    if (hint.mime?.trim()) this.sourceMime = hint.mime.trim();
    if (hint.relayed != null)
      this.currentCapabilities = { ...this.currentCapabilities, relayed: hint.relayed };
  }

  private didlFor(url: string, title?: string): string {
    return buildDidlLite({
      title: this.trackMeta.title || title || url,
      artist: this.trackMeta.artist,
      album: this.trackMeta.album,
      coverUrl: this.trackMeta.artwork,
      url,
      mime: this.sourceMime,
      durationSec: this.trackMeta.durationMs != null ? this.trackMeta.durationMs / 1000 : undefined,
    });
  }

  beginSourceChange(): number {
    this.transportRevision++;
    this.endDetectionEnabled = false;
    this.seekTarget = null;
    this.clearEndEvidence();
    return this.gate.routeEpoch;
  }

  get observationRevision(): number {
    return this.transportRevision;
  }

  get transportCommandPending(): boolean {
    return this.activeTransportCommand !== null;
  }

  get pollIntervalMs(): number {
    if (
      !this.endDetectionEnabled ||
      this.transportCommandPending ||
      this.lastKnownState !== 'PLAYING'
    ) {
      return DLNA_POLL_INTERVAL_MS;
    }
    const now = this.now();
    const candidate = this.wrapCandidate;
    const advancing = this.lastAdvancingSample;
    if (
      (candidate && now - candidate.observedAt <= RECENT_SAMPLE_MS) ||
      (advancing &&
        advancing.durationSec - advancing.positionSec <= TAIL_POLL_WINDOW_SEC &&
        now - advancing.observedAt <= RECENT_SAMPLE_MS)
    ) {
      return DLNA_TAIL_POLL_INTERVAL_MS;
    }
    return DLNA_POLL_INTERVAL_MS;
  }

  async load(url: string): Promise<{ seq: number; duration: number } | null> {
    return this.loadUri(url);
  }

  async loadMkv(url: string, trackId: number): Promise<{ seq: number; duration: number } | null> {
    return this.loadUri(url, `track-${trackId}`);
  }

  private loadUri(url: string, title?: string) {
    return this.runTransportCommand(async (isCurrent) => {
      if (!isCurrent()) return null;
      await this.soap(this.avTransportId, 'SetAVTransportURI', {
        InstanceID: '0',
        CurrentURI: url,
        CurrentURIMetaData: this.didlFor(url, title),
      });
      if (!isCurrent()) return null;
      await this.setNormalPlayMode();
      if (!isCurrent()) return null;
      this.gate = advanceGate(this.gate, true);
      this.recordLoadedSource(url);
      return { seq: this.gate.trackGeneration, duration: this.durationSec ?? 0 };
    });
  }

  private async setNormalPlayMode(): Promise<void> {
    try {
      await this.soap(this.avTransportId, 'SetPlayMode', {
        InstanceID: '0',
        NewPlayMode: 'NORMAL',
      });
    } catch {
      // SetPlayMode 是可选能力；失败不应阻断投放。
    }
  }

  private recordLoadedSource(url: string): void {
    const now = this.now();
    if (this.currentUri && url !== this.currentUri) {
      this.previousUri = { value: this.currentUri, expiresAt: now + RECENT_SAMPLE_MS };
    }
    this.awaitingReplayProgress = url === this.currentUri;
    this.currentUri = url;
    this.durationSec = this.trackMeta.durationMs != null ? this.trackMeta.durationMs / 1000 : null;
    this.lastKnownPositionSec = 0;
    this.positionUpdatedAt = now;
    this.lastKnownState = 'STOPPED';
    this.endDetectionEnabled = false;
    this.seekTarget = null;
    this.endEmitted = false;
    this.takenOver = false;
    this.clearEndEvidence();
  }

  private clearEndEvidence(): void {
    this.lastPlayingSample = null;
    this.lastAdvancingSample = null;
    this.terminalSample = null;
    this.wrapCandidate = null;
    this.progressObserved = false;
  }

  private async runTransportCommand<T>(run: (isCurrent: () => boolean) => Promise<T>): Promise<T> {
    const command = ++this.transportRevision;
    this.activeTransportCommand = command;
    const isCurrent = () => this.transportRevision === command;
    try {
      // A Stop sent at an inferred EOF must finish before setting the next URI.
      if (this.endStopFlight) await this.endStopFlight;
      return await run(isCurrent);
    } finally {
      if (this.activeTransportCommand === command) {
        this.activeTransportCommand = null;
        this.transportRevision++;
      }
    }
  }

  async setTrackMeta(meta: OutputTrackMeta): Promise<void> {
    this.trackMeta = { ...meta };
  }

  async switchSource(
    url: string,
    trackId?: number | null,
  ): Promise<[number, number, number] | null> {
    const position = this.lastKnownPositionSec;
    const wasPlaying = this.lastKnownState === 'PLAYING';
    const loaded = trackId == null ? await this.load(url) : await this.loadMkv(url, trackId);
    if (!loaded) return null;
    if (position > 0) await this.seek(position);
    if (wasPlaying) await this.play();
    return [position, loaded.duration, loaded.seq];
  }

  async play(): Promise<void> {
    await this.runTransportCommand(async (isCurrent) => {
      if (!isCurrent()) return;
      await this.soap(this.avTransportId, 'Play', { InstanceID: '0', Speed: '1' });
      if (!isCurrent()) return;
      this.lastKnownState = 'PLAYING';
      this.awaitingReplayProgress ||= this.endEmitted;
      this.endEmitted = false;
      this.clearEndEvidence();
      this.endDetectionEnabled = Boolean(this.currentUri);
    });
  }

  async pause(): Promise<void> {
    await this.runTransportCommand(async (isCurrent) => {
      if (!isCurrent()) return;
      await this.soap(this.avTransportId, 'Pause', { InstanceID: '0' });
      if (!isCurrent()) return;
      this.endDetectionEnabled = false;
      this.clearEndEvidence();
      if (!this.endEmitted) this.lastKnownState = 'PAUSED_PLAYBACK';
    });
  }

  async stop(): Promise<void> {
    await this.runTransportCommand(async (isCurrent) => {
      if (!isCurrent()) return;
      await this.soap(this.avTransportId, 'Stop', { InstanceID: '0' });
      if (!isCurrent()) return;
      this.endDetectionEnabled = false;
      this.seekTarget = null;
      this.endEmitted = false;
      this.clearEndEvidence();
      this.lastKnownState = 'STOPPED';
      this.lastKnownPositionSec = 0;
    });
  }

  async seek(timeSec: number): Promise<void> {
    await this.runTransportCommand(async (isCurrent) => {
      if (!isCurrent()) return;
      await this.soap(this.avTransportId, 'Seek', {
        InstanceID: '0',
        Unit: 'REL_TIME',
        Target: formatUpnpDuration(timeSec) ?? '00:00:00',
      });
      if (!isCurrent()) return;
      const observedAt = this.now();
      this.seekTarget = {
        positionSec: timeSec,
        previousPositionSec: this.lastKnownPositionSec,
        observedAt,
      };
      this.recordPosition(timeSec, observedAt);
      this.clearEndEvidence();
    });
  }

  async setVolume(volume: number): Promise<void> {
    const clamped = Math.max(0, Math.min(100, Math.round(volume)));
    await this.soap(this.renderingId, 'SetVolume', {
      InstanceID: '0',
      Channel: 'Master',
      DesiredVolume: String(clamped),
    });
    this.volume = clamped;
  }

  async getState(): Promise<OutputSessionSnapshot> {
    const state: OutputSessionSnapshot['state'] =
      this.lastKnownState === 'PLAYING'
        ? 'playing'
        : this.lastKnownState === 'PAUSED_PLAYBACK'
          ? 'paused'
          : STOPPED_STATES.has(this.lastKnownState)
            ? 'stopped'
            : 'unknown';
    return {
      sessionId: `${this.protocol}:${this.gate.routeEpoch}`,
      routeEpoch: this.gate.routeEpoch,
      trackGeneration: this.gate.trackGeneration,
      protocol: this.protocol,
      targetId: this.deps.targetId,
      displayName: this.deps.displayName,
      state,
      positionSec: this.lastKnownPositionSec,
      observedAt: this.positionUpdatedAt,
      clockAccuracy: this.positionUpdatedAt > 0 ? 'approximate' : 'unknown',
      durationSec: this.durationSec,
      actualFormat: null,
      volume: this.volume,
      capabilities: { ...this.currentCapabilities },
    };
  }

  async isBusy(): Promise<boolean> {
    return this.lastKnownState !== 'STOPPED';
  }

  private async soap(
    serviceId: string,
    actionName: string,
    args: Record<string, string>,
  ): Promise<Record<string, string>> {
    try {
      return await this.deps.native.action(this.deps.descriptionUrl, serviceId, actionName, args);
    } catch (e) {
      const rejectedAction: CommandResult['rejectedAction'] =
        actionName === 'Pause'
          ? 'pause'
          : actionName === 'Seek'
            ? 'seek'
            : actionName === 'SetVolume'
              ? 'volume'
              : undefined;
      this.currentCapabilities = degradeCapability(this.currentCapabilities, {
        ok: false,
        rejectedAction,
      });
      throw e;
    }
  }

  /** 轮询和 GENA 共用观测入口。 */
  observeTransport(update: TransportObservation, eventGate: EpochGate = this.gate): void {
    if (this.transportCommandPending || !acceptsEvent(this.gate, eventGate)) return;
    const state = update.transportState?.trim().toUpperCase();
    const uri = update.trackUri?.trim() || null;
    const now = this.now();
    const { durationSec, positionSec } = update;
    const duration =
      durationSec != null && Number.isFinite(durationSec) && durationSec > 0 ? durationSec : null;
    const position =
      positionSec != null && Number.isFinite(positionSec) && positionSec >= 0 ? positionSec : null;

    if (update.transportStatus?.toUpperCase().includes('ERROR')) {
      this.clearEndEvidence();
      if (!this.endEmitted) this.lastKnownState = 'UNKNOWN';
      return;
    }

    if (this.currentUri && uri && uri !== this.currentUri) {
      // After SetAVTransportURI, notifications and position replies for the old
      // track can still arrive. Give only that URI a short transition window.
      if (uri === this.previousUri?.value && now <= this.previousUri.expiresAt) return;
      this.endDetectionEnabled = false;
      this.clearEndEvidence();
      this.lastKnownState = 'STOPPED';
      if (!this.takenOver) {
        this.takenOver = true;
        this.bus.emit('taken-over', { targetId: this.deps.targetId });
      }
      return;
    }

    if (this.endEmitted || this.takenOver) return;
    if (duration != null) this.durationSec = duration;
    const sample =
      position != null && duration != null
        ? {
            positionSec: position,
            durationSec: duration,
            observedAt: now,
            ownUri: uri === this.currentUri,
          }
        : null;

    if (this.seekTarget) {
      const elapsed = Math.max(0, now - this.seekTarget.observedAt);
      const reachedSeekTarget =
        state === 'PLAYING' &&
        position != null &&
        position >= this.seekTarget.positionSec - REMOTE_END_TOLERANCE_SEC &&
        (this.seekTarget.previousPositionSec < this.seekTarget.positionSec ||
          position <= this.seekTarget.positionSec + elapsed / 1_000 + REMOTE_END_TOLERANCE_SEC);
      if (!reachedSeekTarget && elapsed <= RECENT_SAMPLE_MS) return;
      this.seekTarget = null;
      if (state) this.lastKnownState = state;
      if (position != null) this.recordPosition(position, now);
      if (reachedSeekTarget) {
        this.awaitingReplayProgress = false;
        this.lastPlayingSample = sample;
      }
      return;
    }

    if (this.endDetectionEnabled && this.awaitingReplayProgress) {
      // A repeated URI may still report the previous play's final STOPPED or position.
      if (state !== 'PLAYING' || position == null) return;
      const knownDuration = this.durationSec;
      if (
        position !== 0 &&
        (knownDuration == null
          ? position > RESTART_WINDOW_SEC
          : position >= knownDuration - REMOTE_END_TOLERANCE_SEC)
      )
        return;
      this.awaitingReplayProgress = false;
    }

    if (state && STOPPED_STATES.has(state)) {
      const tail = this.lastPlayingSample;
      const recentTail = this.isNearEnd(tail) && now - tail.observedAt <= RECENT_SAMPLE_MS;
      const stopped = classifyRemoteStop({
        currentUri: this.currentUri,
        eventUri: uri,
        durationSec: duration ?? tail?.durationSec ?? null,
        positionSec: position,
        reportedState: state,
        pendingCommand: 'none',
        sessionEpoch: eventGate.routeEpoch,
        activeEpoch: this.gate.routeEpoch,
        trackGeneration: eventGate.trackGeneration,
        activeGeneration: this.gate.trackGeneration,
        nearEnd: (position == null || position === 0) && recentTail,
      });
      if (this.endDetectionEnabled && stopped.proceed) {
        this.finishPlayback('reported-stop', duration ?? tail?.durationSec ?? this.durationSec);
        return;
      }
      this.endDetectionEnabled = false;
      this.clearEndEvidence();
    }

    if (state) this.lastKnownState = state;
    if (state === 'PAUSED_PLAYBACK') this.clearEndEvidence();
    if (state === 'PLAYING' && this.endDetectionEnabled && position != null) {
      const sameUri = uri === this.currentUri;
      if (this.observePlaying(position, sample, sameUri, durationSec === 0, now)) return;
    }
    if (position != null) this.recordPosition(position, now);
  }

  /** 返回 true 表示保留曲尾时间，不把设备试图回卷的进度显示给用户。 */
  private observePlaying(
    position: number,
    sample: PlaybackSample | null,
    sameUri: boolean,
    zeroDuration: boolean,
    now: number,
  ): boolean {
    const previous = this.lastPlayingSample;
    this.lastPlayingSample = sample;

    const candidate = this.wrapCandidate;
    if (candidate) {
      const stillAtStart =
        position <= RESTART_WINDOW_SEC &&
        now - candidate.observedAt <= RECENT_SAMPLE_MS &&
        (sample == null ||
          Math.abs(sample.durationSec - candidate.durationSec) <= REMOTE_END_TOLERANCE_SEC);
      if (stillAtStart) {
        if (sameUri && sample && position > candidate.positionSec + POSITION_JITTER_SEC) {
          this.finishPlayback('position-wrapped', sample.durationSec);
        }
        return true;
      }
      this.wrapCandidate = null;
    }

    if (
      this.isNearEnd(previous) &&
      now - previous.observedAt <= RECENT_SAMPLE_MS &&
      position <= RESTART_WINDOW_SEC &&
      previous.positionSec - position > REMOTE_END_TOLERANCE_SEC
    ) {
      const advancing = this.lastAdvancingSample;
      // Repeated integer-second samples are common. Anchor the deadline to the last
      // real advance, and only use it to confirm a reset reported by the renderer.
      const expectedEndReached =
        previous.ownUri &&
        advancing &&
        previous.positionSec === advancing.positionSec &&
        previous.durationSec === advancing.durationSec &&
        now - advancing.observedAt >= (previous.durationSec - previous.positionSec) * 1_000 &&
        now - advancing.observedAt <= RECENT_SAMPLE_MS;
      if (sameUri && position === 0 && zeroDuration && expectedEndReached) {
        this.finishPlayback('position-wrapped', previous.durationSec);
        return true;
      }
      this.wrapCandidate = {
        positionSec: position,
        durationSec: previous.durationSec,
        observedAt: now,
      };
      this.lastAdvancingSample = null;
      this.terminalSample = null;
      return true;
    }

    this.recordPlaybackAdvance(previous, sample);
    if (previous && sample && position > previous.positionSec + POSITION_JITTER_SEC) {
      this.progressObserved = true;
    }
    if (
      sample == null ||
      Math.abs(sample.positionSec - sample.durationSec) > POSITION_JITTER_SEC ||
      !this.progressObserved
    ) {
      this.terminalSample = null;
      return false;
    }

    const terminal = this.terminalSample;
    if (
      terminal == null ||
      previous == null ||
      now - previous.observedAt > RECENT_SAMPLE_MS ||
      Math.abs(position - terminal.positionSec) > POSITION_JITTER_SEC ||
      Math.abs(sample.durationSec - terminal.durationSec) > POSITION_JITTER_SEC
    ) {
      this.terminalSample = sample;
    } else if (now - terminal.observedAt >= TERMINAL_CONFIRM_MS) {
      this.finishPlayback('position-stuck', sample.durationSec);
      return true;
    }
    return false;
  }

  private recordPlaybackAdvance(
    previous: PlaybackSample | null,
    sample: PlaybackSample | null,
  ): void {
    if (!previous?.ownUri || !sample?.ownUri) {
      this.lastAdvancingSample = null;
      return;
    }
    const elapsedMs = sample.observedAt - previous.observedAt;
    const advance = sample.positionSec - previous.positionSec;
    if (
      elapsedMs < 0 ||
      elapsedMs > RECENT_SAMPLE_MS ||
      Math.abs(sample.durationSec - previous.durationSec) > POSITION_JITTER_SEC ||
      advance < -POSITION_JITTER_SEC ||
      advance > elapsedMs / 1_000 + 1
    ) {
      this.lastAdvancingSample = null;
    } else if (advance > POSITION_JITTER_SEC) {
      this.lastAdvancingSample = sample;
    }
  }

  private isNearEnd(sample: PlaybackSample | null): sample is PlaybackSample {
    return (
      sample != null &&
      sample.positionSec > 0 &&
      sample.positionSec >= sample.durationSec - REMOTE_END_TOLERANCE_SEC
    );
  }

  private recordPosition(position: number, observedAt: number): void {
    this.lastKnownPositionSec = position;
    this.positionUpdatedAt = observedAt;
  }

  private finishPlayback(reason: DlnaPlaybackEnded['reason'], duration: number | null): void {
    this.endEmitted = true;
    this.endDetectionEnabled = false;
    this.lastKnownState = 'STOPPED';
    this.clearEndEvidence();
    this.durationSec = duration;
    if (duration != null) this.recordPosition(duration, this.now());
    if (reason !== 'reported-stop') {
      this.endStopFlight = this.soap(this.avTransportId, 'Stop', { InstanceID: '0' })
        .catch(() => undefined)
        .finally(() => {
          this.endStopFlight = null;
        });
    }
    this.bus.emit('playback-ended', {
      ...this.gate,
      positionSec: this.lastKnownPositionSec,
      durationSec: this.durationSec,
      reason,
    } satisfies DlnaPlaybackEnded);
  }

  /** LastChange 上报：只做事件分类与纪元过滤，不在此重放命令。 */
  handleLastChange(payload: string, eventGate: EpochGate = this.gate): void {
    const parsed = parseLastChange(payload);
    if (!parsed) return;
    this.observeTransport(
      {
        transportState: parsed.values.TransportState,
        transportStatus: parsed.values.TransportStatus,
        positionSec: parseUpnpDuration(parsed.values.RelativeTimePosition),
        durationSec: parseUpnpDuration(parsed.values.CurrentTrackDuration),
        trackUri: parsed.values.CurrentTrackURI ?? parsed.values.AVTransportURI ?? null,
      },
      eventGate,
    );
  }
}
