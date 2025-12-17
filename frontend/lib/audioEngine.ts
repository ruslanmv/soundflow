// frontend/lib/audioEngine.ts
// Production-ready AudioEngine for looping tracks + radio streams.
// - Accepts AudioSource objects OR plain URL strings (fixes your fallback issue)
// - Avoids AbortError via playToken serialization
// - Uses loadedmetadata/canplay (stream-friendly); avoids canplaythrough
// - Cleans up sources to stop network activity (important for radio)

export type AudioSource = {
  url: string;
  /** Strongly recommended for radio + Safari: e.g. 'audio/mpeg', 'audio/ogg', 'audio/aac' */
  type?: string;
};

export type AudioSourceInput = string | AudioSource;

type Mode = "loop" | "radio";

export class AudioEngine {
  private music: HTMLAudioElement | null = null;
  private ambience: HTMLAudioElement | null = null;

  private playToken = 0;
  private mode: Mode = "loop";

  constructor() {
    if (typeof window === "undefined") return;

    this.music = document.createElement("audio");
    this.ambience = document.createElement("audio");

    // Defaults
    this.music.preload = "auto";
    this.ambience.preload = "auto";

    // iOS Safari: allow inline playback
    (this.music as any).playsInline = true;
    (this.ambience as any).playsInline = true;

    // Only set crossOrigin if you actually need it (visualizers/WebAudio, etc).
    // For some hosts, forcing anonymous can cause failures.
    // this.music.crossOrigin = "anonymous";
    // this.ambience.crossOrigin = "anonymous";

    this.applyMode();
  }

  /** Switch between looping tracks vs live radio stream behavior */
  setMode(mode: Mode) {
    this.mode = mode;
    this.applyMode();
  }

  private applyMode() {
    if (!this.music || !this.ambience) return;

    if (this.mode === "radio") {
      this.music.loop = false; // radio streams should not loop
      this.music.preload = "none"; // avoid heavy preloading for streams
    } else {
      this.music.loop = true;
      this.music.preload = "auto";
    }

    // ambience is typically a file that can loop
    this.ambience.loop = true;
    this.ambience.preload = "auto";
  }

  isReady() {
    return !!this.music && this.music.querySelector("source") !== null;
  }

  setMusicVolume(v01: number) {
    if (!this.music) return;
    this.music.volume = clamp01(v01);
  }

  setAmbienceVolume(v01: number) {
    if (!this.ambience) return;
    this.ambience.volume = clamp01(v01);
  }

  /**
   * Set file sources (looping music + optional ambience).
   * You can pass multiple candidates for music for fallback (mp3/ogg/aac).
   *
   * Accepts:
   *  - string URL: "https://..."
   *  - AudioSource: { url, type }
   *  - arrays of either
   */
  setSources(
    music: AudioSourceInput | AudioSourceInput[],
    ambience?: AudioSourceInput | AudioSourceInput[]
  ) {
    this.setMode("loop");
    this.setInternalSources(this.music, normalizeSources(music));
    if (this.ambience) this.setInternalSources(this.ambience, normalizeSources(ambience));
  }

  /**
   * Radio modality: live stream (Icecast/Shoutcast/HLS-audio/etc).
   * Pass multiple candidates for best compatibility (esp. Safari).
   *
   * Example:
   *   engine.setRadioSources([
   *     { url: streamMp3, type: "audio/mpeg" },
   *     { url: streamAac, type: "audio/aac" },
   *   ]);
   */
  setRadioSources(streams: AudioSourceInput | AudioSourceInput[]) {
    this.setMode("radio");
    this.setInternalSources(this.music, normalizeSources(streams));
    // radio usually has no ambience; you can still set it separately if you want
  }

  private setInternalSources(el: HTMLAudioElement | null, sources: AudioSource[]) {
    if (!el) return;

    // cancel pending play
    this.playToken++;

    // pause and reset
    safePause(el);
    try {
      el.currentTime = 0;
    } catch {
      // Some streams disallow setting currentTime; ignore.
    }

    // Clear existing <source> nodes
    while (el.firstChild) el.removeChild(el.firstChild);

    // Add sources in order (browser will pick the first it can play)
    for (const s of sources) {
      if (!s?.url) continue;
      const src = document.createElement("source");
      src.src = s.url;
      if (s.type) src.type = s.type;
      el.appendChild(src);
    }

    // Force reload selection
    el.load();
  }

  /**
   * Wait until element is playable.
   * Use canplay/loadedmetadata (works for radio); avoid canplaythrough (often never fires for streams).
   */
  private waitPlayable(el: HTMLAudioElement, timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const fail = (reason: string) => reject(new Error(reason));

      // Immediate error?
      if (el.error) return fail(formatMediaError(el));

      // If we already have enough data to start
      // HAVE_METADATA (1) is often enough for streams; HAVE_CURRENT_DATA (2) for files.
      if (el.readyState >= HTMLMediaElement.HAVE_METADATA) return resolve();

      const t = window.setTimeout(() => {
        cleanup();
        fail(`Audio load timed out (${timeoutMs}ms) for ${currentSrc(el) || "(no src)"}`);
      }, timeoutMs);

      const onOk = () => {
        cleanup();
        resolve();
      };

      const onErr = () => {
        cleanup();
        fail(formatMediaError(el));
      };

      const cleanup = () => {
        window.clearTimeout(t);
        el.removeEventListener("loadedmetadata", onOk);
        el.removeEventListener("canplay", onOk);
        el.removeEventListener("error", onErr);
        el.removeEventListener("stalled", onErr);
      };

      el.addEventListener("loadedmetadata", onOk, { once: true });
      el.addEventListener("canplay", onOk, { once: true });
      el.addEventListener("error", onErr, { once: true });
      // If it stalls immediately (bad stream / server), fail fast
      el.addEventListener("stalled", onErr, { once: true });
    });
  }

  /**
   * Serialized play to prevent AbortError.
   * Handles autoplay restrictions and decode failures cleanly.
   */
  async play() {
    if (!this.music) return;

    const token = ++this.playToken;

    // Ensure we actually have at least one <source>
    if (!this.music.querySelector("source")) {
      console.warn("AudioEngine: play() called but no music source is set.");
      return;
    }

    // Radio can legitimately take longer to become playable than files.
    const timeout = this.mode === "radio" ? 20000 : 8000;

    await this.waitPlayable(this.music, timeout);
    if (token !== this.playToken) return;

    try {
      await this.music.play();
    } catch (err: any) {
      // NotAllowedError = user gesture required (autoplay policy)
      // NotSupportedError / AbortError etc are also possible
      const name = err?.name || "PlayError";
      const msg = err?.message || String(err);
      throw new Error(`music.play() failed [${name}]: ${msg}`);
    }

    if (token !== this.playToken) return;

    if (this.ambience && this.ambience.querySelector("source")) {
      await this.waitPlayable(this.ambience, 8000);
      if (token !== this.playToken) return;

      try {
        await this.ambience.play();
      } catch (err: any) {
        // ambience failure shouldn't necessarily kill music
        console.warn(`ambience.play() failed: ${err?.name ?? ""} ${err?.message ?? err}`);
      }
    }
  }

  pause() {
    if (!this.music) return;
    this.playToken++;
    safePause(this.music);
    if (this.ambience) safePause(this.ambience);
  }

  stop() {
    if (!this.music) return;
    this.playToken++;

    // Stop playback
    safePause(this.music);
    if (this.ambience) safePause(this.ambience);

    // Clear sources to actually stop network activity (important for radio)
    clearElementSources(this.music);
    if (this.ambience) clearElementSources(this.ambience);
  }
}

/** Helpers */

function clamp01(n: number) {
  return Math.max(0, Math.min(1, n));
}

function normalizeSources(input: AudioSourceInput | AudioSourceInput[] | undefined): AudioSource[] {
  const arr = Array.isArray(input) ? input : input ? [input] : [];
  return arr
    .filter((x): x is AudioSourceInput => !!x)
    .map((s) => (typeof s === "string" ? { url: s } : s))
    .filter((s) => !!s.url);
}

function safePause(el: HTMLMediaElement) {
  try {
    el.pause();
  } catch {
    // ignore
  }
}

function clearElementSources(el: HTMLAudioElement) {
  // Remove <source> children
  while (el.firstChild) el.removeChild(el.firstChild);

  // Clearing src + load() helps cancel pending requests in many browsers
  el.removeAttribute("src");
  el.load();
}

function currentSrc(el: HTMLMediaElement) {
  // currentSrc is often more accurate than el.src when <source> is used
  return (el as any).currentSrc || el.getAttribute("src") || "";
}

function formatMediaError(el: HTMLMediaElement) {
  const src = currentSrc(el) || "(no src)";
  const code = el.error?.code;

  const codeName =
    code === MediaError.MEDIA_ERR_ABORTED
      ? "MEDIA_ERR_ABORTED"
      : code === MediaError.MEDIA_ERR_NETWORK
        ? "MEDIA_ERR_NETWORK"
        : code === MediaError.MEDIA_ERR_DECODE
          ? "MEDIA_ERR_DECODE"
          : code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED
            ? "MEDIA_ERR_SRC_NOT_SUPPORTED"
            : "MEDIA_ERR_UNKNOWN";

  // message is not standardized; may be empty
  const msg = (el.error as any)?.message ? ` - ${(el.error as any).message}` : "";
  return `Audio failed to load (${src}): ${code ?? "?"} ${codeName}${msg}`;
}
