/**
 * Minimal production-ready audio engine:
 * - 2 HTMLAudio tracks (music + ambience)
 * - independent gain (volume)
 * - safe for SSR
 *
 * Later upgrade path:
 * - WebAudio API for better mixing + effects + crossfade
 * - waveform analyzer node
 */
export class AudioEngine {
  private music: HTMLAudioElement | null = null;
  private ambience: HTMLAudioElement | null = null;

  constructor() {
    if (typeof window !== "undefined") {
      this.music = new Audio();
      this.ambience = new Audio();

      this.music.loop = true;
      this.ambience.loop = true;

      this.music.preload = "auto";
      this.ambience.preload = "auto";

      // iOS Safari: allow inline playback
      // @ts-ignore
      this.music.playsInline = true;
      // @ts-ignore
      this.ambience.playsInline = true;
    }
  }

  setSources(musicUrl: string, ambienceUrl?: string) {
    if (!this.music) return;

    this.music.src = musicUrl;
    if (this.ambience) {
      this.ambience.src = ambienceUrl ?? "";
    }
  }

  async play() {
    if (!this.music) return;

    try {
      const tasks: Promise<unknown>[] = [this.music.play()];
      if (this.ambience && this.ambience.src) tasks.push(this.ambience.play());
      await Promise.all(tasks);
    } catch (e) {
      // Most common: autoplay policy until user gesture
      console.warn("Audio play blocked or failed. Call play() from a click handler.", e);
    }
  }

  pause() {
    this.music?.pause();
    this.ambience?.pause();
  }

  stop() {
    if (this.music) {
      this.music.pause();
      this.music.currentTime = 0;
    }
    if (this.ambience) {
      this.ambience.pause();
      this.ambience.currentTime = 0;
    }
  }

  setMusicVolume(v01: number) {
    if (!this.music) return;
    this.music.volume = clamp01(v01);
  }

  setAmbienceVolume(v01: number) {
    if (!this.ambience) return;
    this.ambience.volume = clamp01(v01);
  }

  isReady() {
    return !!this.music;
  }
}

function clamp01(n: number) {
  return Math.max(0, Math.min(1, n));
}
