export class AudioEngine {
  private music: HTMLAudioElement;
  private ambience: HTMLAudioElement;

  constructor() {
    if (typeof window !== "undefined") {
      this.music = new Audio();
      this.ambience = new Audio();
      this.music.loop = true;
      this.ambience.loop = true;
      this.music.preload = "auto";
      this.ambience.preload = "auto";
    } else {
      this.music = {} as HTMLAudioElement;
      this.ambience = {} as HTMLAudioElement;
    }
  }

  setSources(musicUrl: string, ambienceUrl: string) {
    if (typeof window === "undefined") return;
    this.music.src = musicUrl;
    this.ambience.src = ambienceUrl;
  }

  async play() {
    if (typeof window === "undefined") return;
    try {
      await Promise.all([this.music.play(), this.ambience.play()]);
    } catch (e) {
      console.warn("Autoplay blocked or failed", e);
    }
  }

  pause() {
    if (typeof window === "undefined") return;
    this.music.pause();
    this.ambience.pause();
  }

  setMusicVolume(v: number) {
    if (this.music && this.music.volume !== undefined) this.music.volume = clamp01(v);
  }

  setAmbienceVolume(v: number) {
    if (this.ambience && this.ambience.volume !== undefined) this.ambience.volume = clamp01(v);
  }
}

function clamp01(n: number) {
  return Math.max(0, Math.min(1, n));
}
