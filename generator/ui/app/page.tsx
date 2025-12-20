"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Activity,
  Disc3,
  Headphones,
  Loader2,
  Moon,
  Music,
  Pause,
  Play,
  Repeat,
  Settings,
  Sun,
  Volume2,
  Zap,
  Wand2,
  TimerReset,
  Gauge,
  Merge,
} from "lucide-react";

import Studio, { Track } from "@/components/studio";

// ============================================================================
// TYPES
// ============================================================================

interface DeckState {
  track: Track | null;
  isPlaying: boolean;
  volume: number; // 0..100
  eq: { low: number; mid: number; high: number }; // 0..100
  tempo: number; // 50..150 (%)
  cuePoint: number; // seconds
  loop: { enabled: boolean; start: number; end: number }; // seconds
  currentTime: number;
  duration: number;
}

interface LibraryInterfaceProps {
  tracks: Track[];
  onLoadTrack: (track: Track, deck: "A" | "B") => void;
  isDarkMode: boolean;
}

// ============================================================================
// HELPERS
// ============================================================================

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

const formatTime = (seconds: number) => {
  if (!seconds || isNaN(seconds)) return "0:00";
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, "0")}`;
};

// deterministic pseudo-random (seeded) for waveform bars
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const seedFromString = (s: string) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};

// Equal-power crossfade (sounds better than linear)
const equalPowerGains = (x01: number) => {
  const x = clamp01(x01);
  const a = Math.cos(x * Math.PI * 0.5);
  const b = Math.sin(x * Math.PI * 0.5);
  return { a, b };
};

// Spectrum bands helper (real analyser -> 36 bars)
const BAND_COUNT = 36;
const makeBands = (fft: Uint8Array, bandCount: number, prev: number[]) => {
  const out = new Array(bandCount).fill(0);
  const n = fft.length;

  for (let b = 0; b < bandCount; b++) {
    const t0 = b / bandCount;
    const t1 = (b + 1) / bandCount;

    // square curve biases low freqs (more bass detail)
    const i0 = Math.floor((t0 * t0) * (n - 1));
    const i1 = Math.max(i0 + 1, Math.floor((t1 * t1) * (n - 1)));

    let sum = 0;
    for (let i = i0; i <= i1; i++) sum += fft[i];
    const avg = sum / (i1 - i0 + 1); // 0..255

    const v01 = avg / 255;
    const curved = Math.pow(v01, 0.7) * 100; // 0..100, nicer visual curve

    // Smooth (attack faster than release)
    const p = prev?.[b] ?? 0;
    const attack = 0.35;
    const release = 0.12;
    const alpha = curved > p ? attack : release;

    // noise gate
    const gated = curved < 2.2 ? 0 : curved;

    out[b] = p + (gated - p) * alpha;
  }

  return out;
};

// ============================================================================
// MAIN
// ============================================================================

export default function DJDashboard() {
  // Theme
  const [isDarkMode, setIsDarkMode] = useState(true);

  // View
  const [activeView, setActiveView] = useState<"mixer" | "generator" | "library">("mixer");

  // Deck states
  const [deckA, setDeckA] = useState<DeckState>({
    track: null,
    isPlaying: false,
    volume: 75,
    eq: { low: 50, mid: 50, high: 50 },
    tempo: 100,
    cuePoint: 0,
    loop: { enabled: false, start: 0, end: 0 },
    currentTime: 0,
    duration: 0,
  });

  const [deckB, setDeckB] = useState<DeckState>({
    track: null,
    isPlaying: false,
    volume: 75,
    eq: { low: 50, mid: 50, high: 50 },
    tempo: 100,
    cuePoint: 0,
    loop: { enabled: false, start: 0, end: 0 },
    currentTime: 0,
    duration: 0,
  });

  // Mixer
  const [crossfader, setCrossfader] = useState(50);
  const [masterVolume, setMasterVolume] = useState(80);
  const [aiMixing, setAiMixing] = useState(false);

  // Library
  const [generatedTracks, setGeneratedTracks] = useState<Track[]>([]);
  const [autoGenerate, setAutoGenerate] = useState(false);

  // meters (0..100)
  const [vuMeterA, setVuMeterA] = useState(0);
  const [vuMeterB, setVuMeterB] = useState(0);
  const [masterMeter, setMasterMeter] = useState(0);

  // ✅ Real spectrum bars (0..100 each)
  const [masterBands, setMasterBands] = useState<number[]>(() => Array(BAND_COUNT).fill(0));

  // Audio elements
  const audioA = useRef<HTMLAudioElement | null>(null);
  const audioB = useRef<HTMLAudioElement | null>(null);

  // WebAudio graph
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceARef = useRef<MediaElementAudioSourceNode | null>(null);
  const sourceBRef = useRef<MediaElementAudioSourceNode | null>(null);

  const gainARef = useRef<GainNode | null>(null);
  const gainBRef = useRef<GainNode | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);

  const analyserARef = useRef<AnalyserNode | null>(null);
  const analyserBRef = useRef<AnalyserNode | null>(null);
  const analyserMasterRef = useRef<AnalyserNode | null>(null);

  const rafRef = useRef<number | null>(null);

  // ========================================================================
  // Init Audio + WebAudio routing
  // ========================================================================

  useEffect(() => {
    if (typeof window === "undefined") return;

    if (!audioA.current) {
      audioA.current = new Audio();
      audioA.current.crossOrigin = "anonymous";
      audioA.current.preload = "auto";
    }
    if (!audioB.current) {
      audioB.current = new Audio();
      audioB.current.crossOrigin = "anonymous";
      audioB.current.preload = "auto";
    }

    if (!audioCtxRef.current) {
      audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    const ctx = audioCtxRef.current!;

    // Build graph once
    if (!masterGainRef.current) {
      const masterGain = ctx.createGain();
      masterGain.gain.value = 0.8;

      const gainA = ctx.createGain();
      const gainB = ctx.createGain();
      gainA.gain.value = 0.5;
      gainB.gain.value = 0.5;

      const analyserA = ctx.createAnalyser();
      const analyserB = ctx.createAnalyser();
      const analyserMaster = ctx.createAnalyser();
      analyserA.fftSize = 2048;
      analyserB.fftSize = 2048;
      analyserMaster.fftSize = 2048;

      gainA.connect(analyserA);
      analyserA.connect(masterGain);

      gainB.connect(analyserB);
      analyserB.connect(masterGain);

      masterGain.connect(analyserMaster);
      analyserMaster.connect(ctx.destination);

      masterGainRef.current = masterGain;
      gainARef.current = gainA;
      gainBRef.current = gainB;
      analyserARef.current = analyserA;
      analyserBRef.current = analyserB;
      analyserMasterRef.current = analyserMaster;
    }

    // Create sources once per audio element
    try {
      if (audioA.current && !sourceARef.current) {
        sourceARef.current = ctx.createMediaElementSource(audioA.current);
        sourceARef.current.connect(gainARef.current!);
      }
      if (audioB.current && !sourceBRef.current) {
        sourceBRef.current = ctx.createMediaElementSource(audioB.current);
        sourceBRef.current.connect(gainBRef.current!);
      }
    } catch (e) {
      console.warn("WebAudio source init warning:", e);
    }

    // Make sure element volumes are 1.0 (we control via WebAudio)
    if (audioA.current) audioA.current.volume = 1;
    if (audioB.current) audioB.current.volume = 1;

    // timeupdate listeners
    const updateTimeA = () => {
      const el = audioA.current;
      if (!el) return;
      setDeckA((p) => ({
        ...p,
        currentTime: el.currentTime,
        duration: el.duration || p.track?.duration || 0,
      }));
    };
    const updateTimeB = () => {
      const el = audioB.current;
      if (!el) return;
      setDeckB((p) => ({
        ...p,
        currentTime: el.currentTime,
        duration: el.duration || p.track?.duration || 0,
      }));
    };

    audioA.current?.addEventListener("timeupdate", updateTimeA);
    audioB.current?.addEventListener("timeupdate", updateTimeB);
    audioA.current?.addEventListener("loadedmetadata", updateTimeA);
    audioB.current?.addEventListener("loadedmetadata", updateTimeB);

    // Meter loop using analyzers
    const bufA = new Uint8Array(analyserARef.current!.frequencyBinCount);
    const bufB = new Uint8Array(analyserBRef.current!.frequencyBinCount);
    const bufM = new Uint8Array(analyserMasterRef.current!.frequencyBinCount);

    const rmsToPercent = (arr: Uint8Array) => {
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i] / 255;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / arr.length);
      const p = Math.pow(rms, 0.6) * 100;
      return clamp(p, 0, 100);
    };

    const tick = () => {
      const a = analyserARef.current;
      const b = analyserBRef.current;
      const m = analyserMasterRef.current;

      if (a && b && m) {
        a.getByteFrequencyData(bufA);
        b.getByteFrequencyData(bufB);
        m.getByteFrequencyData(bufM);

        const vA = rmsToPercent(bufA);
        const vB = rmsToPercent(bufB);
        const vM = rmsToPercent(bufM);

        setVuMeterA(vA < 1.5 ? 0 : vA);
        setVuMeterB(vB < 1.5 ? 0 : vB);
        setMasterMeter(vM < 1.5 ? 0 : vM);

        setMasterBands((prev) => makeBands(bufM, BAND_COUNT, prev));
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      audioA.current?.removeEventListener("timeupdate", updateTimeA);
      audioB.current?.removeEventListener("timeupdate", updateTimeB);

      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;

      if (audioA.current) audioA.current.pause();
      if (audioB.current) audioB.current.pause();
    };
  }, []);

  // ========================================================================
  // Volume / Crossfader (WebAudio gains)
  // ========================================================================

  const updateGains = useCallback(() => {
    const ctx = audioCtxRef.current;
    const gainA = gainARef.current;
    const gainB = gainBRef.current;
    const masterGain = masterGainRef.current;
    if (!ctx || !gainA || !gainB || !masterGain) return;

    const x = clamp(crossfader, 0, 100) / 100;
    const { a, b } = equalPowerGains(x);

    const m = clamp(masterVolume, 0, 100) / 100;
    const va = clamp(deckA.volume, 0, 100) / 100;
    const vb = clamp(deckB.volume, 0, 100) / 100;

    const t = ctx.currentTime;
    const smooth = 0.03;

    masterGain.gain.setTargetAtTime(m, t, smooth);
    gainA.gain.setTargetAtTime(va * a, t, smooth);
    gainB.gain.setTargetAtTime(vb * b, t, smooth);
  }, [crossfader, masterVolume, deckA.volume, deckB.volume]);

  useEffect(() => {
    updateGains();
  }, [updateGains]);

  // ========================================================================
  // Deck Controls
  // ========================================================================

  const resumeAudioContext = useCallback(async () => {
    const ctx = audioCtxRef.current;
    if (!ctx) return;
    if (ctx.state === "suspended") {
      try {
        await ctx.resume();
      } catch (e) {
        console.warn("AudioContext resume failed:", e);
      }
    }
  }, []);

  const loadTrackToDeck = useCallback(
    async (track: Track, deck: "A" | "B") => {
      await resumeAudioContext();

      const audioEl = deck === "A" ? audioA.current : audioB.current;
      const setDeck = deck === "A" ? setDeckA : setDeckB;

      if (!audioEl) return;

      audioEl.pause();
      audioEl.currentTime = 0;

      audioEl.src = track.url;
      audioEl.load();

      setDeck((prev) => ({
        ...prev,
        track,
        isPlaying: false,
        currentTime: 0,
        duration: track.duration || 0,
      }));

      updateGains();
    },
    [resumeAudioContext, updateGains]
  );

  const togglePlay = useCallback(
    async (deck: "A" | "B") => {
      await resumeAudioContext();

      const audioEl = deck === "A" ? audioA.current : audioB.current;
      const setDeck = deck === "A" ? setDeckA : setDeckB;
      if (!audioEl || !audioEl.src) return;

      const tempo = deck === "A" ? deckA.tempo : deckB.tempo;
      audioEl.playbackRate = clamp(tempo, 50, 150) / 100;

      if (audioEl.paused) {
        try {
          await audioEl.play();
          setDeck((p) => ({ ...p, isPlaying: true }));
        } catch (e) {
          console.error("Playback failed:", e);
        }
      } else {
        audioEl.pause();
        setDeck((p) => ({ ...p, isPlaying: false }));
      }
    },
    [resumeAudioContext, deckA.tempo, deckB.tempo]
  );

  const stopDeck = useCallback((deck: "A" | "B") => {
    const audioEl = deck === "A" ? audioA.current : audioB.current;
    const setDeck = deck === "A" ? setDeckA : setDeckB;
    if (!audioEl) return;
    audioEl.pause();
    audioEl.currentTime = 0;
    setDeck((p) => ({ ...p, isPlaying: false, currentTime: 0 }));
  }, []);

  const handleSeek = useCallback((deck: "A" | "B", value: number) => {
    const audioEl = deck === "A" ? audioA.current : audioB.current;
    if (!audioEl || !audioEl.duration) return;
    audioEl.currentTime = clamp(value, 0, audioEl.duration);
  }, []);

  const setCue = useCallback((deck: "A" | "B") => {
    const audioEl = deck === "A" ? audioA.current : audioB.current;
    const setDeck = deck === "A" ? setDeckA : setDeckB;
    if (!audioEl) return;
    setDeck((p) => ({ ...p, cuePoint: audioEl.currentTime }));
  }, []);

  const jumpToCue = useCallback(
    (deck: "A" | "B") => {
      const audioEl = deck === "A" ? audioA.current : audioB.current;
      const state = deck === "A" ? deckA : deckB;
      if (!audioEl) return;
      audioEl.currentTime = clamp(state.cuePoint || 0, 0, audioEl.duration || 999999);
    },
    [deckA, deckB]
  );

  const toggleLoop = useCallback((deck: "A" | "B") => {
    const audioEl = deck === "A" ? audioA.current : audioB.current;
    const setDeck = deck === "A" ? setDeckA : setDeckB;
    if (!audioEl) return;

    setDeck((p) => {
      const enabled = !p.loop.enabled;
      const dur = audioEl.duration || p.duration || 0;
      const end = clamp(audioEl.currentTime || 0, 0, dur);
      const start = clamp(end - 8, 0, dur);
      return {
        ...p,
        loop: enabled ? { enabled: true, start, end } : { ...p.loop, enabled: false },
      };
    });
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      const elA = audioA.current;
      const elB = audioB.current;
      if (elA && deckA.loop.enabled && deckA.loop.end > deckA.loop.start) {
        if (elA.currentTime >= deckA.loop.end) elA.currentTime = deckA.loop.start;
      }
      if (elB && deckB.loop.enabled && deckB.loop.end > deckB.loop.start) {
        if (elB.currentTime >= deckB.loop.end) elB.currentTime = deckB.loop.start;
      }
    }, 50);
    return () => window.clearInterval(id);
  }, [deckA.loop, deckB.loop]);

  const setTempo = useCallback((deck: "A" | "B", tempo: number) => {
    const audioEl = deck === "A" ? audioA.current : audioB.current;
    const setDeck = deck === "A" ? setDeckA : setDeckB;
    const t = clamp(tempo, 50, 150);
    setDeck((p) => ({ ...p, tempo: t }));
    if (audioEl) audioEl.playbackRate = t / 100;
  }, []);

  const syncTempo = useCallback(() => {
    if (!deckA.track || !deckB.track) return;
    const target = deckA.isPlaying ? deckA.tempo : deckB.tempo;
    setTempo("A", target);
    setTempo("B", target);
  }, [deckA.track, deckB.track, deckA.isPlaying, deckA.tempo, deckB.tempo, setTempo]);

  // ========================================================================
  // Studio Link
  // ========================================================================

  const handleTrackGenerated = useCallback(
    (newTrack: Track) => {
      setGeneratedTracks((prev) => [newTrack, ...prev]);
      if (!deckA.track) loadTrackToDeck(newTrack, "A");
      else if (!deckB.track) loadTrackToDeck(newTrack, "B");
      else if (autoGenerate) {
        loadTrackToDeck(newTrack, crossfader < 50 ? "B" : "A");
      }
    },
    [deckA.track, deckB.track, loadTrackToDeck, autoGenerate, crossfader]
  );

  // ========================================================================
  // AI Auto-mix
  // ========================================================================

  const performAIMix = useCallback(async () => {
    if (!deckA.track || !deckB.track || aiMixing) return;
    setAiMixing(true);
    await resumeAudioContext();

    const start = clamp(crossfader, 0, 100);
    const target = start < 50 ? 100 : 0;

    if (target === 100 && !deckB.isPlaying) await togglePlay("B");
    if (target === 0 && !deckA.isPlaying) await togglePlay("A");

    const durationMs = 5500;
    const startTs = performance.now();

    const step = () => {
      const now = performance.now();
      const p = clamp((now - startTs) / durationMs, 0, 1);
      const s = p * p * (3 - 2 * p); // smoothstep
      const val = start + (target - start) * s;
      setCrossfader(val);

      if (p < 1) {
        requestAnimationFrame(step);
      } else {
        if (target === 100) stopDeck("A");
        else stopDeck("B");
        setAiMixing(false);
      }
    };

    requestAnimationFrame(step);
  }, [deckA.track, deckB.track, aiMixing, resumeAudioContext, crossfader, deckA.isPlaying, deckB.isPlaying, togglePlay, stopDeck]);

  // ========================================================================
  // UI
  // ========================================================================

  const anyPlaying = deckA.isPlaying || deckB.isPlaying;

  // ✅ Fullscreen layout:
  // - w-screen + min-h-screen
  // - remove max-w container clamp
  // - let grid/cards stretch vertically
  return (
    <div className={`min-h-screen w-screen transition-colors duration-300 ${isDarkMode ? "bg-black" : "bg-gray-50"}`}>
      {/* Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div
          className={`absolute inset-0 ${
            isDarkMode ? "bg-gradient-to-br from-purple-900/15 via-black to-blue-900/15" : "bg-gradient-to-br from-purple-50 via-white to-blue-50"
          }`}
        />
        <div className="absolute -top-40 left-1/2 h-[720px] w-[720px] -translate-x-1/2 rounded-full bg-gradient-to-r from-purple-600/10 to-blue-600/10 blur-3xl" />
      </div>

      {/* ✅ Full width wrapper */}
      <div className="relative z-10 px-4 md:px-8 py-4 md:py-6 space-y-4 w-full">
        {/* Header */}
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-2xl ${isDarkMode ? "bg-purple-500/10" : "bg-purple-500/20"}`}>
              <Disc3 className={`w-8 h-8 ${isDarkMode ? "text-purple-300" : "text-purple-700"} animate-spin-slow`} />
            </div>
            <div>
              <h1 className={`text-2xl md:text-3xl font-bold ${isDarkMode ? "text-white" : "text-gray-900"}`}>
                SoundFlow AI DJ
              </h1>
              <p className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
                Performance mixer + infinite procedural studio
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 justify-between md:justify-end">
            <div className="flex items-center gap-2">
              <Badge
                variant="secondary"
                className={`px-3 py-1 ${
                  isDarkMode ? "bg-green-500/10 text-green-400 border border-green-500/20" : "bg-green-50 text-green-700 border border-green-200"
                }`}
              >
                <Activity className="w-3 h-3 mr-2 animate-pulse" />
                ONLINE
              </Badge>

              <Badge
                variant="secondary"
                className={`${isDarkMode ? "bg-white/5 text-white/85" : "bg-black/5 text-black/70"} border border-white/10`}
              >
                <Gauge className="w-3 h-3 mr-2" />
                MASTER {Math.round(masterMeter)}%
              </Badge>
            </div>

            <Button variant="outline" size="icon" onClick={() => setIsDarkMode((v) => !v)} className="rounded-2xl">
              {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </Button>
          </div>
        </div>

        {/* Workspace Tabs */}
        <Tabs value={activeView} onValueChange={(v) => setActiveView(v as any)} className="w-full">
          <TabsList
            className={`grid w-full max-w-lg grid-cols-3 mx-auto mb-4 rounded-2xl p-1 border ${
              isDarkMode ? "bg-gray-900/60 border-white/10" : "bg-white border-black/10"
            }`}
          >
            {[
              { value: "mixer", label: "Mixer" },
              { value: "generator", label: "Studio" },
              { value: "library", label: "Library" },
            ].map((t) => (
              <TabsTrigger
                key={t.value}
                value={t.value}
                className={[
                  "rounded-xl transition-colors",
                  isDarkMode ? "text-white/75 hover:text-white" : "text-black/70 hover:text-black",
                  "data-[state=active]:text-white data-[state=active]:bg-white/10 data-[state=active]:shadow-sm",
                  !isDarkMode && "data-[state=active]:text-black data-[state=active]:bg-black/5",
                ]
                  .filter(Boolean)
                  .join(" ")}
              >
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>

          {/* MIXER */}
          <TabsContent value="mixer" className="mt-0">
            {/* ✅ Make mixer fill the viewport height minus header/tabs area */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 min-h-[calc(100vh-170px)]">
              {/* DECK A */}
              <div className="xl:col-span-4 h-full">
                <DeckInterface
                  deck="A"
                  state={deckA}
                  isDarkMode={isDarkMode}
                  vuMeter={vuMeterA}
                  onTogglePlay={() => togglePlay("A")}
                  onStop={() => stopDeck("A")}
                  onSeek={(v) => handleSeek("A", v)}
                  onVolumeChange={(v) => setDeckA((p) => ({ ...p, volume: v }))}
                  onTempoChange={(v) => setTempo("A", v)}
                  onCueSet={() => setCue("A")}
                  onCueJump={() => jumpToCue("A")}
                  onLoopToggle={() => toggleLoop("A")}
                />
              </div>

              {/* MIXER CENTER */}
              <div className="xl:col-span-4 flex flex-col gap-6 h-full">
                <Card
                  className={`h-full rounded-3xl ${
                    isDarkMode ? "bg-gray-950/70 border-gray-800" : "bg-white border-gray-200"
                  } backdrop-blur-xl shadow-2xl overflow-hidden`}
                >
                  <CardContent className="p-6 md:p-8 space-y-6 h-full flex flex-col">
                    {/* Mixer Header */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className={`h-10 w-10 rounded-2xl flex items-center justify-center ${isDarkMode ? "bg-white/5" : "bg-black/5"}`}>
                          <Merge className={`${isDarkMode ? "text-white/85" : "text-black/70"} w-5 h-5`} />
                        </div>
                        <div>
                          <div className={`font-bold ${isDarkMode ? "text-white" : "text-gray-900"}`}>Mixer</div>
                          <div className={`text-xs ${isDarkMode ? "text-white/55" : "text-black/45"}`}>
                            Equal-power crossfade • WebAudio meters
                          </div>
                        </div>
                      </div>

                      <Button
                        variant={isDarkMode ? "secondary" : "outline"}
                        size="sm"
                        className="rounded-2xl"
                        onClick={syncTempo}
                        disabled={!deckA.track || !deckB.track}
                      >
                        <Wand2 className="w-4 h-4 mr-2" />
                        SYNC TEMPO
                      </Button>
                    </div>

                    {/* ✅ Realistic spectrum (true analyser bands) */}
                    <SpectrumReal isDarkMode={isDarkMode} bands={masterBands} isActive={anyPlaying} />

                    {/* Crossfader */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs font-bold tracking-widest">
                        <span className="text-blue-400">DECK A</span>
                        <span className={`${isDarkMode ? "text-white/60" : "text-black/50"}`}>CROSSFADER</span>
                        <span className="text-purple-400">DECK B</span>
                      </div>
                      <div className="relative">
                        <div className="pointer-events-none absolute inset-y-0 left-1/2 w-px bg-white/25" />
                        <Slider value={[crossfader]} onValueChange={([v]) => setCrossfader(v)} min={0} max={100} step={1} className="py-4" />
                        <div className={`text-[11px] mt-1 ${isDarkMode ? "text-white/55" : "text-black/45"}`}>
                          {crossfader < 45 ? "Focus: Deck A" : crossfader > 55 ? "Focus: Deck B" : "Centered"}
                        </div>
                      </div>
                    </div>

                    {/* Master */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs font-bold tracking-widest">
                        <span className={`${isDarkMode ? "text-white/70" : "text-black/60"}`}>MASTER OUT</span>
                        <span className={`${isDarkMode ? "text-white/70" : "text-black/60"}`}>{masterVolume}%</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <Volume2 className={`${isDarkMode ? "text-white/70" : "text-black/60"} w-4 h-4`} />
                        <Slider value={[masterVolume]} onValueChange={([v]) => setMasterVolume(v)} min={0} max={100} step={1} />
                      </div>
                      <MeterBar value={masterMeter} isDarkMode={isDarkMode} label="MASTER" />
                    </div>

                    {/* Push bottom controls to bottom */}
                    <div className="mt-auto space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        <Button
                          className={`h-12 rounded-2xl font-bold ${
                            isDarkMode ? "bg-white/5 hover:bg-white/10 text-white border border-white/10" : "bg-black/5 hover:bg-black/10 text-black border border-black/10"
                          }`}
                          onClick={() => setAutoGenerate((v) => !v)}
                          variant="ghost"
                        >
                          <Repeat className="w-4 h-4 mr-2" />
                          Auto-gen: {autoGenerate ? "ON" : "OFF"}
                        </Button>

                        <Button
                          className={`h-12 rounded-2xl font-bold ${
                            isDarkMode ? "bg-gradient-to-r from-purple-600 to-blue-600" : "bg-black text-white"
                          }`}
                          onClick={performAIMix}
                          disabled={aiMixing || !deckA.track || !deckB.track}
                        >
                          {aiMixing ? <Loader2 className="w-5 h-5 mr-2 animate-spin" /> : <Zap className="w-5 h-5 mr-2" />}
                          {aiMixing ? "MIXING..." : "AUTO MIX"}
                        </Button>
                      </div>

                      <div className={`text-xs ${isDarkMode ? "text-white/55" : "text-black/45"}`}>
                        Tip: Load tracks on both decks, hit <span className="font-semibold">SYNC TEMPO</span>, then <span className="font-semibold">AUTO MIX</span>.
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* DECK B */}
              <div className="xl:col-span-4 h-full">
                <DeckInterface
                  deck="B"
                  state={deckB}
                  isDarkMode={isDarkMode}
                  vuMeter={vuMeterB}
                  onTogglePlay={() => togglePlay("B")}
                  onStop={() => stopDeck("B")}
                  onSeek={(v) => handleSeek("B", v)}
                  onVolumeChange={(v) => setDeckB((p) => ({ ...p, volume: v }))}
                  onTempoChange={(v) => setTempo("B", v)}
                  onCueSet={() => setCue("B")}
                  onCueJump={() => jumpToCue("B")}
                  onLoopToggle={() => toggleLoop("B")}
                />
              </div>
            </div>
          </TabsContent>

          {/* STUDIO */}
          <TabsContent value="generator">
            <div className="min-h-[calc(100vh-170px)]">
              <Studio onTrackGenerated={handleTrackGenerated} isDarkMode={isDarkMode} />
            </div>
          </TabsContent>

          {/* LIBRARY */}
          <TabsContent value="library">
            <div className="min-h-[calc(100vh-170px)]">
              <LibraryInterface tracks={generatedTracks} onLoadTrack={loadTrackToDeck} isDarkMode={isDarkMode} />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

// ============================================================================
// ✅ REALISTIC SPECTRUM (uses analyser bands, no fake motion when idle)
// ============================================================================

function SpectrumReal({
  isDarkMode,
  bands,
  isActive,
}: {
  isDarkMode: boolean;
  bands: number[]; // 0..100, length 36
  isActive: boolean;
}) {
  const safeBands = bands?.length ? bands : Array(BAND_COUNT).fill(0);
  const peak = Math.max(...safeBands);

  if (!isActive || peak < 2) {
    return (
      <div
        className={`h-28 rounded-2xl border overflow-hidden relative flex items-center justify-center ${
          isDarkMode ? "bg-black/40 border-white/10" : "bg-gray-100 border-black/10"
        }`}
      >
        <div className={`text-xs ${isDarkMode ? "text-white/55" : "text-black/45"}`}>
          No audio — load a track and press Play
        </div>
      </div>
    );
  }

  return (
    <div
      className={`h-28 rounded-2xl border overflow-hidden relative ${
        isDarkMode ? "bg-black/40 border-white/10" : "bg-gray-100 border-black/10"
      }`}
    >
      <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent pointer-events-none" />
      <div className="h-full p-4 flex items-end gap-1">
        {safeBands.map((v, i) => {
          const h = Math.max(4, Math.min(100, v));
          return (
            <div
              key={i}
              className="flex-1 rounded-t-sm opacity-95"
              style={{
                height: `${h}%`,
                background: "linear-gradient(to top, rgba(147,51,234,0.95), rgba(59,130,246,0.85))",
              }}
            />
          );
        })}
      </div>

      <div className={`absolute bottom-2 right-3 text-[11px] ${isDarkMode ? "text-white/45" : "text-black/45"}`}>
        Spectrum (live)
      </div>
    </div>
  );
}

// ============================================================================
// COMPONENT: DECK UI
// ============================================================================

interface DeckProps {
  deck: "A" | "B";
  state: DeckState;
  isDarkMode: boolean;
  vuMeter: number;
  onTogglePlay: () => void;
  onStop: () => void;
  onSeek: (val: number) => void;
  onVolumeChange: (val: number) => void;
  onTempoChange: (val: number) => void;
  onCueSet: () => void;
  onCueJump: () => void;
  onLoopToggle: () => void;
}

function DeckInterface({
  deck,
  state,
  isDarkMode,
  vuMeter,
  onTogglePlay,
  onStop,
  onSeek,
  onVolumeChange,
  onTempoChange,
  onCueSet,
  onCueJump,
  onLoopToggle,
}: DeckProps) {
  const isA = deck === "A";
  const accentText = isA ? "text-blue-300" : "text-purple-300";
  const accentBorder = isA ? "border-blue-500/30" : "border-purple-500/30";
  const accentBg = isA ? "bg-blue-500/10" : "bg-purple-500/10";
  const playedBar = isA ? "bg-blue-500" : "bg-purple-500";

  const seed = useMemo(() => seedFromString(state.track?.id ?? `${deck}-empty`), [state.track?.id, deck]);
  const waveformBars = useMemo(() => {
    const rand = mulberry32(seed);
    return Array.from({ length: 84 }).map(() => 18 + rand() * 72);
  }, [seed]);

  const progressPercent = state.duration > 0 ? (state.currentTime / state.duration) * 100 : 0;
  const remaining = Math.max(0, (state.duration || 0) - (state.currentTime || 0));

  const deckReady = !!state.track;

  return (
    <Card
      className={`h-full rounded-3xl overflow-hidden border ${
        isDarkMode ? "bg-gray-950/70 border-gray-800" : "bg-white border-gray-200"
      } backdrop-blur-xl shadow-2xl`}
    >
      <CardContent className="p-6 flex flex-col h-full gap-5">
        {/* Header */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className={`h-11 w-11 rounded-2xl flex items-center justify-center ${accentBg} border ${accentBorder}`}>
              <Disc3 className={`${accentText} w-5 h-5`} />
            </div>
            <div>
              <div className={`text-xs font-bold tracking-widest ${isDarkMode ? "text-white/65" : "text-black/45"}`}>
                DECK {deck}
              </div>
              <div className={`font-bold ${isDarkMode ? "text-white" : "text-gray-900"} leading-tight`}>
                {state.track?.name ?? "Empty deck"}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button size="icon" variant="ghost" className="rounded-2xl" title="Cue in headphones">
              <Headphones className="w-5 h-5 opacity-70" />
            </Button>
            <Button size="icon" variant="ghost" className="rounded-2xl" title="Deck settings">
              <Settings className="w-5 h-5 opacity-70" />
            </Button>
          </div>
        </div>

        {/* Track Meta */}
        <div className={`rounded-2xl border p-4 ${isDarkMode ? "bg-black/45 border-white/10" : "bg-gray-50 border-black/10"}`}>
          {state.track ? (
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary" className={`border ${accentBorder} ${isDarkMode ? "bg-white/10 text-white" : "bg-black/5"}`}>
                {state.track.bpm} BPM
              </Badge>
              <Badge variant="secondary" className={`${isDarkMode ? "bg-white/10 text-white" : "bg-black/5"}`}>
                {state.track.key}
              </Badge>
              <Badge variant="secondary" className={`${isDarkMode ? "bg-white/10 text-white" : "bg-black/5"}`}>
                {state.track.genre}
              </Badge>

              <div className="ml-auto flex items-center gap-2">
                <MeterBar value={vuMeter} isDarkMode={isDarkMode} label="VU" compact />
              </div>
            </div>
          ) : (
            <div className={`flex items-center justify-between ${isDarkMode ? "text-white/60" : "text-black/45"}`}>
              <div className="flex items-center gap-2">
                <Music className="w-4 h-4" />
                <span className="text-sm">Load a track from Studio or Library</span>
              </div>
              <Badge variant="outline" className={`${isDarkMode ? "border-white/15 text-white/80" : "border-black/10"}`}>
                READY
              </Badge>
            </div>
          )}
        </div>

        {/* Waveform (hide “fake” waveform if empty deck) */}
        {!deckReady ? (
          <div className={`h-24 rounded-2xl border flex items-center justify-center ${isDarkMode ? "bg-black/35 border-white/10" : "bg-gray-100 border-black/10"}`}>
            <div className={`text-xs ${isDarkMode ? "text-white/55" : "text-black/45"}`}>Waveform appears when a track is loaded</div>
          </div>
        ) : (
          <div className="select-none">
            <div className="flex items-center justify-between text-[11px] font-mono font-bold opacity-70 mb-2">
              <span>{formatTime(state.currentTime)}</span>
              <span>-{formatTime(remaining)}</span>
            </div>

            <div className="relative">
              <div className={`flex items-end gap-[2px] h-20 w-full rounded-2xl border overflow-hidden p-3 ${isDarkMode ? "bg-black/40 border-white/10" : "bg-gray-100 border-black/10"}`}>
                {waveformBars.map((h, i) => {
                  const barPos = (i / waveformBars.length) * 100;
                  const played = barPos <= progressPercent;
                  return (
                    <div
                      key={i}
                      className={`flex-1 rounded-full transition-colors duration-100 ${played ? playedBar : isDarkMode ? "bg-white/10" : "bg-black/10"}`}
                      style={{ height: `${h}%`, opacity: played ? 1 : 0.5 }}
                    />
                  );
                })}
              </div>

              <Slider
                value={[state.currentTime]}
                max={state.duration || 1}
                step={0.05}
                onValueChange={([val]) => onSeek(val)}
                className="absolute inset-0 h-20 opacity-0 cursor-pointer"
                disabled={!deckReady}
              />

              <div
                className="absolute top-3 bottom-3 w-[2px] bg-white/80 pointer-events-none shadow-[0_0_12px_rgba(255,255,255,0.7)]"
                style={{ left: `calc(${progressPercent}% + 12px)` }}
              />
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="grid grid-cols-12 gap-4 items-center">
          <div className="col-span-12 md:col-span-5 flex items-center gap-3">
            <Button
              size="icon"
              className={`w-14 h-14 rounded-2xl shadow-2xl transition-transform active:scale-95 ${
                state.isPlaying
                  ? isDarkMode
                    ? "bg-white text-black hover:bg-gray-200"
                    : "bg-black text-white hover:bg-black/90"
                  : isDarkMode
                  ? "bg-white/10 hover:bg-white/15 text-white"
                  : "bg-black/5 hover:bg-black/10 text-black"
              }`}
              onClick={onTogglePlay}
              disabled={!deckReady}
              title={state.isPlaying ? "Pause" : "Play"}
            >
              {state.isPlaying ? <Pause className="w-7 h-7" /> : <Play className="w-7 h-7 ml-[2px]" />}
            </Button>

            <div className="flex flex-col gap-2 flex-1">
              <div className="flex items-center justify-between">
                <div className={`text-[11px] font-bold tracking-widest ${isDarkMode ? "text-white/65" : "text-black/45"}`}>
                  VOLUME
                </div>
                <div className={`text-[11px] font-mono ${isDarkMode ? "text-white/65" : "text-black/45"}`}>
                  {Math.round(state.volume)}%
                </div>
              </div>
              <Slider value={[state.volume]} max={100} onValueChange={([v]) => onVolumeChange(v)} disabled={!deckReady} />
            </div>

            <Button size="icon" variant="ghost" className="rounded-2xl" onClick={onStop} disabled={!deckReady} title="Stop">
              <TimerReset className="w-5 h-5 opacity-70" />
            </Button>
          </div>

          <div className="col-span-12 md:col-span-4">
            <div className="flex items-center justify-between">
              <div className={`text-[11px] font-bold tracking-widest ${isDarkMode ? "text-white/65" : "text-black/45"}`}>
                TEMPO
              </div>
              <div className={`text-[11px] font-mono ${isDarkMode ? "text-white/65" : "text-black/45"}`}>
                {Math.round(state.tempo)}%
              </div>
            </div>
            <Slider value={[state.tempo]} min={50} max={150} step={1} onValueChange={([v]) => onTempoChange(v)} disabled={!deckReady} />
            <div className={`mt-1 text-[11px] ${isDarkMode ? "text-white/55" : "text-black/45"}`}>
              Playback rate: {(state.tempo / 100).toFixed(2)}x
            </div>
          </div>

          <div className="col-span-12 md:col-span-3 flex md:flex-col gap-2 md:items-stretch">
            <Button variant={isDarkMode ? "secondary" : "outline"} className="rounded-2xl w-full" onClick={onCueSet} disabled={!deckReady}>
              <Badge variant="secondary" className="mr-2 bg-transparent border border-white/15">
                CUE
              </Badge>
              Set
            </Button>
            <Button variant={isDarkMode ? "secondary" : "outline"} className="rounded-2xl w-full" onClick={onCueJump} disabled={!deckReady}>
              Jump
            </Button>

            <Button
              variant={state.loop.enabled ? "default" : isDarkMode ? "secondary" : "outline"}
              className={`rounded-2xl w-full ${state.loop.enabled ? (isDarkMode ? "bg-white text-black hover:bg-gray-200" : "bg-black text-white hover:bg-black/90") : ""}`}
              onClick={onLoopToggle}
              disabled={!deckReady}
            >
              <Repeat className="w-4 h-4 mr-2" />
              {state.loop.enabled ? "LOOP ON" : "LOOP"}
            </Button>
          </div>
        </div>

        {/* Status row */}
        <div className={`grid grid-cols-3 gap-3 text-xs ${isDarkMode ? "text-white/60" : "text-black/45"}`}>
          <div className={`rounded-2xl border p-3 ${isDarkMode ? "border-white/10 bg-white/5" : "border-black/10 bg-black/5"}`}>
            <div className="font-bold tracking-widest mb-1">CUE</div>
            <div className="font-mono">{formatTime(state.cuePoint)}</div>
          </div>

          <div className={`rounded-2xl border p-3 ${isDarkMode ? "border-white/10 bg-white/5" : "border-black/10 bg-black/5"}`}>
            <div className="font-bold tracking-widest mb-1">LOOP</div>
            <div className="font-mono">{state.loop.enabled ? `${formatTime(state.loop.start)} → ${formatTime(state.loop.end)}` : "OFF"}</div>
          </div>

          <div className={`rounded-2xl border p-3 ${isDarkMode ? "border-white/10 bg-white/5" : "border-black/10 bg-black/5"}`}>
            <div className="font-bold tracking-widest mb-1">STATE</div>
            <div className="font-mono">{state.isPlaying ? "PLAYING" : deckReady ? "PAUSED" : "EMPTY"}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Meter bar
// ============================================================================

function MeterBar({
  value,
  isDarkMode,
  label,
  compact,
}: {
  value: number;
  isDarkMode: boolean;
  label: string;
  compact?: boolean;
}) {
  const v = clamp(value, 0, 100);
  const hot = v > 92;

  return (
    <div className={`flex items-center gap-2 ${compact ? "" : "mt-2"}`}>
      <div className={`text-[10px] font-bold tracking-widest ${isDarkMode ? "text-white/65" : "text-black/45"}`}>
        {label}
      </div>
      <div
        className={`h-3 ${compact ? "w-24" : "w-full"} rounded-full overflow-hidden border ${
          isDarkMode ? "bg-black/30 border-white/12" : "bg-black/5 border-black/10"
        }`}
        title={`${Math.round(v)}%`}
      >
        <div className={`h-full transition-all duration-100 ${hot ? "bg-red-500" : "bg-green-500"}`} style={{ width: `${v}%` }} />
      </div>
    </div>
  );
}

// ============================================================================
// LIBRARY
// ============================================================================

function LibraryInterface({ tracks, onLoadTrack, isDarkMode }: LibraryInterfaceProps) {
  return (
    <Card className={`rounded-3xl ${isDarkMode ? "bg-gray-950/70 border-gray-800" : "bg-white border-gray-200"} backdrop-blur-xl shadow-2xl`}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between gap-3 mb-4">
          <div>
            <h2 className={`text-xl font-bold ${isDarkMode ? "text-white" : "text-gray-900"}`}>Track Library</h2>
            <p className={`text-sm ${isDarkMode ? "text-white/65" : "text-black/45"}`}>Load to Deck A / B • Generated tracks appear here</p>
          </div>
          <Badge variant="secondary" className={`${isDarkMode ? "bg-white/5" : "bg-black/5"} border border-white/10`}>
            {tracks.length} tracks
          </Badge>
        </div>

        <ScrollArea className="h-[calc(100vh-260px)] pr-4">
          <div className="space-y-2">
            {tracks.map((t) => (
              <div
                key={t.id}
                className={`group flex items-center justify-between gap-3 p-3 rounded-2xl border transition-all ${
                  isDarkMode ? "bg-white/5 hover:bg-white/8 border-white/10" : "bg-black/5 hover:bg-black/8 border-black/10"
                }`}
              >
                <div className="flex items-center gap-4 min-w-0">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center shrink-0">
                    <Music className="w-5 h-5 text-white" />
                  </div>
                  <div className="min-w-0">
                    <div className={`font-bold text-sm truncate ${isDarkMode ? "text-white" : "text-gray-900"}`}>{t.name}</div>
                    <div className={`text-xs flex gap-2 ${isDarkMode ? "text-white/65" : "text-black/45"}`}>
                      <span>{t.bpm} BPM</span>
                      <span>•</span>
                      <span className="truncate">{t.genre}</span>
                      {t.key ? (
                        <>
                          <span>•</span>
                          <span>{t.key}</span>
                        </>
                      ) : null}
                    </div>
                  </div>
                </div>

                <div className="flex gap-2 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity shrink-0">
                  <Button size="sm" variant={isDarkMode ? "secondary" : "outline"} className="rounded-2xl" onClick={() => onLoadTrack(t, "A")}>
                    Load A
                  </Button>
                  <Button size="sm" variant={isDarkMode ? "secondary" : "outline"} className="rounded-2xl" onClick={() => onLoadTrack(t, "B")}>
                    Load B
                  </Button>
                </div>
              </div>
            ))}

            {tracks.length === 0 && (
              <div className={`text-center py-24 ${isDarkMode ? "text-white/65" : "text-black/45"}`}>
                <Disc3 className="w-12 h-12 mx-auto mb-3 opacity-70" />
                <p className="font-bold">Library empty</p>
                <p className="text-sm opacity-80">Generate tracks in Studio to populate your crate.</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
