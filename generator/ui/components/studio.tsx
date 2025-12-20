"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Dna,
  RefreshCw,
  Lock,
  Unlock,
  TrendingUp,
  Activity,
  Repeat,
  Disc3,
  Music,
  CloudRain,
  Sliders,
  Brain,
  Zap,
  Loader2,
  CheckCircle2,
  Headphones,
} from "lucide-react";

// ============================================================================
// CONFIGURATION & TYPES
// ============================================================================

const API_BASE = "http://localhost:8000";

const GENRES = ["Techno", "House", "Trance", "Lofi", "Ambient", "Jazz"];

const LAYERS = [
  { id: "drums", label: "Drums", icon: Disc3 },
  { id: "bass", label: "Bass", icon: Activity },
  { id: "music", label: "Melody", icon: Music },
  { id: "pad", label: "Atmosphere", icon: CloudRain },
];

export interface Track {
  id: string;
  name: string;
  url: string;
  bpm: number;
  key: string;
  genre: string;
  duration: number;
  seed?: string;
}

interface StudioProps {
  onTrackGenerated?: (track: Track) => void;
  isDarkMode?: boolean;
  className?: string;
}

// ============================================================================
// SMALL HELPERS
// ============================================================================

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

const fmtDuration = (s: number) => {
  if (!s || Number.isNaN(s)) return "0:00";
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, "0")}`;
};

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(" ");
}

// ============================================================================
// STUDIO COMPONENT
// ============================================================================

export default function Studio({ onTrackGenerated, isDarkMode = true, className = "" }: StudioProps) {
  // --- GENERATION STATE ---
  const [isGenerating, setIsGenerating] = useState(false);
  const [genStatus, setGenStatus] = useState(""); // UI status line
  const statusTimerRef = useRef<number | null>(null);

  // after generation: show what was created
  const [lastGenerated, setLastGenerated] = useState<Track | null>(null);
  const [lastMeta, setLastMeta] = useState<{
    binaural: "off" | "focus" | "relax";
    layers: string[];
    intensity: number;
    energyCurve: string;
    ambience: { rain: number; vinyl: number; white: number };
  } | null>(null);

  // --- DNA PANEL STATE ---
  const [genre, setGenre] = useState<(typeof GENRES)[number]>("Techno");
  const [bpm, setBpm] = useState(128);
  const [seed, setSeed] = useState("");
  const [isLocked, setIsLocked] = useState(false);

  // ✅ KEEP binaural as a “second option”
  const [binaural, setBinaural] = useState<"off" | "focus" | "relax">("off");

  // --- STRUCTURE PANEL STATE ---
  const [layers, setLayers] = useState<string[]>(["drums", "bass", "music", "pad"]);
  const [energyCurve, setEnergyCurve] = useState<"linear" | "drop" | "peak" | "custom">("peak");

  // Manual Energy Drawing (32 bars resolution)
  const [energyLevels, setEnergyLevels] = useState<number[]>(Array(32).fill(50));
  const isDrawing = useRef(false);

  // --- MIXER PANEL STATE ---
  const [intensity, setIntensity] = useState(50);
  const [ambience, setAmbience] = useState({ rain: 0, vinyl: 0, white: 0 });
  const [synthParams, setSynthParams] = useState({
    cutoff: 75,
    resonance: 30,
    drive: 10,
    space: 20,
  });

  // Initialize seed on mount
  useEffect(() => {
    randomizeSeed();
    applyPresetCurve("peak");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const randomizeSeed = () => {
    if (!isLocked) setSeed(Math.random().toString(36).substring(7).toUpperCase());
  };

  const toggleLayer = (id: string) => {
    setLayers((prev) => (prev.includes(id) ? prev.filter((l) => l !== id) : [...prev, id]));
  };

  // --- ENERGY CURVE PRESETS ---
  const applyPresetCurve = (type: "linear" | "drop" | "peak") => {
    setEnergyCurve(type);
    const newLevels = Array(32)
      .fill(0)
      .map((_, i) => {
        let h = 40;
        if (type === "peak") h = 20 + Math.pow(i / 32, 2) * 80;
        if (type === "drop") h = i > 12 && i < 20 ? 10 : 60;
        if (type === "linear") h = 50 + Math.sin(i) * 5;
        return clamp(h, 10, 100);
      });
    setEnergyLevels(newLevels);
  };

  const handleDraw = (index: number, e: React.MouseEvent) => {
    if (e.buttons !== 1) return;

    const rect = (e.target as HTMLDivElement).getBoundingClientRect();
    const height = 100 - ((e.clientY - rect.top) / rect.height) * 100;

    const next = [...energyLevels];
    next[index] = clamp(height, 5, 100);
    setEnergyLevels(next);
    setEnergyCurve("custom");
  };

  // ============================================================================
  // COLOR TOKENS (READABLE IN DARK MODE)
  // ============================================================================

  // Surfaces
  const bgCard = isDarkMode ? "bg-gray-950/55 border-white/10" : "bg-white border-black/10";
  const surface = isDarkMode ? "bg-black/35 border-white/10" : "bg-gray-50 border-black/10";

  // Text rules
  const textPrimary = isDarkMode ? "text-white" : "text-gray-900";
  const textSecondary = isDarkMode ? "text-white/75" : "text-black/60";
  const textTertiary = isDarkMode ? "text-white/60" : "text-black/50";

  // Buttons/inputs
  const softBtn = isDarkMode
    ? "bg-white/5 hover:bg-white/10 border border-white/10 text-white"
    : "bg-black/5 hover:bg-black/10 border border-black/10 text-black";

  // ============================================================================
  // API LOGIC
  // ============================================================================

  const clearStatusTimer = () => {
    if (statusTimerRef.current) {
      window.clearInterval(statusTimerRef.current);
      statusTimerRef.current = null;
    }
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    setGenStatus("Initializing Engines...");

    clearStatusTimer();
    statusTimerRef.current = window.setInterval(() => {
      setGenStatus((prev) => {
        if (prev.includes("Initializing")) return "Synthesizing Audio Stems...";
        if (prev.includes("Synthesizing")) return "Applying Mastering Chain...";
        return prev;
      });
    }, 2500);

    const payload = {
      genre,
      bpm,
      key: "A",
      seed: isLocked ? seed : undefined,
      layers,
      duration: 180,
      binaural, // ✅ preserved
      ambience,
      intensity,
      synth_params: synthParams,
      energy_curve: energyCurve,
      target_lufs: -14.0,
    };

    // backend compatibility mapping
    const backendPayload = {
      ...payload,
      energy_curve: energyCurve === "custom" ? "peak" : energyCurve,
    };

    try {
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(backendPayload),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || "Server error");
      }

      const data = await res.json();

      const newTrack: Track = {
        id: data.id,
        name: data.name,
        url: data.url,
        bpm: data.bpm,
        key: data.key,
        genre: data.genre,
        duration: data.duration,
        seed: data.seed,
      };

      // ✅ show what we created (confidence + debug)
      setLastGenerated(newTrack);
      setLastMeta({
        binaural,
        layers: [...layers],
        intensity,
        energyCurve: backendPayload.energy_curve,
        ambience: { ...ambience },
      });

      onTrackGenerated?.(newTrack);

      setGenStatus("Done ✓ Track added to Library");

      if (!isLocked) randomizeSeed();
    } catch (err) {
      console.error("Generation Failed:", err);
      setGenStatus("Failed — check backend logs / console");
      alert("Failed to generate. Check console.");
    } finally {
      clearStatusTimer();
      setIsGenerating(false);
      // keep status visible briefly
      window.setTimeout(() => setGenStatus(""), 1800);
    }
  };

  // small derived UI text
  const binauralLabel = useMemo(() => {
    if (binaural === "off") return "Off";
    if (binaural === "focus") return "Focus";
    return "Relax";
  }, [binaural]);

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div className={cx("grid grid-cols-1 lg:grid-cols-12 gap-6", className)}>
      {/* === MODULE A: DNA ENGINE (LEFT) === */}
      <Card className={cx("lg:col-span-3 backdrop-blur-xl rounded-3xl", bgCard)}>
        <CardContent className="p-6 space-y-6">
          <div className="flex items-center gap-2 mb-2 text-purple-400">
            <Dna className="w-5 h-5" />
            <h2 className="font-bold text-sm tracking-wider">DNA ENGINE</h2>
          </div>

          {/* Genre Selector (FIXED: readable in dark mode) */}
          <div className="space-y-3">
            <label className={cx("text-xs font-bold", textSecondary)}>GENRE</label>
            <div className="grid grid-cols-2 gap-2">
              {GENRES.map((g) => {
                const active = genre === g;
                return (
                  <Button
                    key={g}
                    type="button"
                    size="sm"
                    className={cx(
                      "justify-start rounded-xl border",
                      active
                        ? "bg-purple-600 hover:bg-purple-700 text-white border-purple-500/40"
                        : isDarkMode
                        ? "bg-white/5 hover:bg-white/10 text-white border-white/10"
                        : "bg-black/5 hover:bg-black/10 text-black border-black/10"
                    )}
                    onClick={() => setGenre(g)}
                  >
                    {g}
                  </Button>
                );
              })}
            </div>
          </div>

          {/* BPM */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className={cx("text-xs font-bold", textSecondary)}>BPM</label>
              <span className={cx("text-xs font-mono", textSecondary)}>{bpm}</span>
            </div>
            <Slider value={[bpm]} min={60} max={180} step={1} onValueChange={([v]) => setBpm(v)} />
            <div className={cx("text-[11px]", textTertiary)}>Club: 120–135 • Lofi: 70–95</div>
          </div>

          {/* Seed Control */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <label className={cx("text-xs font-bold", textSecondary)}>INFINITE SEED</label>
              <Button
                size="icon"
                variant="ghost"
                className={cx("h-7 w-7 rounded-xl", isDarkMode ? "hover:bg-white/10" : "hover:bg-black/5")}
                onClick={() => setIsLocked(!isLocked)}
                title={isLocked ? "Unlock seed" : "Lock seed"}
              >
                {isLocked ? <Lock className="w-3.5 h-3.5 text-red-400" /> : <Unlock className={cx("w-3.5 h-3.5", textSecondary)} />}
              </Button>
            </div>

            <div className="flex gap-2">
              <div
                className={cx(
                  "flex-1 border rounded-xl flex items-center px-3 font-mono",
                  isDarkMode ? "bg-black/45 border-white/10 text-green-300" : "bg-gray-100 border-gray-300 text-green-700"
                )}
              >
                {seed || "---"}
              </div>
              <Button
                size="icon"
                className={cx("rounded-xl", softBtn)}
                onClick={randomizeSeed}
                disabled={isLocked}
                title="Randomize seed"
              >
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* ✅ Focus Engine / Binaural (SECOND OPTION, preserved) */}
          <div
            className={cx(
              "p-4 rounded-3xl border space-y-3",
              isDarkMode
                ? "bg-gradient-to-br from-purple-900/20 to-blue-900/20 border-white/10"
                : "bg-gradient-to-br from-purple-50 to-blue-50 border-purple-100"
            )}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-blue-300">
                <Brain className="w-4 h-4" />
                <span className={cx("text-xs font-bold tracking-wider", textPrimary)}>FOCUS ENGINE</span>
              </div>
              <div className={cx("text-[11px] font-mono", textSecondary)}>
                <Headphones className="inline-block w-3.5 h-3.5 mr-1 opacity-80" />
                {binauralLabel}
              </div>
            </div>

            <div className={cx("flex gap-1 p-1 rounded-2xl border", surface)}>
              {(["off", "focus", "relax"] as const).map((mode) => {
                const active = binaural === mode;
                return (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setBinaural(mode)}
                    className={cx(
                      "flex-1 text-[10px] uppercase py-2 rounded-xl transition-all font-bold tracking-wider",
                      active
                        ? "bg-blue-600 text-white shadow-lg"
                        : isDarkMode
                        ? "text-white/75 hover:text-white hover:bg-white/5"
                        : "text-black/60 hover:text-black hover:bg-black/5"
                    )}
                  >
                    {mode}
                  </button>
                );
              })}
            </div>

            <div className={cx("text-[11px]", textTertiary)}>Headphones recommended. Optional enhancement for focus sessions.</div>
          </div>
        </CardContent>
      </Card>

      {/* === MODULE B: STRUCTURE & ARRANGEMENT (CENTER) === */}
      <Card className={cx("lg:col-span-6 backdrop-blur-xl rounded-3xl", bgCard)}>
        <CardContent className="p-6 flex flex-col h-full">
          <div className="flex items-center gap-2 mb-6 text-blue-300">
            <TrendingUp className="w-5 h-5" />
            <h2 className="font-bold text-sm tracking-wider">STRUCTURE & ENERGY</h2>
          </div>

          {/* Interactive Energy Curve Canvas */}
          <div className={cx("flex-1 min-h-[220px] rounded-3xl border p-6 relative overflow-hidden", isDarkMode ? "bg-gray-950/45 border-white/10" : "bg-gray-50 border-black/10")}>
            <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle_at_20%_20%,rgba(59,130,246,0.22),transparent_55%),radial-gradient(circle_at_80%_30%,rgba(147,51,234,0.18),transparent_55%)] pointer-events-none" />

            <div className="flex flex-col h-full justify-between relative z-10">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                <span className={cx("text-xs font-mono", textSecondary)}>ENERGY PROFILE</span>

                <div className="flex flex-wrap gap-2">
                  {[
                    { id: "linear", label: "Loop", icon: Repeat },
                    { id: "drop", label: "Drop", icon: TrendingUp },
                    { id: "peak", label: "Peak", icon: Activity },
                  ].map((curve) => {
                    const active = energyCurve === curve.id;
                    return (
                      <Button
                        key={curve.id}
                        type="button"
                        size="sm"
                        className={cx(
                          "h-8 text-xs gap-1 rounded-2xl border",
                          active
                            ? isDarkMode
                              ? "bg-white/10 text-white border-white/10"
                              : "bg-black/5 text-black border-black/10"
                            : isDarkMode
                            ? "text-white/75 hover:text-white hover:bg-white/5 border-transparent"
                            : "text-black/60 hover:text-black hover:bg-black/5 border-transparent"
                        )}
                        onClick={() => applyPresetCurve(curve.id as any)}
                      >
                        <curve.icon className="w-3 h-3" />
                        {curve.label}
                      </Button>
                    );
                  })}
                </div>
              </div>

              {/* Drawing Area */}
              <div
                className="flex items-end gap-1 h-36 mt-4 cursor-crosshair select-none"
                onMouseLeave={() => (isDrawing.current = false)}
                onMouseUp={() => (isDrawing.current = false)}
              >
                {energyLevels.map((h, i) => (
                  <div
                    key={i}
                    className={cx(
                      "flex-1 rounded-t-sm transition-all duration-75 relative",
                      i % 4 === 0 ? "bg-blue-500" : isDarkMode ? "bg-blue-500/35" : "bg-blue-300/55",
                      "hover:bg-blue-400"
                    )}
                    style={{ height: `${h}%` }}
                    onMouseDown={(e) => {
                      isDrawing.current = true;
                      handleDraw(i, e);
                    }}
                    onMouseEnter={(e) => isDrawing.current && handleDraw(i, e)}
                  />
                ))}
              </div>

              <p className={cx("text-[11px] text-center mt-3", textTertiary)}>
                Draw to customize. (Custom maps to “peak” for backend compatibility.)
              </p>
            </div>
          </div>

          {/* Layer Matrix */}
          <div className="mt-6">
            <label className={cx("text-xs font-bold block mb-3", textSecondary)}>LAYER MATRIX</label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {LAYERS.map((layer) => {
                const active = layers.includes(layer.id);
                return (
                  <button
                    key={layer.id}
                    type="button"
                    onClick={() => toggleLayer(layer.id)}
                    className={cx(
                      "relative p-4 rounded-2xl border flex flex-col items-center gap-2 transition-all",
                      active
                        ? "bg-blue-600/20 border-blue-500/50 text-blue-200"
                        : isDarkMode
                        ? "bg-white/5 border-white/10 text-white/75 hover:bg-white/8 hover:text-white hover:border-white/15"
                        : "bg-white border-black/10 text-black/60 hover:bg-black/5 hover:text-black"
                    )}
                  >
                    <layer.icon className={cx("w-6 h-6", active && "animate-pulse")} />
                    <span className={cx("text-xs font-bold", active ? "text-blue-100" : textPrimary)}>{layer.label}</span>
                    <div
                      className={cx(
                        "absolute top-2 right-2 w-2 h-2 rounded-full",
                        active ? "bg-blue-400 shadow-[0_0_10px_#60a5fa]" : isDarkMode ? "bg-white/20" : "bg-black/20"
                      )}
                    />
                  </button>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* === MODULE C: SMART MIXER (RIGHT) === */}
      <Card className={cx("lg:col-span-3 backdrop-blur-xl rounded-3xl", bgCard)}>
        <CardContent className="p-6 space-y-6">
          <div className="flex items-center gap-2 mb-2 text-pink-300">
            <Sliders className="w-5 h-5" />
            <h2 className="font-bold text-sm tracking-wider">SMART MIXER</h2>
          </div>

          {/* Intensity */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className={cx("text-xs font-bold", textSecondary)}>INTENSITY</label>
              <span className={cx("text-xs font-mono", textSecondary)}>{intensity}</span>
            </div>
            <Slider value={[intensity]} max={100} step={1} onValueChange={([v]) => setIntensity(v)} />
            <div className={cx("text-[11px]", textTertiary)}>Higher = denser grooves + stronger drums.</div>
          </div>

          {/* Synth Params */}
          <div className={cx("grid grid-cols-2 gap-4 pt-4 border-t", isDarkMode ? "border-white/10" : "border-black/10")}>
            <Knob label="FILTER" value={synthParams.cutoff} color="text-pink-400" isDarkMode={isDarkMode} onChange={(v) => setSynthParams((p) => ({ ...p, cutoff: v }))} />
            <Knob label="RES" value={synthParams.resonance} color="text-purple-400" isDarkMode={isDarkMode} onChange={(v) => setSynthParams((p) => ({ ...p, resonance: v }))} />
            <Knob label="DRIVE" value={synthParams.drive} color="text-orange-400" isDarkMode={isDarkMode} onChange={(v) => setSynthParams((p) => ({ ...p, drive: v }))} />
            <Knob label="SPACE" value={synthParams.space} color="text-cyan-400" isDarkMode={isDarkMode} onChange={(v) => setSynthParams((p) => ({ ...p, space: v }))} />
          </div>

          {/* Ambience */}
          <div className={cx("space-y-4 pt-4 border-t", isDarkMode ? "border-white/10" : "border-black/10")}>
            <label className={cx("text-xs font-bold flex items-center gap-2", textSecondary)}>
              <CloudRain className="w-3 h-3" /> AMBIENCE
            </label>

            <div className="space-y-3">
              {[
                { label: "Rain", val: ambience.rain, key: "rain" as const },
                { label: "Vinyl", val: ambience.vinyl, key: "vinyl" as const },
                { label: "White", val: ambience.white, key: "white" as const },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-3">
                  <span className={cx("text-[11px] w-12", textSecondary)}>{item.label}</span>
                  <Slider value={[item.val]} max={100} className="flex-1" onValueChange={([v]) => setAmbience((p) => ({ ...p, [item.key]: v }))} />
                  <span className={cx("text-[11px] font-mono w-10 text-right", textSecondary)}>{Math.round(item.val)}</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* === ACTION BAR (BOTTOM) === */}
      <div className="lg:col-span-12 space-y-3">
        {(isGenerating || genStatus) && (
          <div
            className={cx(
              "rounded-2xl p-3 flex items-center justify-between border animate-in fade-in slide-in-from-bottom-2",
              isDarkMode ? "bg-black/35 border-purple-500/30" : "bg-white border-purple-200"
            )}
          >
            <div className="flex items-center gap-3 min-w-0">
              {isGenerating ? <Loader2 className="w-5 h-5 text-purple-300 animate-spin" /> : <CheckCircle2 className="w-5 h-5 text-green-500" />}
              <span className={cx("text-sm font-mono truncate", isDarkMode ? "text-white/85" : "text-black/70")}>{genStatus}</span>
            </div>
            <div className={cx("h-1 w-32 rounded-full overflow-hidden", isDarkMode ? "bg-white/10" : "bg-black/10")}>
              <div className={cx("h-full", isGenerating ? "bg-purple-500 animate-progress origin-left w-full" : "bg-green-500 w-full")} />
            </div>
          </div>
        )}

        <Button
          className={cx(
            "w-full h-16 text-xl font-bold tracking-widest rounded-3xl shadow-2xl transition-all",
            isGenerating
              ? isDarkMode
                ? "bg-white/10 cursor-not-allowed opacity-70 text-white"
                : "bg-black/10 cursor-not-allowed opacity-70 text-black"
              : "bg-gradient-to-r from-purple-600 via-blue-600 to-purple-600 bg-[length:200%_auto] hover:bg-right text-white"
          )}
          onClick={handleGenerate}
          disabled={isGenerating}
        >
          {isGenerating ? (
            <span className="flex items-center gap-3">Processing...</span>
          ) : (
            <span className="flex items-center gap-3">
              <Zap className="w-6 h-6 fill-white" /> GENERATE SESSION
            </span>
          )}
        </Button>

        {/* ✅ “what was generated” panel */}
        {lastGenerated && lastMeta && (
          <div className={cx("rounded-2xl border p-4", surface)}>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
              <div className="min-w-0">
                <div className={cx("text-[10px] font-bold tracking-widest", textSecondary)}>LAST GENERATED</div>
                <div className={cx("font-bold truncate", textPrimary)}>{lastGenerated.name}</div>

                <div className={cx("text-xs mt-1", textSecondary)}>
                  {lastGenerated.genre} • {lastGenerated.bpm} BPM • Key {lastGenerated.key} • {fmtDuration(lastGenerated.duration)}
                </div>

                <div className={cx("text-[11px] mt-2", textTertiary)}>
                  Focus Engine: <span className={cx("font-mono", textSecondary)}>{lastMeta.binaural}</span> • Layers:{" "}
                  <span className={cx("font-mono", textSecondary)}>{lastMeta.layers.join(", ")}</span> • Curve:{" "}
                  <span className={cx("font-mono", textSecondary)}>{lastMeta.energyCurve}</span> • Intensity:{" "}
                  <span className={cx("font-mono", textSecondary)}>{lastMeta.intensity}</span>
                  {lastGenerated.seed ? (
                    <>
                      {" "}
                      • Seed: <span className={cx("font-mono", textSecondary)}>{lastGenerated.seed}</span>
                    </>
                  ) : null}
                </div>
              </div>

              <audio controls preload="none" src={lastGenerated.url} className="w-full md:w-[360px]" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// HELPER COMPONENT: KNOB
// ============================================================================

function Knob({
  label,
  value,
  onChange,
  color,
  isDarkMode,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  color: string;
  isDarkMode: boolean;
}) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={cx(
          "relative w-16 h-16 flex items-center justify-center rounded-full border shadow-inner group cursor-pointer transition-colors",
          isDarkMode ? "bg-gray-950 border-white/10 hover:border-white/20" : "bg-gray-50 border-gray-300 hover:border-gray-400"
        )}
      >
        <svg className="absolute inset-0 w-full h-full -rotate-90">
          <circle cx="32" cy="32" r="28" stroke="currentColor" strokeWidth="4" className={isDarkMode ? "text-white/10" : "text-black/10"} fill="none" />
          <circle
            cx="32"
            cy="32"
            r="28"
            stroke="currentColor"
            strokeWidth="4"
            className={cx(color, "transition-all duration-100")}
            fill="none"
            strokeDasharray={`${(value / 100) * 175}, 200`}
            strokeLinecap="round"
          />
        </svg>

        <span className={cx("text-xs font-bold font-mono", isDarkMode ? "text-white" : "text-gray-900")}>{Math.round(value)}</span>

        <input
          type="range"
          min="0"
          max="100"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 opacity-0 cursor-ns-resize"
        />
      </div>

      <span className={cx("text-[10px] font-bold tracking-wider", isDarkMode ? "text-white/70" : "text-black/50")}>{label}</span>
    </div>
  );
}
