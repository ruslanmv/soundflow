"use client";

import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Copy, Download, Plus, Trash2, Wand2, PlayCircle } from "lucide-react";

/**
 * SoundFlow Combination UI
 *
 * Goal: Provide an interface to explore and save ALL possible combinations (or a filtered subset)
 * of music-generation parameters. Later, an AI can use the same schema to pick best combinations.
 *
 * This UI is frontend-only: "Generate" just emits a JSON payload you can POST to your generator.
 */

// ------------------------------
// Schema (editable)
// ------------------------------

const GENRES = [
  "Trance",
  "House",
  "Lounge",
  "Techno",
  "Deep",
  "EDM",
  "Chillout",
  "Bass",
  "Dance",
  "Vocal",
  "Hard",
  "Ambient",
  "Synth",
  "Classic",
];

const SCALES = ["major", "minor", "dorian", "phrygian", "lydian", "pentatonic"];

const KEYS = [
  { name: "C", freq: 261.63 },
  { name: "D", freq: 293.66 },
  { name: "E", freq: 329.63 },
  { name: "F", freq: 349.23 },
  { name: "G", freq: 392.0 },
  { name: "A", freq: 440.0 },
  { name: "B", freq: 493.88 },
];

// Layer archetypes map to your stems / generator blocks
const LAYERS = [
  { id: "drums", label: "Drums" },
  { id: "bass", label: "Bass" },
  { id: "music", label: "Music" },
  { id: "pad", label: "Pad" },
  { id: "texture", label: "Texture" },
  { id: "ambience", label: "Ambience" },
];

const TEXTURES = ["none", "vinyl", "rain", "tape", "room"];

const INSTRUMENT_MODES = ["hybrid", "pluck", "supersaw", "piano", "vocal"];

// Defaults per genre (you can align with your generator)
const GENRE_DEFAULTS: Record<
  string,
  { bpm: number; scale: string; instrument: string; texture: string }
> = {
  Trance: { bpm: 128, scale: "phrygian", instrument: "hybrid", texture: "none" },
  House: { bpm: 124, scale: "pentatonic", instrument: "pluck", texture: "none" },
  Lounge: { bpm: 92, scale: "dorian", instrument: "piano", texture: "vinyl" },
  Techno: { bpm: 132, scale: "phrygian", instrument: "supersaw", texture: "none" },
  Deep: { bpm: 122, scale: "pentatonic", instrument: "hybrid", texture: "none" },
  EDM: { bpm: 128, scale: "major", instrument: "supersaw", texture: "none" },
  Chillout: { bpm: 88, scale: "pentatonic", instrument: "pluck", texture: "vinyl" },
  Bass: { bpm: 174, scale: "minor", instrument: "hybrid", texture: "none" },
  Dance: { bpm: 128, scale: "major", instrument: "hybrid", texture: "none" },
  Vocal: { bpm: 124, scale: "major", instrument: "vocal", texture: "none" },
  Hard: { bpm: 150, scale: "phrygian", instrument: "supersaw", texture: "none" },
  Ambient: { bpm: 60, scale: "lydian", instrument: "hybrid", texture: "rain" },
  Synth: { bpm: 100, scale: "lydian", instrument: "hybrid", texture: "tape" },
  Classic: { bpm: 72, scale: "major", instrument: "piano", texture: "room" },
};

// ------------------------------
// Helpers
// ------------------------------

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function asJSON(obj: any) {
  return JSON.stringify(obj, null, 2);
}

function estimateCount(opts: {
  genres: string[];
  scales: string[];
  keys: number;
  instruments: string[];
  textures: string[];
  includeLayers: string[];
}) {
  // Basic upper-bound count for combos.
  // Layers are treated as "on/off" switches per selected layer.
  const layerCombos = Math.max(1, 2 ** opts.includeLayers.length);
  return (
    Math.max(1, opts.genres.length) *
    Math.max(1, opts.scales.length) *
    Math.max(1, opts.keys) *
    Math.max(1, opts.instruments.length) *
    Math.max(1, opts.textures.length) *
    layerCombos
  );
}

// ------------------------------
// Main UI
// ------------------------------

export default function SoundFlowCombinationUI() {
  const [presetName, setPresetName] = useState("Daily Generator");

  // Selection
  const [selectedGenres, setSelectedGenres] = useState<string[]>(["Trance"]);
  const [selectedScales, setSelectedScales] = useState<string[]>(["phrygian"]);
  const [selectedKeys, setSelectedKeys] = useState(KEYS.map((k) => k.name));
  const [selectedInstruments, setSelectedInstruments] = useState<string[]>(["hybrid"]);
  const [selectedTextures, setSelectedTextures] = useState<string[]>(["none"]);

  const [layerToggles, setLayerToggles] = useState<Record<string, boolean>>({
    drums: true,
    bass: true,
    music: true,
    pad: true,
    texture: false,
    ambience: false,
  });

  // Mix/style controls (for a *combination* payload)
  const [durationSec, setDurationSec] = useState(120);
  const [humanizeMs, setHumanizeMs] = useState(10);
  const [targetLufs, setTargetLufs] = useState(-14);

  // Nature/noise controls
  const [noiseEnabled, setNoiseEnabled] = useState(true);
  const [noiseGainDb, setNoiseGainDb] = useState(-16);
  const [noiseLP, setNoiseLP] = useState(6000);
  const [noiseHP, setNoiseHP] = useState(120);

  // Generator limits
  const [maxCombos, setMaxCombos] = useState(250);
  const [includeLayerCombos, setIncludeLayerCombos] = useState(true);

  // Saved combos
  const [saved, setSaved] = useState<any[]>([]);
  const [activeJson, setActiveJson] = useState<string>("");

  // Server Interaction State
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [serverLog, setServerLog] = useState("");

  // Derived
  const selectedKeyFreqs = useMemo(() => {
    const names = new Set(selectedKeys);
    return KEYS.filter((k) => names.has(k.name)).map((k) => ({ name: k.name, freq: k.freq }));
  }, [selectedKeys]);

  const layerList = useMemo(() => {
    return LAYERS.filter((l) => layerToggles[l.id]).map((l) => l.id);
  }, [layerToggles]);

  const approxCount = useMemo(() => {
    return estimateCount({
      genres: selectedGenres,
      scales: selectedScales,
      keys: selectedKeyFreqs.length,
      instruments: selectedInstruments,
      textures: selectedTextures,
      includeLayers: includeLayerCombos ? layerList : [],
    });
  }, [selectedGenres, selectedScales, selectedKeyFreqs.length, selectedInstruments, selectedTextures, layerList, includeLayerCombos]);

  function toggleInList<T extends string>(value: T, list: T[], setList: (v: T[]) => void) {
    setList(list.includes(value) ? list.filter((x) => x !== value) : [...list, value]);
  }

  function applyGenreDefaults(genre: string) {
    const d = GENRE_DEFAULTS[genre];
    if (!d) return;
    // Snap selections to defaults for a clean start
    setSelectedGenres([genre]);
    setSelectedScales([d.scale]);
    setSelectedInstruments([d.instrument]);
    setSelectedTextures([d.texture]);
  }

  function buildOneCombo(seed: string) {
    // Build a single combo payload from current selections (picks first item from each list).
    const genre = selectedGenres[0] ?? "Trance";
    const defaults = GENRE_DEFAULTS[genre] ?? GENRE_DEFAULTS.Trance;

    const scale = selectedScales[0] ?? defaults.scale;
    const instrument = selectedInstruments[0] ?? defaults.instrument;
    const texture = selectedTextures[0] ?? defaults.texture;

    const key = selectedKeyFreqs[0] ?? { name: "A", freq: 440.0 };

    const payload = {
      id: `combo-${uid()}`,
      name: presetName,
      seed,
      genre,
      bpm: defaults.bpm,
      key: key.name,
      key_freq: key.freq,
      scale,
      instrument,
      texture,
      layers: {
        enabled: layerList,
      },
      mix: {
        duration_sec: durationSec,
        humanize_ms: humanizeMs,
        target_lufs: targetLufs,
        noise: {
          enabled: noiseEnabled,
          gain_db: noiseGainDb,
          lowpass_hz: noiseLP,
          highpass_hz: noiseHP,
        },
      },
      // future: model scoring hooks
      metadata: {
        created_at: new Date().toISOString(),
        ui_version: "1.0.0",
      },
    };

    return payload;
  }

  // --- SERVER ACTION ---
  async function runOnServer() {
    setIsGenerating(true);
    setServerLog("Sending request to generator engine...");
    setAudioUrl(null);

    try {
      // 1. Construct the payload (Single combo based on current selection)
      const combo = buildOneCombo(`manual:${Date.now()}`);
      
      // 2. Call the API
      // Ensure your backend is running on port 8000 (python3 server.py)
      const res = await fetch("http://localhost:8000/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(combo),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Server error (${res.status}): ${errText}`);
      }

      const data = await res.json();
      
      setServerLog(`✅ Success! Generated: ${data.filename}`);
      setAudioUrl(data.url); // This URL points to the python static file server

    } catch (err: any) {
      console.error(err);
      setServerLog(`❌ Error: ${err.message || String(err)}`);
    } finally {
      setIsGenerating(false);
    }
  }

  function generateCombos() {
    // Create up to maxCombos combinations by cartesian product, but capped.
    const genres = selectedGenres.length ? selectedGenres : ["Trance"];
    const scales = selectedScales.length ? selectedScales : [GENRE_DEFAULTS.Trance.scale];
    const keys = selectedKeyFreqs.length ? selectedKeyFreqs : [{ name: "A", freq: 440.0 }];
    const instruments = selectedInstruments.length ? selectedInstruments : [GENRE_DEFAULTS.Trance.instrument];
    const textures = selectedTextures.length ? selectedTextures : [GENRE_DEFAULTS.Trance.texture];

    // layer combos
    const enabledLayers = layerList;
    const layerCombos: string[][] = [];
    if (!includeLayerCombos || enabledLayers.length === 0) {
      layerCombos.push(enabledLayers);
    } else {
      // generate on/off subsets (excluding empty)
      const n = enabledLayers.length;
      for (let mask = 1; mask < (1 << n); mask++) {
        const subset: string[] = [];
        for (let i = 0; i < n; i++) {
          if (mask & (1 << i)) subset.push(enabledLayers[i]);
        }
        layerCombos.push(subset);
      }
    }

    const out: any[] = [];
    let count = 0;

    outer: for (const genre of genres) {
      const d = GENRE_DEFAULTS[genre] ?? GENRE_DEFAULTS.Trance;
      for (const scale of scales) {
        for (const key of keys) {
          for (const instrument of instruments) {
            for (const texture of textures) {
              for (const layers of layerCombos) {
                const payload = {
                  id: `combo-${uid()}`,
                  name: presetName,
                  seed: `${genre}:${scale}:${key.name}:${instrument}:${texture}:${layers.join("+")}`,
                  genre,
                  bpm: d.bpm,
                  key: key.name,
                  key_freq: key.freq,
                  scale,
                  instrument,
                  texture,
                  layers: { enabled: layers },
                  mix: {
                    duration_sec: durationSec,
                    humanize_ms: humanizeMs,
                    target_lufs: targetLufs,
                    noise: {
                      enabled: noiseEnabled,
                      gain_db: noiseGainDb,
                      lowpass_hz: noiseLP,
                      highpass_hz: noiseHP,
                    },
                  },
                  metadata: {
                    created_at: new Date().toISOString(),
                    ui_version: "1.0.0",
                  },
                };

                out.push(payload);
                count++;
                if (count >= maxCombos) break outer;
              }
            }
          }
        }
      }
    }

    setActiveJson(asJSON({ schema: "soundflow.combo.v1", combinations: out }));
  }

  function saveActiveAsSet() {
    if (!activeJson) return;
    try {
      const parsed = JSON.parse(activeJson);
      const set = {
        id: `set-${uid()}`,
        title: presetName,
        created_at: new Date().toISOString(),
        payload: parsed,
      };
      setSaved((s) => [set, ...s]);
    } catch {
      // ignore
    }
  }

  async function copyToClipboard(text: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // ignore
    }
  }

  function downloadJson(filename: string, text: string) {
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="p-4 md:p-8 space-y-6">
      {/* HEADER BAR */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">SoundFlow Combo Builder</h1>
          <p className="text-sm text-muted-foreground">
            Pick parameters → generate combinations → or run the engine instantly.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Input
            className="w-[200px]"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            placeholder="Preset name"
          />
          <Button
            variant="secondary"
            onClick={() => {
              const one = buildOneCombo(`manual:${new Date().toISOString()}`);
              setActiveJson(asJSON({ schema: "soundflow.combo.v1", combinations: [one] }));
            }}
          >
            <Wand2 className="w-4 h-4 mr-2" />
            One combo
          </Button>
          <Button onClick={generateCombos}>
            <Plus className="w-4 h-4 mr-2" />
            Add to List
          </Button>
          
          {/* RUN ENGINE BUTTON */}
          <Button onClick={runOnServer} disabled={isGenerating} className="bg-green-600 hover:bg-green-700 text-white">
            {isGenerating ? (
              <Wand2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <PlayCircle className="w-4 h-4 mr-2" />
            )}
            {isGenerating ? "Cooking..." : "Run Engine"}
          </Button>
        </div>
      </div>

      {/* AUDIO PLAYER RESULT */}
      {(serverLog || audioUrl) && (
        <Card className="border-blue-200 bg-blue-50/50 shadow-sm animate-in fade-in slide-in-from-top-2">
          <CardContent className="p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
            <div className="font-mono text-sm text-blue-900">{serverLog}</div>
            {audioUrl && (
              <div className="flex items-center gap-3 w-full md:w-auto">
                <audio controls autoPlay className="w-full md:w-[400px] h-10 rounded-full shadow-sm">
                  <source src={audioUrl} type="audio/mpeg" />
                  Your browser does not support the audio element.
                </audio>
                <Button size="icon" variant="ghost" asChild title="Download MP3">
                  <a href={audioUrl} download>
                    <Download className="w-4 h-4" />
                  </a>
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Controls */}
        <Card className="lg:col-span-2 rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Combination Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <Tabs defaultValue="core" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="core">Core</TabsTrigger>
                <TabsTrigger value="layers">Layers</TabsTrigger>
                <TabsTrigger value="mix">Mix / Noise</TabsTrigger>
              </TabsList>

              <TabsContent value="core" className="space-y-5 mt-4">
                <Section title="Genres">
                  <div className="flex flex-wrap gap-2">
                    {GENRES.map((g) => {
                      const active = selectedGenres.includes(g);
                      return (
                        <Button
                          key={g}
                          variant={active ? "default" : "secondary"}
                          className="rounded-2xl"
                          onClick={() => toggleInList(g, selectedGenres, setSelectedGenres)}
                          onDoubleClick={() => applyGenreDefaults(g)}
                          title="Click to toggle; double-click to apply defaults"
                        >
                          {g}
                        </Button>
                      );
                    })}
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Tip: double-click a genre to snap scale/instrument/texture to recommended defaults.
                  </p>
                </Section>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Section title="Scales">
                    <div className="flex flex-wrap gap-2">
                      {SCALES.map((s) => {
                        const active = selectedScales.includes(s);
                        return (
                          <Button
                            key={s}
                            variant={active ? "default" : "secondary"}
                            className="rounded-2xl"
                            onClick={() => toggleInList(s, selectedScales, setSelectedScales)}
                          >
                            {s}
                          </Button>
                        );
                      })}
                    </div>
                  </Section>

                  <Section title="Instrument modes">
                    <div className="flex flex-wrap gap-2">
                      {INSTRUMENT_MODES.map((m) => {
                        const active = selectedInstruments.includes(m);
                        return (
                          <Button
                            key={m}
                            variant={active ? "default" : "secondary"}
                            className="rounded-2xl"
                            onClick={() => toggleInList(m, selectedInstruments, setSelectedInstruments)}
                          >
                            {m}
                          </Button>
                        );
                      })}
                    </div>
                  </Section>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Section title="Keys">
                    <div className="flex flex-wrap gap-2">
                      {KEYS.map((k) => {
                        const active = selectedKeys.includes(k.name);
                        return (
                          <Button
                            key={k.name}
                            variant={active ? "default" : "secondary"}
                            className="rounded-2xl"
                            onClick={() => toggleInList(k.name, selectedKeys, setSelectedKeys)}
                          >
                            {k.name}
                          </Button>
                        );
                      })}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Keys map to base frequencies (e.g., A=440 Hz). Update mapping in code if you want different octaves.
                    </p>
                  </Section>

                  <Section title="Textures">
                    <div className="flex flex-wrap gap-2">
                      {TEXTURES.map((t) => {
                        const active = selectedTextures.includes(t);
                        return (
                          <Button
                            key={t}
                            variant={active ? "default" : "secondary"}
                            className="rounded-2xl"
                            onClick={() => toggleInList(t, selectedTextures, setSelectedTextures)}
                          >
                            {t}
                          </Button>
                        );
                      })}
                    </div>
                  </Section>
                </div>
              </TabsContent>

              <TabsContent value="layers" className="space-y-5 mt-4">
                <Section title="Enabled layers">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {LAYERS.map((l) => (
                      <div key={l.id} className="flex items-center justify-between rounded-2xl border p-3">
                        <div className="space-y-0.5">
                          <div className="font-medium">{l.label}</div>
                          <div className="text-xs text-muted-foreground">id: {l.id}</div>
                        </div>
                        <Switch
                          checked={!!layerToggles[l.id]}
                          onCheckedChange={(v) => setLayerToggles((s) => ({ ...s, [l.id]: !!v }))}
                        />
                      </div>
                    ))}
                  </div>
                  <div className="flex items-center justify-between mt-4 rounded-2xl border p-3">
                    <div>
                      <div className="font-medium">Include layer combinations</div>
                      <div className="text-xs text-muted-foreground">If enabled, generates subsets of enabled layers (capped by Max combos).</div>
                    </div>
                    <Switch checked={includeLayerCombos} onCheckedChange={(v) => setIncludeLayerCombos(!!v)} />
                  </div>
                </Section>

                <Section title="Limits">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium">Max combinations</div>
                        <Badge variant="secondary">{maxCombos}</Badge>
                      </div>
                      <Slider
                        value={[maxCombos]}
                        min={25}
                        max={2000}
                        step={25}
                        onValueChange={(v) => setMaxCombos(v[0] ?? 250)}
                      />
                      <p className="text-xs text-muted-foreground">Safety cap to avoid generating huge payloads in the browser.</p>
                    </div>

                    <div className="space-y-2">
                      <div className="text-sm font-medium">Estimated combos (upper bound)</div>
                      <div className="text-2xl font-semibold tracking-tight">{approxCount.toLocaleString()}</div>
                      <p className="text-xs text-muted-foreground">
                        If estimated &gt; Max, generation will stop at Max. Use AI later to rank/select.
                      </p>
                    </div>
                  </div>
                </Section>
              </TabsContent>

              <TabsContent value="mix" className="space-y-5 mt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Section title="Duration">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium">Seconds</div>
                        <Badge variant="secondary">{durationSec}s</Badge>
                      </div>
                      <Slider
                        value={[durationSec]}
                        min={30}
                        max={3600}
                        step={30}
                        onValueChange={(v) => setDurationSec(v[0] ?? 120)}
                      />
                    </div>
                  </Section>

                  <Section title="Humanize">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium">Timing jitter</div>
                        <Badge variant="secondary">{humanizeMs}ms</Badge>
                      </div>
                      <Slider
                        value={[humanizeMs]}
                        min={0}
                        max={30}
                        step={1}
                        onValueChange={(v) => setHumanizeMs(v[0] ?? 10)}
                      />
                      <p className="text-xs text-muted-foreground">Higher values feel more human; too high can sound sloppy.</p>
                    </div>
                  </Section>
                </div>

                <Section title="Loudness">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-medium">Target LUFS</div>
                      <Badge variant="secondary">{targetLufs}</Badge>
                    </div>
                    <Slider
                      value={[targetLufs]}
                      min={-20}
                      max={-10}
                      step={1}
                      onValueChange={(v) => setTargetLufs(v[0] ?? -14)}
                    />
                  </div>
                </Section>

                <Section title="Noise / Nature Bed">
                  <div className="flex items-center justify-between rounded-2xl border p-3 mb-3">
                    <div>
                      <div className="font-medium">Enable noise layer</div>
                      <div className="text-xs text-muted-foreground">Used for vinyl/rain/room textures and ambience beds.</div>
                    </div>
                    <Switch checked={noiseEnabled} onCheckedChange={(v) => setNoiseEnabled(!!v)} />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium">Noise gain</div>
                        <Badge variant="secondary">{noiseGainDb} dB</Badge>
                      </div>
                      <Slider
                        value={[noiseGainDb]}
                        min={-30}
                        max={-6}
                        step={1}
                        onValueChange={(v) => setNoiseGainDb(v[0] ?? -16)}
                      />
                      <p className="text-xs text-muted-foreground">Tip: -16 to -22 dB usually keeps noise subtle.</p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium">Highpass</div>
                        <Badge variant="secondary">{noiseHP} Hz</Badge>
                      </div>
                      <Slider
                        value={[noiseHP]}
                        min={20}
                        max={400}
                        step={10}
                        onValueChange={(v) => setNoiseHP(v[0] ?? 120)}
                      />
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium">Lowpass</div>
                        <Badge variant="secondary">{noiseLP} Hz</Badge>
                      </div>
                      <Slider
                        value={[noiseLP]}
                        min={2000}
                        max={12000}
                        step={250}
                        onValueChange={(v) => setNoiseLP(v[0] ?? 6000)}
                      />
                    </div>
                  </div>
                </Section>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Right: Output */}
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Output</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-2xl border p-3">
              <div className="text-xs text-muted-foreground">Current selection</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {selectedGenres.slice(0, 3).map((g) => (
                  <Badge key={g} variant="secondary">{g}</Badge>
                ))}
                {selectedGenres.length > 3 && <Badge variant="secondary">+{selectedGenres.length - 3} genres</Badge>}
                <Separator orientation="vertical" className="h-5" />
                <Badge variant="secondary">{durationSec}s</Badge>
                <Badge variant="secondary">{targetLufs} LUFS</Badge>
              </div>
            </div>

            <div className="flex gap-2">
              <Button variant="secondary" className="w-full" onClick={saveActiveAsSet} disabled={!activeJson}>
                <Plus className="w-4 h-4 mr-2" />
                Save
              </Button>
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="secondary" className="w-full" disabled={!activeJson}>
                    <Copy className="w-4 h-4 mr-2" />
                    Copy
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-2xl">
                  <DialogHeader>
                    <DialogTitle>JSON Payload</DialogTitle>
                  </DialogHeader>
                  <Textarea value={activeJson} readOnly className="h-[360px] font-mono text-xs" />
                  <div className="flex gap-2">
                    <Button className="w-full" onClick={() => copyToClipboard(activeJson)}>
                      <Copy className="w-4 h-4 mr-2" />
                      Copy to clipboard
                    </Button>
                    <Button
                      variant="secondary"
                      className="w-full"
                      onClick={() => downloadJson(`soundflow-combos-${Date.now()}.json`, activeJson)}
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>

            <Separator />

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="font-medium">Saved sets</div>
                <Badge variant="secondary">{saved.length}</Badge>
              </div>
              <ScrollArea className="h-[360px] rounded-2xl border p-2">
                <div className="space-y-2">
                  {saved.length === 0 && (
                    <div className="text-sm text-muted-foreground p-3">No saved sets yet. Generate → Save.</div>
                  )}
                  {saved.map((s) => (
                    <div key={s.id} className="rounded-2xl border p-3">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="font-medium">{s.title}</div>
                          <div className="text-xs text-muted-foreground">{new Date(s.created_at).toLocaleString()}</div>
                        </div>
                        <div className="flex gap-2">
                          <Button
                            size="icon"
                            variant="secondary"
                            onClick={() => {
                              const txt = asJSON(s.payload);
                              setActiveJson(txt);
                            }}
                            title="Load into Output"
                          >
                            <Wand2 className="w-4 h-4" />
                          </Button>
                          <Button
                            size="icon"
                            variant="secondary"
                            onClick={() => downloadJson(`${s.title.replace(/\s+/g, "-").toLowerCase()}-${Date.now()}.json`, asJSON(s.payload))}
                            title="Download"
                          >
                            <Download className="w-4 h-4" />
                          </Button>
                          <Button
                            size="icon"
                            variant="destructive"
                            onClick={() => setSaved((prev) => prev.filter((x) => x.id !== s.id))}
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                      <div className="mt-2 text-xs text-muted-foreground">
                        {(() => {
                          const combos = s.payload?.combinations?.length ?? 0;
                          return `${combos} combinations`;
                        })()}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Integration Notes</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-2">
          <p>
            Your backend generator should accept the payload schema <span className="font-mono">soundflow.combo.v1</span>.
            Each entry in <span className="font-mono">combinations</span> is a single recipe that an AI can later score.
          </p>
          <p>
            Suggested flow: UI generates a large set → AI ranks based on authenticity → backend renders only top-N.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="text-sm font-semibold">{title}</div>
      <div>{children}</div>
    </div>
  );
}