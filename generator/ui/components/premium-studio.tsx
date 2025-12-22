"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Sparkles,
  Zap,
  Brain,
  Waves,
  Volume2,
  Music2,
  Download,
  PlayCircle,
  Loader2,
  CheckCircle2,
  Activity,
  Headphones,
} from "lucide-react";

// ============================================================================
// TYPES & CONFIGURATION
// ============================================================================

const API_BASE = "http://localhost:8000";

interface PremiumTrack {
  id: string;
  title: string;
  url: string;
  duration: number;
  category: string;
  dsp_flags?: {
    binaural?: boolean;
    binaural_freq?: number;
    binaural_base?: number;
    sidechain?: boolean;
    bpm?: number;
    ducking_strength?: number;
  };
}

interface GenerationParams {
  prompt: string;
  duration: number;
  category: string;
  bpm: number;
  key: string;

  // DSP Options
  enable_binaural: boolean;
  binaural_type: "gamma" | "beta" | "alpha" | "theta" | "delta";
  binaural_mix: number;

  enable_sidechain: boolean;
  sidechain_strength: number;

  // Quality
  export_bitrate: "192k" | "256k" | "320k";
  target_lufs: number;
}

const BINAURAL_PRESETS = {
  gamma: { freq: 40, label: "Gamma (Focus)", base: 200, desc: "Peak concentration" },
  beta: { freq: 20, label: "Beta (Alert)", base: 180, desc: "Active thinking" },
  alpha: { freq: 10, label: "Alpha (Calm)", base: 180, desc: "Relaxed focus" },
  theta: { freq: 6, label: "Theta (Meditate)", base: 150, desc: "Deep meditation" },
  delta: { freq: 3, label: "Delta (Sleep)", base: 100, desc: "Deep sleep" },
};

const CATEGORY_PRESETS = [
  { id: "deep_work", label: "Deep Work", icon: Brain, prompt: "ambient electronic focus music, minimal beats, warm pads, meditative" },
  { id: "meditation", label: "Meditation", icon: Sparkles, prompt: "deep ambient meditation music, slowly evolving drones, ethereal atmosphere" },
  { id: "energy", label: "Energy", icon: Zap, prompt: "driving techno music, powerful kick drum, dark atmospheric synths, high energy" },
  { id: "flow", label: "Flow", icon: Waves, prompt: "lo-fi house music, deep bassline, dusty drums, jazzy atmosphere, groovy" },
  { id: "study", label: "Study", icon: Headphones, prompt: "calm lo-fi beats, soft piano, vinyl texture, peaceful atmosphere" },
];

const KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const MODES = ["major", "minor"];

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function PremiumStudio() {
  const [params, setParams] = useState<GenerationParams>({
    prompt: CATEGORY_PRESETS[0].prompt,
    duration: 180,
    category: CATEGORY_PRESETS[0].id,
    bpm: 120,
    key: "C minor",

    enable_binaural: false,
    binaural_type: "gamma",
    binaural_mix: 0.12,

    enable_sidechain: false,
    sidechain_strength: 0.7,

    export_bitrate: "320k",
    target_lufs: -14.0,
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTrack, setCurrentTrack] = useState<PremiumTrack | null>(null);
  const [audioPlayer, setAudioPlayer] = useState<HTMLAudioElement | null>(null);

  // -------------------------------------------------------------------------
  // HANDLERS
  // -------------------------------------------------------------------------

  const handleCategorySelect = (preset: typeof CATEGORY_PRESETS[0]) => {
    setParams({
      ...params,
      category: preset.id,
      prompt: preset.prompt,
    });
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    setProgress(0);

    // Build DSP flags
    const dsp_flags: any = {
      fade_in_ms: 1500,
      fade_out_ms: 3000,
    };

    if (params.enable_binaural) {
      const preset = BINAURAL_PRESETS[params.binaural_type];
      dsp_flags.binaural = true;
      dsp_flags.binaural_freq = preset.freq;
      dsp_flags.binaural_base = preset.base;
      dsp_flags.binaural_mix = params.binaural_mix;
    }

    if (params.enable_sidechain) {
      dsp_flags.sidechain = true;
      dsp_flags.bpm = params.bpm;
      dsp_flags.ducking_strength = params.sidechain_strength;
    }

    try {
      // Simulate progress (in real implementation, use WebSocket or polling)
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 5, 90));
      }, 1000);

      const response = await fetch(`${API_BASE}/api/generate/premium`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: params.prompt,
          duration: params.duration,
          category: params.category,
          bpm: params.bpm,
          key: params.key,
          dsp_flags,
          export_bitrate: params.export_bitrate,
          target_lufs: params.target_lufs,
        }),
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        throw new Error(`Generation failed: ${response.statusText}`);
      }

      const result = await response.json();
      setProgress(100);

      const track: PremiumTrack = {
        id: result.id,
        title: result.title,
        url: result.url,
        duration: params.duration,
        category: params.category,
        dsp_flags,
      };

      setCurrentTrack(track);
    } catch (error) {
      console.error("Generation error:", error);
      alert(`Generation failed: ${error}`);
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  };

  const handlePlay = () => {
    if (!currentTrack) return;

    if (audioPlayer) {
      audioPlayer.pause();
      setAudioPlayer(null);
    } else {
      const audio = new Audio(currentTrack.url);
      audio.play();
      setAudioPlayer(audio);
    }
  };

  const handleDownload = () => {
    if (!currentTrack) return;
    window.open(currentTrack.url, "_blank");
  };

  // -------------------------------------------------------------------------
  // RENDER
  // -------------------------------------------------------------------------

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-2">
          <Sparkles className="w-8 h-8 text-purple-500" />
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
            Premium AI Music Studio
          </h1>
        </div>
        <p className="text-muted-foreground">
          Enterprise-grade music generation with neuro-symbolic DSP enhancements
        </p>
        <div className="flex items-center justify-center gap-2 text-sm">
          <Badge variant="secondary">MusicGen Stereo Large</Badge>
          <Badge variant="secondary">320kbps</Badge>
          <Badge variant="secondary">-14 LUFS</Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel: Category & Prompt */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Music2 className="w-5 h-5" />
              Music Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Category Presets */}
            <div>
              <label className="text-sm font-medium mb-2 block">Category Preset</label>
              <div className="grid grid-cols-5 gap-2">
                {CATEGORY_PRESETS.map((preset) => {
                  const Icon = preset.icon;
                  return (
                    <Button
                      key={preset.id}
                      variant={params.category === preset.id ? "default" : "outline"}
                      onClick={() => handleCategorySelect(preset)}
                      className="flex flex-col h-auto py-3"
                    >
                      <Icon className="w-5 h-5 mb-1" />
                      <span className="text-xs">{preset.label}</span>
                    </Button>
                  );
                })}
              </div>
            </div>

            {/* Prompt */}
            <div>
              <label className="text-sm font-medium mb-2 block">Music Description</label>
              <Textarea
                value={params.prompt}
                onChange={(e) => setParams({ ...params, prompt: e.target.value })}
                rows={4}
                placeholder="Describe your desired music..."
                className="font-mono text-sm"
              />
            </div>

            {/* Musical Parameters */}
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Duration (sec)</label>
                <Input
                  type="number"
                  value={params.duration}
                  onChange={(e) => setParams({ ...params, duration: parseInt(e.target.value) })}
                  min={30}
                  max={900}
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">BPM</label>
                <Input
                  type="number"
                  value={params.bpm}
                  onChange={(e) => setParams({ ...params, bpm: parseInt(e.target.value) })}
                  min={60}
                  max={180}
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Key</label>
                <div className="flex gap-2">
                  <select
                    value={params.key.split(" ")[0]}
                    onChange={(e) => setParams({ ...params, key: `${e.target.value} ${params.key.split(" ")[1]}` })}
                    className="flex-1 px-3 py-2 border rounded-md"
                  >
                    {KEYS.map((k) => <option key={k} value={k}>{k}</option>)}
                  </select>
                  <select
                    value={params.key.split(" ")[1]}
                    onChange={(e) => setParams({ ...params, key: `${params.key.split(" ")[0]} ${e.target.value}` })}
                    className="flex-1 px-3 py-2 border rounded-md"
                  >
                    {MODES.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Right Panel: DSP Enhancements */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Waves className="w-5 h-5" />
              DSP Enhancements
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Binaural Beats */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-purple-500" />
                  <span className="font-medium">Binaural Beats</span>
                </div>
                <Switch
                  checked={params.enable_binaural}
                  onCheckedChange={(checked) => setParams({ ...params, enable_binaural: checked })}
                />
              </div>

              {params.enable_binaural && (
                <>
                  <div>
                    <label className="text-xs text-muted-foreground mb-2 block">Wave Type</label>
                    <div className="grid grid-cols-1 gap-1">
                      {Object.entries(BINAURAL_PRESETS).map(([key, preset]) => (
                        <Button
                          key={key}
                          variant={params.binaural_type === key ? "default" : "outline"}
                          size="sm"
                          onClick={() => setParams({ ...params, binaural_type: key as any })}
                          className="justify-start text-xs"
                        >
                          <span className="font-medium">{preset.label}</span>
                          <span className="ml-auto text-muted-foreground">{preset.freq}Hz</span>
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="text-xs text-muted-foreground mb-2 block">
                      Mix Level: {(params.binaural_mix * 100).toFixed(0)}%
                    </label>
                    <Slider
                      value={[params.binaural_mix * 100]}
                      onValueChange={([v]) => setParams({ ...params, binaural_mix: v / 100 })}
                      min={5}
                      max={25}
                      step={1}
                    />
                  </div>
                </>
              )}
            </div>

            {/* Sidechain */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-blue-500" />
                  <span className="font-medium">Sidechain Pump</span>
                </div>
                <Switch
                  checked={params.enable_sidechain}
                  onCheckedChange={(checked) => setParams({ ...params, enable_sidechain: checked })}
                />
              </div>

              {params.enable_sidechain && (
                <div>
                  <label className="text-xs text-muted-foreground mb-2 block">
                    Intensity: {(params.sidechain_strength * 100).toFixed(0)}%
                  </label>
                  <Slider
                    value={[params.sidechain_strength * 100]}
                    onValueChange={([v]) => setParams({ ...params, sidechain_strength: v / 100 })}
                    min={30}
                    max={90}
                    step={5}
                  />
                </div>
              )}
            </div>

            {/* Quality */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Volume2 className="w-4 h-4 text-green-500" />
                <span className="font-medium">Export Quality</span>
              </div>

              <div>
                <label className="text-xs text-muted-foreground mb-2 block">Bitrate</label>
                <div className="grid grid-cols-3 gap-2">
                  {["192k", "256k", "320k"].map((br) => (
                    <Button
                      key={br}
                      variant={params.export_bitrate === br ? "default" : "outline"}
                      size="sm"
                      onClick={() => setParams({ ...params, export_bitrate: br as any })}
                    >
                      {br}
                    </Button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs text-muted-foreground mb-2 block">
                  Loudness: {params.target_lufs} LUFS
                </label>
                <Slider
                  value={[Math.abs(params.target_lufs)]}
                  onValueChange={([v]) => setParams({ ...params, target_lufs: -v })}
                  min={10}
                  max={18}
                  step={1}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Generate Button */}
      <Card>
        <CardContent className="pt-6">
          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !params.prompt}
            size="lg"
            className="w-full h-14 text-lg"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Generating Premium Track... {progress}%
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Generate Premium Music
              </>
            )}
          </Button>

          {isGenerating && (
            <div className="mt-4">
              <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 to-blue-500 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Result */}
      {currentTrack && (
        <Card className="border-green-500/50 bg-green-500/5">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="w-6 h-6 text-green-500" />
                <div>
                  <h3 className="font-semibold">{currentTrack.title}</h3>
                  <p className="text-sm text-muted-foreground">
                    {currentTrack.duration}s • {currentTrack.category} • 320kbps MP3
                  </p>
                </div>
              </div>

              <div className="flex gap-2">
                <Button onClick={handlePlay} variant="outline">
                  <PlayCircle className="w-4 h-4 mr-2" />
                  {audioPlayer ? "Pause" : "Play"}
                </Button>
                <Button onClick={handleDownload}>
                  <Download className="w-4 h-4 mr-2" />
                  Download
                </Button>
              </div>
            </div>

            {currentTrack.dsp_flags && (
              <div className="mt-4 flex gap-2">
                {currentTrack.dsp_flags.binaural && (
                  <Badge variant="secondary">
                    <Brain className="w-3 h-3 mr-1" />
                    Binaural {currentTrack.dsp_flags.binaural_freq}Hz
                  </Badge>
                )}
                {currentTrack.dsp_flags.sidechain && (
                  <Badge variant="secondary">
                    <Activity className="w-3 h-3 mr-1" />
                    Sidechain {currentTrack.dsp_flags.bpm} BPM
                  </Badge>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
