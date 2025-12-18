"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import {
  Play,
  Pause,
  SkipForward,
  Volume2,
  VolumeX,
  Zap,
  Settings,
  Download,
  Disc3,
  Radio,
  Sparkles,
  Moon,
  Sun,
  Maximize2,
  Activity,
  Headphones,
  Filter,
  Repeat,
  AlertCircle,
  CheckCircle,
  Loader2,
  Music,
  Layers,
  Sliders,
} from "lucide-react";

// ============================================================================
// TYPES & CONSTANTS
// ============================================================================

interface Track {
  id: string;
  name: string;
  url: string;
  bpm: number;
  key: string;
  genre: string;
  duration: number;
  waveform?: number[];
}

interface DeckState {
  track: Track | null;
  isPlaying: boolean;
  volume: number;
  eq: { low: number; mid: number; high: number };
  tempo: number;
  cuePoint: number;
  loop: { enabled: boolean; start: number; end: number };
}

interface GenerationRequest {
  genre: string;
  bpm: number;
  key: string;
  layers: string[];
  duration: number;
}

const GENRES = ["Trance", "House", "Techno", "Deep", "Bass", "Ambient", "Hard", "Chillout"];
const KEYS = ["C", "D", "E", "F", "G", "A", "B"];
const LAYERS = ["drums", "bass", "music", "pad", "texture"];

const API_BASE = "http://localhost:8000";

// ... rest of the code continues exactly the same

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function DJDashboard() {
  // Theme
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Decks
  const [deckA, setDeckA] = useState<DeckState>({
    track: null,
    isPlaying: false,
    volume: 75,
    eq: { low: 50, mid: 50, high: 50 },
    tempo: 100,
    cuePoint: 0,
    loop: { enabled: false, start: 0, end: 0 },
  });

  const [deckB, setDeckB] = useState<DeckState>({
    track: null,
    isPlaying: false,
    volume: 75,
    eq: { low: 50, mid: 50, high: 50 },
    tempo: 100,
    cuePoint: 0,
    loop: { enabled: false, start: 0, end: 0 },
  });

  // Mixer
  const [crossfader, setCrossfader] = useState(50); // 0 = full A, 100 = full B
  const [masterVolume, setMasterVolume] = useState(80);
  const [masterEQ, setMasterEQ] = useState({ low: 50, mid: 50, high: 50 });

  // Effects
  const [effects, setEffects] = useState({
    reverb: 0,
    delay: 0,
    filter: 50,
    enabled: false,
  });

  // Generation
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationQueue, setGenerationQueue] = useState<GenerationRequest[]>([]);
  const [generatedTracks, setGeneratedTracks] = useState<Track[]>([]);
  const [autoGenerate, setAutoGenerate] = useState(false);

  // AI Assistant
  const [aiMixing, setAiMixing] = useState(false);
  const [aiSuggestions, setAiSuggestions] = useState<string[]>([]);

  // UI State
  const [activeView, setActiveView] = useState<"mixer" | "generator" | "library">("mixer");
  const [showSettings, setShowSettings] = useState(false);
  const [vuMeterA, setVuMeterA] = useState(0);
  const [vuMeterB, setVuMeterB] = useState(0);

  // Audio References
  const audioA = useRef<HTMLAudioElement | null>(null);
  const audioB = useRef<HTMLAudioElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // ============================================================================
  // AUDIO ENGINE SETUP
  // ============================================================================

  useEffect(() => {
    // Initialize Web Audio API
    if (typeof window !== "undefined" && !audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    // Initialize audio elements
    if (!audioA.current) {
      audioA.current = new Audio();
      audioA.current.crossOrigin = "anonymous";
    }
    if (!audioB.current) {
      audioB.current = new Audio();
      audioB.current.crossOrigin = "anonymous";
    }

    return () => {
      if (audioA.current) {
        audioA.current.pause();
        audioA.current.src = "";
      }
      if (audioB.current) {
        audioB.current.pause();
        audioB.current.src = "";
      }
    };
  }, []);

  // ============================================================================
  // DECK CONTROLS
  // ============================================================================

  const loadTrackToDeck = useCallback((track: Track, deck: "A" | "B") => {
    const audioElement = deck === "A" ? audioA.current : audioB.current;
    const setDeck = deck === "A" ? setDeckA : setDeckB;

    if (audioElement) {
      audioElement.src = track.url;
      audioElement.load();

      setDeck((prev) => ({
        ...prev,
        track,
        isPlaying: false,
      }));
    }
  }, []);

  const togglePlay = useCallback((deck: "A" | "B") => {
    const audioElement = deck === "A" ? audioA.current : audioB.current;
    const currentDeck = deck === "A" ? deckA : deckB;
    const setDeck = deck === "A" ? setDeckA : setDeckB;

    if (!audioElement || !currentDeck.track) return;

    if (currentDeck.isPlaying) {
      audioElement.pause();
      setDeck((prev) => ({ ...prev, isPlaying: false }));
    } else {
      audioElement.play().catch((err) => console.error("Playback error:", err));
      setDeck((prev) => ({ ...prev, isPlaying: true }));
    }
  }, [deckA, deckB]);

  const setDeckVolume = useCallback((deck: "A" | "B", volume: number) => {
    const audioElement = deck === "A" ? audioA.current : audioB.current;
    const setDeck = deck === "A" ? setDeckA : setDeckB;

    if (audioElement) {
      // Apply crossfader and master volume
      const crossfaderGain = deck === "A" 
        ? (100 - crossfader) / 100 
        : crossfader / 100;
      const finalVolume = (volume / 100) * (masterVolume / 100) * crossfaderGain;
      audioElement.volume = Math.max(0, Math.min(1, finalVolume));
    }

    setDeck((prev) => ({ ...prev, volume }));
  }, [crossfader, masterVolume]);

  // Update volumes when crossfader or master changes
  useEffect(() => {
    setDeckVolume("A", deckA.volume);
    setDeckVolume("B", deckB.volume);
  }, [crossfader, masterVolume, deckA.volume, deckB.volume, setDeckVolume]);

  // VU Meter simulation
  useEffect(() => {
    const interval = setInterval(() => {
      if (deckA.isPlaying) {
        setVuMeterA(Math.random() * 60 + 40);
      } else {
        setVuMeterA(Math.max(0, vuMeterA - 10));
      }

      if (deckB.isPlaying) {
        setVuMeterB(Math.random() * 60 + 40);
      } else {
        setVuMeterB(Math.max(0, vuMeterB - 10));
      }
    }, 100);

    return () => clearInterval(interval);
  }, [deckA.isPlaying, deckB.isPlaying]);

  // ============================================================================
  // MUSIC GENERATION
  // ============================================================================

 const generateTrack = useCallback(async (request: GenerationRequest) => {
  setIsGenerating(true);

  try {
    // ✅ FIXED: Match the backend's GenerateRequest model
    const payload = {
      genre: request.genre,
      bpm: request.bpm,
      key: request.key,
      layers: request.layers,  // ✅ Send array directly, not { enabled: [...] }
      duration: request.duration,
      scale: "minor",
      instrument: "hybrid",
      texture: "none",
      target_lufs: -14.0
    };

    console.log("🎵 Sending generation request:", payload);

    const response = await fetch(`${API_BASE}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    console.log("📡 Response status:", response.status);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: "Unknown error" }));
      console.error("❌ Generation failed:", errorData);
      throw new Error(errorData.error || errorData.detail || "Generation failed");
    }

    const data = await response.json();
    console.log("✅ Generation successful:", data);

    const newTrack: Track = {
      id: data.id,
      name: data.name,
      url: data.url,
      bpm: data.bpm,
      key: data.key,
      genre: data.genre,
      duration: data.duration,
    };

    setGeneratedTracks((prev) => [newTrack, ...prev]);
    setIsGenerating(false);

    // Auto-load to empty deck
    if (!deckA.track) {
      loadTrackToDeck(newTrack, "A");
    } else if (!deckB.track) {
      loadTrackToDeck(newTrack, "B");
    }

    return newTrack;
  } catch (error) {
    console.error("❌ Generation error:", error);
    setIsGenerating(false);
    
    // Show error to user
    alert(`Generation failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    throw error;
  }
}, [deckA.track, deckB.track, loadTrackToDeck]);

  // Auto-generate next track when current is near end
  useEffect(() => {
    if (!autoGenerate) return;

    const checkInterval = setInterval(() => {
      const audioElement = deckA.isPlaying ? audioA.current : audioB.current;
      if (!audioElement) return;

      const remaining = audioElement.duration - audioElement.currentTime;

      // Generate next track when 30 seconds remaining
      if (remaining > 0 && remaining < 30 && !isGenerating) {
        const currentGenre = deckA.track?.genre || "Trance";
        const currentBPM = deckA.track?.bpm || 128;

        generateTrack({
          genre: currentGenre,
          bpm: currentBPM,
          key: "A",
          layers: ["drums", "bass", "music"],
          duration: 180,
        });
      }
    }, 5000);

    return () => clearInterval(checkInterval);
  }, [autoGenerate, deckA.isPlaying, deckA.track, isGenerating, generateTrack]);

  // ============================================================================
  // AI MIXING ASSISTANT
  // ============================================================================

  const performAIMix = useCallback(() => {
    if (!deckA.track || !deckB.track) return;

    setAiMixing(true);

    // Simulate AI decision making
    setTimeout(() => {
      // Smart crossfade over 8 seconds
      const startCrossfader = crossfader;
      const targetCrossfader = deckA.isPlaying ? 100 : 0;
      const steps = 80; // 8 seconds at 100ms intervals

      let currentStep = 0;
      const fadeInterval = setInterval(() => {
        currentStep++;
        const progress = currentStep / steps;
        const newValue = startCrossfader + (targetCrossfader - startCrossfader) * progress;
        setCrossfader(newValue);

        if (currentStep >= steps) {
          clearInterval(fadeInterval);
          setAiMixing(false);

          // Auto-play the faded-in deck
          if (targetCrossfader === 100 && !deckB.isPlaying) {
            togglePlay("B");
          } else if (targetCrossfader === 0 && !deckA.isPlaying) {
            togglePlay("A");
          }
        }
      }, 100);
    }, 500);
  }, [deckA, deckB, crossfader, togglePlay]);

  // ============================================================================
  // THEME TOGGLE
  // ============================================================================

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDarkMode]);

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div className={`min-h-screen transition-colors duration-300 ${isDarkMode ? "bg-black" : "bg-gray-50"}`}>
      {/* Animated Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div
          className={`absolute inset-0 ${
            isDarkMode
              ? "bg-gradient-to-br from-purple-900/10 via-black to-blue-900/10"
              : "bg-gradient-to-br from-purple-50 via-white to-blue-50"
          }`}
        >
          {/* Animated particles */}
          {isDarkMode && (
            <>
              <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-purple-500/5 rounded-full blur-3xl animate-pulse" />
              <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-pulse delay-1000" />
              <div className="absolute top-1/2 right-1/3 w-48 h-48 bg-pink-500/5 rounded-full blur-2xl animate-pulse delay-500" />
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 p-4 md:p-6 space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-2xl ${isDarkMode ? "bg-purple-500/10" : "bg-purple-500/20"}`}>
              <Disc3 className={`w-8 h-8 ${isDarkMode ? "text-purple-400" : "text-purple-600"} animate-spin-slow`} />
            </div>
            <div>
              <h1 className={`text-2xl md:text-3xl font-bold ${isDarkMode ? "text-white" : "text-gray-900"}`}>
                SoundFlow AI DJ
              </h1>
              <p className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
                Professional Live Mixing & AI Generation
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Theme Toggle */}
            <Button
              variant="outline"
              size="icon"
              className={`rounded-2xl ${isDarkMode ? "border-gray-700" : "border-gray-300"}`}
              onClick={() => setIsDarkMode(!isDarkMode)}
            >
              {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </Button>

            {/* Settings */}
            <Dialog open={showSettings} onOpenChange={setShowSettings}>
              <DialogTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  className={`rounded-2xl ${isDarkMode ? "border-gray-700" : "border-gray-300"}`}
                >
                  <Settings className="w-5 h-5" />
                </Button>
              </DialogTrigger>
              <DialogContent className={`${isDarkMode ? "bg-gray-900 text-white" : "bg-white"} rounded-3xl`}>
                <DialogHeader>
                  <DialogTitle>Settings</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span>Auto-generate next track</span>
                    <Switch checked={autoGenerate} onCheckedChange={setAutoGenerate} />
                  </div>
                  <div className="flex items-center justify-between">
                    <span>AI Auto-mixing</span>
                    <Switch checked={aiMixing} onCheckedChange={setAiMixing} />
                  </div>
                </div>
              </DialogContent>
            </Dialog>

            {/* Status Badge */}
            <Badge
              variant="secondary"
              className={`rounded-full px-4 py-2 ${
                isDarkMode ? "bg-green-500/10 text-green-400" : "bg-green-500/20 text-green-700"
              }`}
            >
              <Activity className="w-4 h-4 mr-2 animate-pulse" />
              LIVE
            </Badge>
          </div>
        </div>

        {/* Main Interface */}
        <Tabs value={activeView} onValueChange={(v: any) => setActiveView(v)} className="w-full">
          <TabsList
            className={`grid w-full max-w-md grid-cols-3 ${
              isDarkMode ? "bg-gray-900/50" : "bg-white"
            } rounded-2xl p-1`}
          >
            <TabsTrigger value="mixer" className="rounded-xl">
              <Sliders className="w-4 h-4 mr-2" />
              Mixer
            </TabsTrigger>
            <TabsTrigger value="generator" className="rounded-xl">
              <Sparkles className="w-4 h-4 mr-2" />
              Generator
            </TabsTrigger>
            <TabsTrigger value="library" className="rounded-xl">
              <Music className="w-4 h-4 mr-2" />
              Library
            </TabsTrigger>
          </TabsList>

          {/* ===== MIXER VIEW ===== */}
          <TabsContent value="mixer" className="mt-4 space-y-4">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
              {/* Deck A */}
              <DeckInterface
                deck="A"
                state={deckA}
                onTogglePlay={() => togglePlay("A")}
                onVolumeChange={(v) => setDeckVolume("A", v)}
                onEQChange={(eq) => setDeckA((prev) => ({ ...prev, eq }))}
                vuMeter={vuMeterA}
                isDarkMode={isDarkMode}
              />

              {/* Center Mixer */}
              <Card
                className={`rounded-3xl ${
                  isDarkMode ? "bg-gray-900/50 border-gray-800" : "bg-white border-gray-200"
                } backdrop-blur-xl`}
              >
                <CardContent className="p-6 space-y-6">
                  {/* Master VU Meter */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className={`text-sm font-medium ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
                        MASTER
                      </span>
                      <Badge variant="secondary" className="rounded-full">
                        {masterVolume}
                      </Badge>
                    </div>
                    <div className="flex gap-1 h-32">
                      {Array.from({ length: 20 }).map((_, i) => {
                        const threshold = (i / 20) * 100;
                        const isActive = Math.max(vuMeterA, vuMeterB) > threshold;
                        return (
                          <div
                            key={i}
                            className={`flex-1 rounded-full transition-colors ${
                              isActive
                                ? i > 16
                                  ? "bg-red-500"
                                  : i > 12
                                  ? "bg-yellow-500"
                                  : "bg-green-500"
                                : isDarkMode
                                ? "bg-gray-800"
                                : "bg-gray-200"
                            }`}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Crossfader */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Badge variant="secondary" className="rounded-full">A</Badge>
                      <span className={`text-xs ${isDarkMode ? "text-gray-500" : "text-gray-400"}`}>
                        CROSSFADER
                      </span>
                      <Badge variant="secondary" className="rounded-full">B</Badge>
                    </div>
                    <div className="relative">
                      <Slider
                        value={[crossfader]}
                        onValueChange={([v]) => setCrossfader(v)}
                        min={0}
                        max={100}
                        step={1}
                        className="py-4"
                      />
                      <div
                        className={`absolute top-0 left-1/2 transform -translate-x-1/2 w-1 h-full ${
                          isDarkMode ? "bg-gray-700" : "bg-gray-300"
                        } rounded-full`}
                      />
                    </div>
                  </div>

                  {/* Master Volume */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className={`text-sm font-medium ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
                        Volume
                      </span>
                      <Badge variant="secondary" className="rounded-full">
                        {masterVolume}%
                      </Badge>
                    </div>
                    <Slider
                      value={[masterVolume]}
                      onValueChange={([v]) => setMasterVolume(v)}
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>

                  {/* Master EQ */}
                  <div className="space-y-3">
                    <span className={`text-sm font-medium ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
                      Master EQ
                    </span>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="space-y-2">
                        <span className="text-xs">LOW</span>
                        <Slider
                          value={[masterEQ.low]}
                          onValueChange={([v]) => setMasterEQ((prev) => ({ ...prev, low: v }))}
                          min={0}
                          max={100}
                          orientation="vertical"
                          className="h-24"
                        />
                      </div>
                      <div className="space-y-2">
                        <span className="text-xs">MID</span>
                        <Slider
                          value={[masterEQ.mid]}
                          onValueChange={([v]) => setMasterEQ((prev) => ({ ...prev, mid: v }))}
                          min={0}
                          max={100}
                          orientation="vertical"
                          className="h-24"
                        />
                      </div>
                      <div className="space-y-2">
                        <span className="text-xs">HIGH</span>
                        <Slider
                          value={[masterEQ.high]}
                          onValueChange={([v]) => setMasterEQ((prev) => ({ ...prev, high: v }))}
                          min={0}
                          max={100}
                          orientation="vertical"
                          className="h-24"
                        />
                      </div>
                    </div>
                  </div>

                  {/* AI Mixing */}
                  <Button
                    className={`w-full rounded-2xl ${
                      isDarkMode
                        ? "bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                        : "bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600"
                    }`}
                    onClick={performAIMix}
                    disabled={!deckA.track || !deckB.track || aiMixing}
                  >
                    {aiMixing ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        AI Mixing...
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4 mr-2" />
                        AI Auto-Mix
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Deck B */}
              <DeckInterface
                deck="B"
                state={deckB}
                onTogglePlay={() => togglePlay("B")}
                onVolumeChange={(v) => setDeckVolume("B", v)}
                onEQChange={(eq) => setDeckB((prev) => ({ ...prev, eq }))}
                vuMeter={vuMeterB}
                isDarkMode={isDarkMode}
              />
            </div>
          </TabsContent>

          {/* ===== GENERATOR VIEW ===== */}
          <TabsContent value="generator" className="mt-4">
            <GeneratorInterface
              onGenerate={generateTrack}
              isGenerating={isGenerating}
              isDarkMode={isDarkMode}
            />
          </TabsContent>

          {/* ===== LIBRARY VIEW ===== */}
          <TabsContent value="library" className="mt-4">
            <LibraryInterface
              tracks={generatedTracks}
              onLoadTrack={loadTrackToDeck}
              isDarkMode={isDarkMode}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

// ============================================================================
// DECK INTERFACE COMPONENT
// ============================================================================

interface DeckInterfaceProps {
  deck: "A" | "B";
  state: DeckState;
  onTogglePlay: () => void;
  onVolumeChange: (volume: number) => void;
  onEQChange: (eq: { low: number; mid: number; high: number }) => void;
  vuMeter: number;
  isDarkMode: boolean;
}

function DeckInterface({
  deck,
  state,
  onTogglePlay,
  onVolumeChange,
  onEQChange,
  vuMeter,
  isDarkMode,
}: DeckInterfaceProps) {
  return (
    <Card
      className={`rounded-3xl ${
        isDarkMode ? "bg-gray-900/50 border-gray-800" : "bg-white border-gray-200"
      } backdrop-blur-xl`}
    >
      <CardContent className="p-6 space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <Badge
            variant="secondary"
            className={`rounded-full text-lg px-4 py-2 ${
              deck === "A"
                ? isDarkMode
                  ? "bg-blue-500/20 text-blue-400"
                  : "bg-blue-500/30 text-blue-700"
                : isDarkMode
                ? "bg-purple-500/20 text-purple-400"
                : "bg-purple-500/30 text-purple-700"
            }`}
          >
            DECK {deck}
          </Badge>
          <div className="flex gap-2">
            <Button variant="ghost" size="icon" className="rounded-xl">
              <Headphones className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" className="rounded-xl">
              <Repeat className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Track Info */}
        {state.track ? (
          <div
            className={`p-4 rounded-2xl ${
              isDarkMode ? "bg-gray-800/50" : "bg-gray-100"
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1 min-w-0">
                <h3 className={`font-semibold truncate ${isDarkMode ? "text-white" : "text-gray-900"}`}>
                  {state.track.name}
                </h3>
                <div className="flex items-center gap-2 mt-1 flex-wrap">
                  <Badge variant="secondary" className="rounded-full text-xs">
                    {state.track.bpm} BPM
                  </Badge>
                  <Badge variant="secondary" className="rounded-full text-xs">
                    {state.track.key}
                  </Badge>
                  <Badge variant="secondary" className="rounded-full text-xs">
                    {state.track.genre}
                  </Badge>
                </div>
              </div>
              <div
                className={`w-16 h-16 rounded-xl ${
                  isDarkMode ? "bg-gray-700" : "bg-gray-200"
                } flex items-center justify-center`}
              >
                <Activity className="w-8 h-8" />
              </div>
            </div>
          </div>
        ) : (
          <div
            className={`p-8 rounded-2xl border-2 border-dashed ${
              isDarkMode ? "border-gray-700" : "border-gray-300"
            } text-center`}
          >
            <Music className={`w-12 h-12 mx-auto mb-2 ${isDarkMode ? "text-gray-600" : "text-gray-400"}`} />
            <p className={`text-sm ${isDarkMode ? "text-gray-500" : "text-gray-600"}`}>
              No track loaded
            </p>
          </div>
        )}

        {/* AudioWaveform Display */}
        <div className="relative h-24 rounded-2xl overflow-hidden bg-gradient-to-r from-blue-500/10 to-purple-500/10">
          <div className="absolute inset-0 flex items-center justify-center">
            {state.isPlaying ? (
              <div className="flex items-end gap-1 h-full p-2">
                {Array.from({ length: 50 }).map((_, i) => (
                  <div
                    key={i}
                    className={`flex-1 ${
                      deck === "A" ? "bg-blue-500" : "bg-purple-500"
                    } rounded-full animate-pulse`}
                    style={{
                      height: `${Math.random() * 80 + 20}%`,
                      animationDelay: `${i * 50}ms`,
                    }}
                  />
                ))}
              </div>
            ) : (
              <div className="flex items-end gap-1 h-full p-2">
                {Array.from({ length: 50 }).map((_, i) => (
                  <div
                    key={i}
                    className={`flex-1 ${
                      isDarkMode ? "bg-gray-700" : "bg-gray-300"
                    } rounded-full`}
                    style={{ height: "30%" }}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Transport Controls */}
        <div className="flex items-center justify-center gap-3">
          <Button
            size="icon"
            variant="outline"
            className="rounded-2xl"
            onClick={onTogglePlay}
            disabled={!state.track}
          >
            {state.isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </Button>
          <Button size="icon" variant="outline" className="rounded-2xl" disabled={!state.track}>
            <SkipForward className="w-5 h-5" />
          </Button>
        </div>

        {/* Volume */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
              Volume
            </span>
            <Badge variant="secondary" className="rounded-full">
              {state.volume}%
            </Badge>
          </div>
          <Slider
            value={[state.volume]}
            onValueChange={([v]) => onVolumeChange(v)}
            min={0}
            max={100}
            step={1}
          />
        </div>

        {/* EQ */}
        <div className="space-y-3">
          <span className={`text-sm font-medium ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
            EQ
          </span>
          <div className="grid grid-cols-3 gap-3">
            <div className="space-y-2">
              <span className="text-xs">LOW</span>
              <Slider
                value={[state.eq.low]}
                onValueChange={([v]) => onEQChange({ ...state.eq, low: v })}
                min={0}
                max={100}
                orientation="vertical"
                className="h-20"
              />
            </div>
            <div className="space-y-2">
              <span className="text-xs">MID</span>
              <Slider
                value={[state.eq.mid]}
                onValueChange={([v]) => onEQChange({ ...state.eq, mid: v })}
                min={0}
                max={100}
                orientation="vertical"
                className="h-20"
              />
            </div>
            <div className="space-y-2">
              <span className="text-xs">HIGH</span>
              <Slider
                value={[state.eq.high]}
                onValueChange={([v]) => onEQChange({ ...state.eq, high: v })}
                min={0}
                max={100}
                orientation="vertical"
                className="h-20"
              />
            </div>
          </div>
        </div>

        {/* VU Meter */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className={`text-xs ${isDarkMode ? "text-gray-500" : "text-gray-400"}`}>
              LEVEL
            </span>
            <Badge variant="secondary" className="rounded-full text-xs">
              {Math.round(vuMeter)}
            </Badge>
          </div>
          <Progress value={vuMeter} className="h-2" />
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// GENERATOR INTERFACE COMPONENT
// ============================================================================

interface GeneratorInterfaceProps {
  onGenerate: (request: GenerationRequest) => Promise<Track>;
  isGenerating: boolean;
  isDarkMode: boolean;
}

function GeneratorInterface({ onGenerate, isGenerating, isDarkMode }: GeneratorInterfaceProps) {
  const [genre, setGenre] = useState("Trance");
  const [bpm, setBpm] = useState(128);
  const [key, setKey] = useState("A");
  const [duration, setDuration] = useState(180);
  const [selectedLayers, setSelectedLayers] = useState<string[]>(["drums", "bass", "music"]);

  const handleGenerate = () => {
    onGenerate({
      genre,
      bpm,
      key,
      layers: selectedLayers,
      duration,
    });
  };

  return (
    <Card
      className={`rounded-3xl ${
        isDarkMode ? "bg-gray-900/50 border-gray-800" : "bg-white border-gray-200"
      } backdrop-blur-xl`}
    >
      <CardContent className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className={`text-xl font-bold ${isDarkMode ? "text-white" : "text-gray-900"}`}>
              AI Music Generator
            </h2>
            <p className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
              Generate tracks in real-time while mixing
            </p>
          </div>
          <Badge
            variant="secondary"
            className={`rounded-full ${
              isGenerating
                ? isDarkMode
                  ? "bg-yellow-500/20 text-yellow-400"
                  : "bg-yellow-500/30 text-yellow-700"
                : isDarkMode
                ? "bg-green-500/20 text-green-400"
                : "bg-green-500/30 text-green-700"
            }`}
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-3 h-3 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <CheckCircle className="w-3 h-3 mr-2" />
                Ready
              </>
            )}
          </Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Genre Selection */}
          <div className="space-y-3">
            <label className={`text-sm font-medium ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Genre
            </label>
            <div className="grid grid-cols-2 gap-2">
              {GENRES.map((g) => (
                <Button
                  key={g}
                  variant={genre === g ? "default" : "outline"}
                  className="rounded-2xl"
                  onClick={() => setGenre(g)}
                >
                  {g}
                </Button>
              ))}
            </div>
          </div>

          {/* BPM */}
          <div className="space-y-3">
            <label className={`text-sm font-medium ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              BPM
            </label>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Badge variant="secondary" className="rounded-full">
                  {bpm}
                </Badge>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" className="rounded-xl" onClick={() => setBpm(Math.max(60, bpm - 1))}>
                    -1
                  </Button>
                  <Button size="sm" variant="outline" className="rounded-xl" onClick={() => setBpm(Math.min(200, bpm + 1))}>
                    +1
                  </Button>
                </div>
              </div>
              <Slider value={[bpm]} onValueChange={([v]) => setBpm(v)} min={60} max={200} step={1} />
            </div>
          </div>

          {/* Key */}
          <div className="space-y-3">
            <label className={`text-sm font-medium ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Key
            </label>
            <div className="flex gap-2">
              {KEYS.map((k) => (
                <Button
                  key={k}
                  variant={key === k ? "default" : "outline"}
                  className="rounded-2xl flex-1"
                  onClick={() => setKey(k)}
                >
                  {k}
                </Button>
              ))}
            </div>
          </div>

          {/* Duration */}
          <div className="space-y-3">
            <label className={`text-sm font-medium ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Duration
            </label>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Badge variant="secondary" className="rounded-full">
                  {Math.floor(duration / 60)}:{(duration % 60).toString().padStart(2, "0")}
                </Badge>
              </div>
              <Slider value={[duration]} onValueChange={([v]) => setDuration(v)} min={30} max={600} step={30} />
            </div>
          </div>

          {/* Layers */}
          <div className="space-y-3 md:col-span-2">
            <label className={`text-sm font-medium ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Layers
            </label>
            <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
              {LAYERS.map((layer) => {
                const isSelected = selectedLayers.includes(layer);
                return (
                  <Button
                    key={layer}
                    variant={isSelected ? "default" : "outline"}
                    className="rounded-2xl"
                    onClick={() => {
                      if (isSelected) {
                        setSelectedLayers(selectedLayers.filter((l) => l !== layer));
                      } else {
                        setSelectedLayers([...selectedLayers, layer]);
                      }
                    }}
                  >
                    <Layers className="w-4 h-4 mr-2" />
                    {layer}
                  </Button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Generate Button */}
        <Button
          className={`w-full rounded-2xl h-14 text-lg ${
            isDarkMode
              ? "bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
              : "bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600"
          }`}
          onClick={handleGenerate}
          disabled={isGenerating || selectedLayers.length === 0}
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              Generating Track...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5 mr-2" />
              Generate Track
            </>
          )}
        </Button>

        {/* Tips */}
        <div
          className={`p-4 rounded-2xl ${
            isDarkMode ? "bg-blue-500/10 border border-blue-500/20" : "bg-blue-50 border border-blue-200"
          }`}
        >
          <div className="flex items-start gap-3">
            <AlertCircle className={`w-5 h-5 mt-0.5 ${isDarkMode ? "text-blue-400" : "text-blue-600"}`} />
            <div className="flex-1">
              <h4 className={`font-medium ${isDarkMode ? "text-blue-300" : "text-blue-900"}`}>
                Pro Tip
              </h4>
              <p className={`text-sm mt-1 ${isDarkMode ? "text-blue-400/80" : "text-blue-700"}`}>
                Enable "Auto-generate" in settings to automatically create the next track 30 seconds before the current
                one ends.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// LIBRARY INTERFACE COMPONENT
// ============================================================================

interface LibraryInterfaceProps {
  tracks: Track[];
  onLoadTrack: (track: Track, deck: "A" | "B") => void;
  isDarkMode: boolean;
}

function LibraryInterface({ tracks, onLoadTrack, isDarkMode }: LibraryInterfaceProps) {
  return (
    <Card
      className={`rounded-3xl ${
        isDarkMode ? "bg-gray-900/50 border-gray-800" : "bg-white border-gray-200"
      } backdrop-blur-xl`}
    >
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className={`text-xl font-bold ${isDarkMode ? "text-white" : "text-gray-900"}`}>
              Track Library
            </h2>
            <p className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
              {tracks.length} generated tracks
            </p>
          </div>
        </div>

        <ScrollArea className="h-[600px]">
          {tracks.length === 0 ? (
            <div
              className={`p-12 rounded-2xl border-2 border-dashed ${
                isDarkMode ? "border-gray-700" : "border-gray-300"
              } text-center`}
            >
              <Music className={`w-16 h-16 mx-auto mb-4 ${isDarkMode ? "text-gray-600" : "text-gray-400"}`} />
              <h3 className={`text-lg font-medium mb-2 ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
                No tracks yet
              </h3>
              <p className={`text-sm ${isDarkMode ? "text-gray-500" : "text-gray-500"}`}>
                Generate your first track using the AI Generator
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {tracks.map((track) => (
                <div
                  key={track.id}
                  className={`p-4 rounded-2xl ${
                    isDarkMode ? "bg-gray-800/50 hover:bg-gray-800" : "bg-gray-50 hover:bg-gray-100"
                  } transition-colors`}
                >
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <h3 className={`font-semibold truncate ${isDarkMode ? "text-white" : "text-gray-900"}`}>
                        {track.name}
                      </h3>
                      <div className="flex items-center gap-2 mt-1 flex-wrap">
                        <Badge variant="secondary" className="rounded-full text-xs">
                          {track.bpm} BPM
                        </Badge>
                        <Badge variant="secondary" className="rounded-full text-xs">
                          {track.key}
                        </Badge>
                        <Badge variant="secondary" className="rounded-full text-xs">
                          {track.genre}
                        </Badge>
                        <Badge variant="secondary" className="rounded-full text-xs">
                          {Math.floor(track.duration / 60)}:{(track.duration % 60).toString().padStart(2, "0")}
                        </Badge>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="rounded-xl"
                        onClick={() => onLoadTrack(track, "A")}
                      >
                        Load A
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="rounded-xl"
                        onClick={() => onLoadTrack(track, "B")}
                      >
                        Load B
                      </Button>
                      <Button variant="ghost" size="icon" className="rounded-xl">
                        <Download className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}