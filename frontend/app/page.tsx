"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import { soundscapeData, colorClasses } from "@/lib/presets";
import { AudioEngine } from "@/lib/audioEngine";
import { getClientTier } from "@/lib/tier";
import type { CreateSessionResponse, PremiumDailyResponse } from "@/lib/types";

// Reliable fallback audio for testing/offline scenarios (Wikimedia Commons)
const FALLBACK_MUSIC_URL_1 =
  "https://upload.wikimedia.org/wikipedia/commons/9/9d/Anthem_of_Europe_%28US_Navy_instrumental_long_version%29.ogg";

const FALLBACK_MUSIC_URL = "/fallback/electronic.mp3";

  const FALLBACK_NATURE_URL =
  "https://upload.wikimedia.org/wikipedia/commons/8/8a/Sound_of_rain.ogg";



export default function Page() {
  const engine = useMemo(() => new AudioEngine(), []);
  const tier = useMemo(() => getClientTier(), []);

  // Modal UI
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Player state
  const [isPlaying, setIsPlaying] = useState(false);
  const [musicVol, setMusicVol] = useState(0.7);
  const [natureVol, setNatureVol] = useState(0.3);

  const [nowTitle, setNowTitle] = useState("Deep Focus Session");
  const [nowSubtitle, setNowSubtitle] = useState("Ready to Start");

  // Start widget inputs
  const [goalSelect, setGoalSelect] = useState("Deep Focus");
  const [durationSelect, setDurationSelect] = useState("50 minutes");

  // Builder steps
  const [goalBuilder, setGoalBuilder] = useState<string | null>(null);
  const [durationChip, setDurationChip] = useState<string | null>(null); // "25m" | "50m" | "90m" | "Custom"
  const [energy, setEnergy] = useState(50);
  const [ambience, setAmbience] = useState(30);
  const [nature, setNature] = useState("Rain");

  const [energyLabel, setEnergyLabel] = useState("Medium");
  const [ambienceLabel, setAmbienceLabel] = useState("Light");

  const [isGenerating, setIsGenerating] = useState(false);

  // Keep volumes applied
  useEffect(() => {
    engine.setMusicVolume(musicVol);
    engine.setAmbienceVolume(natureVol);
  }, [engine, musicVol, natureVol]);

  useEffect(() => {
    setEnergyLabel(energy < 33 ? "Calm" : energy < 66 ? "Medium" : "Driven");
  }, [energy]);

  useEffect(() => {
    setAmbienceLabel(ambience < 33 ? "Light" : ambience < 66 ? "Medium" : "Heavy");
  }, [ambience]);

  function minutesFromLabel(label: string) {
    const m = label.match(/(\d+)\s*minutes/i);
    if (m?.[1]) return parseInt(m[1], 10);
    return 50;
  }

  function minutesFromChip(chip: string | null) {
    if (!chip) return minutesFromLabel(durationSelect);
    if (chip.toLowerCase() === "custom") return minutesFromLabel(durationSelect);
    const m = chip.match(/(\d+)/);
    return m ? parseInt(m[1], 10) : 50;
  }

  /**
   * Helper to load the default "Radio" session
   */
  function loadDefaultSession() {
    console.log("Loading default daily session...");
    engine.setSources(
  { url: FALLBACK_MUSIC_URL, type: "audio/mpeg" },
  { url: FALLBACK_NATURE_URL, type: "audio/ogg" }
);

    setNowTitle("Daily Flow Radio");
    setNowSubtitle("Infinite Mix • System Audio");
  }

  async function togglePlayPause() {
    // FIX: "Radio Mode" - If no session is ready, load the default daily session automatically
    // instead of showing an error alert.
    if (!engine.isReady()) {
      loadDefaultSession();
    }

    if (!isPlaying) {
      try {
        await engine.play();
        setIsPlaying(true);
      } catch (e) {
        console.error("Playback failed", e);
        alert("Audio playback failed. Please check your connection or try generating a new session.");
      }
    } else {
      engine.pause();
      setIsPlaying(false);
    }
  }

  async function generateSession() {
    setIsGenerating(true);
    try {
      // Determine goal/duration exactly like your UI
      const goal = goalBuilder ?? goalSelect;
      const durationMin = minutesFromChip(durationChip);

      if (tier === "premium") {
        // Premium = fetch signed daily (or parameter-based in future)
        const res = await fetch("/api/premium", { method: "GET" });
        if (!res.ok) throw new Error(await res.text());
        const data = (await res.json()) as PremiumDailyResponse;

        // IMPORTANT: setSources queues stop->swap->(optional play) safely inside engine
        engine.setSources(data.musicUrl, data.ambienceUrl);

        setNowTitle(data.title || `${goal} Session`);
        setNowSubtitle(`Premium • ${Math.round(data.durationSec / 60)}m`);
      } else {
        // Free = deterministic routing (your existing /api/session)
        const res = await fetch("/api/session", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            goal,
            durationMin,
            energy,
            ambience,
            nature
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = (await res.json()) as CreateSessionResponse;

        engine.setSources(data.musicUrl, data.ambienceUrl);

        setNowTitle(`${goal} Session`);
        setNowSubtitle(`AI Generated • ${Math.round(data.durationSec / 60)}m`);
      }

      setIsModalOpen(false);

      // Call play() from this click handler → satisfies autoplay policies
      await engine.play();
      setIsPlaying(true);
    } catch (e) {
      console.warn("Primary audio generation/playback failed. Attempting fallback...", e);
      
      // --- FALLBACK LOGIC ---
      try {
        // Switch to known-good static URLs to test audio reproduction
        loadDefaultSession();
        
        // Try playing the fallback
        await engine.play();
        setIsPlaying(true);
        setIsModalOpen(false);
      } catch (fallbackError) {
        console.error("Fallback audio failed too:", fallbackError);
        alert("Failed to generate session and fallback audio failed to load. Please check your connection.");
      }
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <>
      {/* Navigation */}
      <header className="sticky top-0 z-50 glass border-b border-white/10">
        <nav className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-8">
              <a
                href="#"
                onClick={(e) => e.preventDefault()}
                className="text-xl font-semibold gradient-text"
              >
                SoundFlow AI
              </a>
              <div className="hidden md:flex items-center space-x-6">
                <a
                  href="#"
                  onClick={(e) => e.preventDefault()}
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  Explore
                </a>
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setIsModalOpen(true);
                  }}
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  AI Session
                </a>
                <a
                  href="#"
                  onClick={(e) => e.preventDefault()}
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  Playlists
                </a>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button className="hidden md:block px-4 py-2 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 transition-all duration-200">
                <i className="fas fa-user mr-2" />
                Account
              </button>
              <button className="md:hidden p-2 rounded-lg bg-white/5 hover:bg-white/10">
                <i className="fas fa-bars" />
              </button>
            </div>
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="pb-32">
        {/* Hero Section */}
        <section className="container mx-auto px-4 py-12 md:py-20">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              SoundFlow AI — <span className="gradient-text">focus music that flows.</span>
            </h1>
            <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto">
              AI-powered soundscapes that adapt to your focus needs. Personalized sessions for deep work,
              study, and relaxation.
            </p>

            {/* Pill Buttons */}
            <div className="flex flex-wrap justify-center gap-3 mb-12">
              <button
                onClick={() => {
                  setGoalBuilder("Deep Work");
                  setIsModalOpen(true);
                }}
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(45,212,191,0.2)]"
              >
                <i className="fas fa-brain mr-2" />
                Deep Work
              </button>

              <button
                onClick={() => {
                  setGoalBuilder("Study");
                  setIsModalOpen(true);
                }}
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(96,165,250,0.2)]"
              >
                <i className="fas fa-graduation-cap mr-2" />
                Study
              </button>

              <button
                onClick={() => {
                  setGoalBuilder("Relax");
                  setIsModalOpen(true);
                }}
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-purple-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(192,132,252,0.2)]"
              >
                <i className="fas fa-couch mr-2" />
                Relax
              </button>

              <button
                onClick={() => {
                  setGoalBuilder("Nature");
                  setIsModalOpen(true);
                }}
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(45,212,191,0.2)]"
              >
                <i className="fas fa-tree mr-2" />
                Nature
              </button>

              <button
                onClick={() => {
                  setGoalBuilder("Flow State");
                  setIsModalOpen(true);
                }}
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(96,165,250,0.2)]"
              >
                <i className="fas fa-infinity mr-2" />
                Flow State
              </button>
            </div>

            {/* Start Widget */}
            <div className="glass rounded-2xl p-8 max-w-xl mx-auto mb-16 text-left">
              <h3 className="text-xl font-semibold mb-6 text-center md:text-left">
                Start Your Focus Session
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Goal</label>
                  <select
                    value={goalSelect}
                    onChange={(e) => setGoalSelect(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                  >
                    <option>Deep Focus</option>
                    <option>Creative Work</option>
                    <option>Study Session</option>
                    <option>Meditation</option>
                    <option>Power Nap</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Duration</label>
                  <select
                    value={durationSelect}
                    onChange={(e) => setDurationSelect(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                  >
                    <option>25 minutes</option>
                    <option>50 minutes</option>
                    <option>90 minutes</option>
                    <option>Custom</option>
                  </select>
                </div>
                <div className="flex items-end">
                  <button
                    onClick={() => setIsModalOpen(true)}
                    className="w-full bg-gradient-to-r from-teal-500 to-blue-500 hover:from-teal-600 hover:to-blue-600 text-white font-semibold py-3 rounded-xl transition-all duration-300 hover:shadow-[0_0_30px_rgba(45,212,191,0.4)]"
                  >
                    <i className="fas fa-play mr-2" />
                    Start Session
                  </button>
                </div>
              </div>
              <p className="text-sm text-gray-500 text-center">
                AI will generate a personalized soundscape based on your selection
              </p>
            </div>
          </div>
        </section>

        {/* Explore Grid */}
        <section className="container mx-auto px-4 py-8">
          <h2 className="text-2xl font-bold mb-8">Explore Soundscapes</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {soundscapeData.map((item) => (
              <div
                key={item.id}
                className="group bg-white/5 rounded-2xl overflow-hidden border border-white/10 hover:border-white/20 transition-all duration-300 hover:shadow-[0_0_30px_rgba(255,255,255,0.1)]"
              >
                <div className="relative overflow-hidden">
                  <div className="relative h-48 w-full">
                    <Image
                      src={`https://picsum.photos/400/300?random=${item.imageId}`}
                      alt={`${item.title} cover image`}
                      fill
                      className="object-cover group-hover:scale-105 transition-transform duration-500"
                    />
                  </div>
                  <div
                    className={`absolute top-4 right-4 px-3 py-1 rounded-full text-xs font-semibold ${colorClasses[item.color]} border`}
                  >
                    {item.energy}
                  </div>
                </div>
                <div className="p-6">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-semibold text-lg">{item.title}</h3>
                    <span className="text-sm text-gray-400">{item.duration}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">{item.category}</span>
                    <button
                      onClick={() => setIsModalOpen(true)}
                      className="p-2 rounded-full bg-white/5 hover:bg-white/10 transition-colors duration-200"
                    >
                      <i className="fas fa-play" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* AI Session Builder Modal */}
        {isModalOpen && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
            onClick={(e) => {
              if (e.target === e.currentTarget) setIsModalOpen(false);
            }}
          >
            {/* Higher-contrast modal surface */}
            <div className="glass rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-white/20 bg-black/80">
              <div className="p-8">
                <div className="flex justify-between items-center mb-8">
                  <h3 className="text-2xl font-bold">AI Session Builder</h3>
                  <button onClick={() => setIsModalOpen(false)} className="p-2 rounded-lg hover:bg-white/10">
                    <i className="fas fa-times" />
                  </button>
                </div>

                {/* Step 1 */}
                <div className="mb-10">
                  <h4 className="text-lg font-semibold mb-4">Step 1: Select Your Goal</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {[
                      {
                        label: "Deep Work",
                        icon: "fa-laptop-code",
                        color: "text-teal-400",
                        desc: "Intense focus sessions"
                      },
                      { label: "Coding", icon: "fa-code", color: "text-blue-400", desc: "Programming flow state" },
                      {
                        label: "Reading",
                        icon: "fa-book",
                        color: "text-purple-400",
                        desc: "Extended reading sessions"
                      }
                    ].map((g) => (
                      <button
                        key={g.label}
                        onClick={() => setGoalBuilder(g.label)}
                        className={`p-6 rounded-xl bg-white/5 border transition-all duration-300 text-left ${
                          goalBuilder === g.label
                            ? "border-white/40 bg-white/10"
                            : "border-white/10 hover:border-white/20 hover:bg-white/10"
                        }`}
                      >
                        <i className={`fas ${g.icon} text-2xl mb-3 ${g.color}`} />
                        <h5 className="font-semibold mb-2">{g.label}</h5>
                        <p className="text-sm text-gray-300">{g.desc}</p>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Step 2 */}
                <div className="mb-10">
                  <h4 className="text-lg font-semibold mb-4">Step 2: Choose Duration</h4>
                  <div className="flex flex-wrap gap-3">
                    {["25m", "50m", "90m", "Custom"].map((d) => (
                      <button
                        key={d}
                        onClick={() => setDurationChip(d)}
                        className={`px-6 py-3 rounded-full bg-white/5 border transition-all duration-200 ${
                          durationChip === d
                            ? "border-white/40 bg-white/10"
                            : "border-white/10 hover:border-white/30"
                        }`}
                      >
                        {d}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Step 3 */}
                <div className="mb-10">
                  <h4 className="text-lg font-semibold mb-6">Step 3: Fine-tune Your Session</h4>
                  <div className="space-y-8">
                    <div>
                      <div className="flex justify-between mb-2">
                        <label className="font-medium">Energy Level</label>
                        <span className="text-teal-400">{energyLabel}</span>
                      </div>
                      <div className="flex items-center space-x-4">
                        <span className="text-sm text-gray-300">Calm</span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={energy}
                          onChange={(e) => setEnergy(parseInt(e.target.value, 10))}
                          className="flex-1 h-2 bg-white/15 rounded-lg appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-teal-400"
                        />
                        <span className="text-sm text-gray-300">Driven</span>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between mb-2">
                        <label className="font-medium">Ambience Level</label>
                        <span className="text-blue-400">{ambienceLabel}</span>
                      </div>
                      <div className="flex items-center space-x-4">
                        <span className="text-sm text-gray-300">None</span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={ambience}
                          onChange={(e) => setAmbience(parseInt(e.target.value, 10))}
                          className="flex-1 h-2 bg-white/15 rounded-lg appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-400"
                        />
                        <span className="text-sm text-gray-300">Heavy</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Step 4 */}
                <div className="mb-10">
                  <h4 className="text-lg font-semibold mb-4">Step 4: Add Nature Sounds</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                      { label: "Rain", icon: "fa-cloud-rain", color: "text-blue-400" },
                      { label: "Forest", icon: "fa-tree", color: "text-green-400" },
                      { label: "Fireplace", icon: "fa-fire", color: "text-orange-400" },
                      { label: "Ocean", icon: "fa-water", color: "text-cyan-400" }
                    ].map((n) => (
                      <button
                        key={n.label}
                        onClick={() => setNature(n.label)}
                        className={`p-4 rounded-xl bg-white/5 border transition-all duration-200 ${
                          nature === n.label
                            ? "border-white/40 bg-white/10"
                            : "border-white/10 hover:border-white/20"
                        }`}
                      >
                        <i className={`fas ${n.icon} text-2xl mb-2 ${n.color}`} />
                        <span>{n.label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Generate */}
                <button
                  onClick={generateSession}
                  disabled={isGenerating}
                  className="w-full bg-gradient-to-r from-teal-500 to-purple-500 hover:from-teal-600 hover:to-purple-600 text-white font-semibold py-4 rounded-xl transition-all duration-300 hover:shadow-[0_0_30px_rgba(168,85,247,0.4)] disabled:opacity-50"
                >
                  <i className="fas fa-magic mr-2" />
                  {isGenerating ? "Generating..." : "Generate AI Session"}
                </button>

                <p className="text-xs text-gray-300 mt-4">
                  Tier: <span className="font-semibold">{tier}</span>
                  {tier === "free" ? " (free uses deterministic routing)" : " (premium uses signed daily AI track)"}
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Bottom Player */}
      <div className="fixed bottom-0 left-0 right-0 z-40 glass border-t border-white/10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            {/* Track Info */}
            <div className="flex items-center space-x-4">
              <div className="relative w-12 h-12 rounded-xl overflow-hidden">
                <Image src="https://picsum.photos/60?random=1" alt="Current track cover" fill className="object-cover" />
              </div>
              <div>
                <h4 className="font-semibold">{nowTitle}</h4>
                <p className="text-sm text-gray-400">{nowSubtitle}</p>
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center space-x-6">
              <button className="p-2 rounded-full hover:bg-white/10" onClick={(e) => e.preventDefault()}>
                <i className="fas fa-step-backward" />
              </button>

              <button
                onClick={togglePlayPause}
                className="p-3 rounded-full bg-white/10 hover:bg-white/20 border border-white/20"
              >
                <i className={`fas ${isPlaying ? "fa-pause" : "fa-play"}`} />
              </button>

              <button className="p-2 rounded-full hover:bg-white/10" onClick={(e) => e.preventDefault()}>
                <i className="fas fa-step-forward" />
              </button>

              {/* Waveform */}
              <div className="hidden md:flex items-center space-x-1 h-8">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="waveform-bar w-1 rounded-full bg-teal-400/80"
                    style={{
                      // @ts-ignore
                      "--i": i,
                      height: i === 2 ? "2rem" : i === 1 || i === 3 ? "1.25rem" : "0.75rem",
                      animationPlayState: isPlaying ? "running" : "paused"
                    }}
                  />
                ))}
              </div>
            </div>

            {/* Timer & Mixer */}
            <div className="flex items-center space-x-6">
              {/* Timer (static UI for now) */}
              <div className="flex items-center space-x-3">
                <div className="text-center">
                  <div className="text-lg font-mono font-semibold">25:00</div>
                  <div className="text-xs text-gray-400">Focus</div>
                </div>
                <button className="p-2 rounded-lg hover:bg-white/10">
                  <i className="fas fa-clock" />
                </button>
              </div>

              {/* Mixer */}
              <div className="hidden md:block">
                <div className="flex items-center space-x-2">
                  <i className="fas fa-music text-sm text-gray-400" />
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(musicVol * 100)}
                    onChange={(e) => setMusicVol(parseInt(e.target.value, 10) / 100)}
                    className="w-24 h-1 bg-white/10 rounded-lg"
                  />
                  <i className="fas fa-tree text-sm text-gray-400" />
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(natureVol * 100)}
                    onChange={(e) => setNatureVol(parseInt(e.target.value, 10) / 100)}
                    className="w-24 h-1 bg-white/10 rounded-lg"
                  />
                </div>
                <div className="text-xs text-gray-400 mt-1">Music vs Nature</div>
              </div>

              {/* Focus Mode (UI only for now) */}
              <button className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10">
                <i className="fas fa-eye-slash" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}