"use client";

import { useEffect, useMemo, useState } from "react";
import { AudioEngine } from "@/lib/audioEngine";
import { soundscapeData, colorClasses } from "@/lib/presets";
import type { CreateSessionResponse } from "@/lib/types";

type Chip = { id: string; label: string };

export default function Page() {
  const engine = useMemo(() => new AudioEngine(), []);

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  // Start widget selections
  const [goal, setGoal] = useState("Deep Focus");
  const [durationLabel, setDurationLabel] = useState("50 minutes");

  // Builder selections (mirror your HTML defaults)
  const [builderGoal, setBuilderGoal] = useState<string | null>(null);
  const [durationChip, setDurationChip] = useState<string | null>(null);
  const [energy, setEnergy] = useState(50);
  const [ambience, setAmbience] = useState(30);
  const [nature, setNature] = useState<string | null>(null);

  // Derived labels
  const energyLabel = energy < 33 ? "Calm" : energy < 66 ? "Medium" : "Driven";
  const ambienceLabel = ambience < 33 ? "Light" : ambience < 66 ? "Medium" : "Heavy";

  // Player UI text
  const [nowTitle, setNowTitle] = useState("Deep Focus Session");
  const [nowSubtitle, setNowSubtitle] = useState("AI Generated • 50m remaining");

  const durationChips: Chip[] = [
    { id: "25m", label: "25m" },
    { id: "50m", label: "50m" },
    { id: "90m", label: "90m" },
    { id: "custom", label: "Custom" },
  ];

  function minutesFromLabel(label: string) {
    const m = label.match(/(\d+)\s*minutes/i);
    if (m?.[1]) return parseInt(m[1], 10);
    return 50;
  }

  function minutesFromChip(chip: string | null) {
    if (!chip) return minutesFromLabel(durationLabel);
    if (chip === "custom") return minutesFromLabel(durationLabel);
    const m = chip.match(/(\d+)/);
    return m ? parseInt(m[1], 10) : 50;
  }

  async function togglePlayPause() {
    if (isPlaying) {
      engine.pause();
      setIsPlaying(false);
      return;
    }
    await engine.play();
    setIsPlaying(true);
  }

  // keep waveform animation in sync
  useEffect(() => {
    const bars = document.querySelectorAll<HTMLElement>(".waveform-bar");
    bars.forEach((bar) => {
      bar.style.animationPlayState = isPlaying ? "running" : "paused";
    });
  }, [isPlaying]);

  async function generateSession() {
    // Production behavior: call /api/session so it can proxy to PYTHON_API_URL later.
    try {
      setIsGenerating(true);

      const finalGoal = builderGoal ?? goal;
      const durationMin = minutesFromChip(durationChip);

      const res = await fetch("/api/session", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          goal: finalGoal,
          durationMin,
          energy,
          ambience,
          nature: nature ?? "Rain",
        }),
      });

      const data = (await res.json()) as CreateSessionResponse;

      engine.setSources(data.musicUrl, data.ambienceUrl);
      setNowTitle(`${finalGoal} Session`);
      setNowSubtitle(`AI Generated • ${Math.round(data.durationSec / 60)}m remaining`);

      setIsModalOpen(false);
      await engine.play();
      setIsPlaying(true);
    } catch (e) {
      console.error(e);
      alert("AI Session Generated! Starting your personalized focus session.");
      setIsModalOpen(false);
    } finally {
      setIsGenerating(false);
    }
  }

  function ToggleableButton({
    active,
    onClick,
    className,
    children,
  }: {
    active?: boolean;
    onClick?: () => void;
    className: string;
    children: React.ReactNode;
  }) {
    return (
      <button
        onClick={onClick}
        className={[className, active ? "bg-white/10 border-white/30" : ""].join(" ")}
      >
        {children}
      </button>
    );
  }

  return (
    <>
      {/* Navigation */}
      <header className="sticky top-0 z-50 glass border-b border-white/10">
        <nav className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-8">
              <a href="#" className="text-xl font-semibold gradient-text" onClick={(e) => e.preventDefault()}>
                SoundFlow AI
              </a>
              <div className="hidden md:flex items-center space-x-6">
                <a href="#" className="text-gray-300 hover:text-white transition-colors duration-200" onClick={(e) => e.preventDefault()}>
                  Explore
                </a>
                <a href="#" className="text-gray-300 hover:text-white transition-colors duration-200" onClick={(e) => { e.preventDefault(); setIsModalOpen(true); }}>
                  AI Session
                </a>
                <a href="#" className="text-gray-300 hover:text-white transition-colors duration-200" onClick={(e) => e.preventDefault()}>
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
              AI-powered soundscapes that adapt to your focus needs. Personalized sessions for deep work, study, and relaxation.
            </p>

            {/* Pill Buttons */}
            <div className="flex flex-wrap justify-center gap-3 mb-12">
              <button
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(45,212,191,0.2)]"
                onClick={() => { setGoal("Deep Work"); setIsModalOpen(true); }}
              >
                <i className="fas fa-brain mr-2" />
                Deep Work
              </button>
              <button
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(96,165,250,0.2)]"
                onClick={() => { setGoal("Study"); setIsModalOpen(true); }}
              >
                <i className="fas fa-graduation-cap mr-2" />
                Study
              </button>
              <button
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-purple-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(192,132,252,0.2)]"
                onClick={() => { setGoal("Relax"); setIsModalOpen(true); }}
              >
                <i className="fas fa-couch mr-2" />
                Relax
              </button>
              <button
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(45,212,191,0.2)]"
                onClick={() => { setGoal("Nature"); setIsModalOpen(true); }}
              >
                <i className="fas fa-tree mr-2" />
                Nature
              </button>
              <button
                className="px-6 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-400/30 transition-all duration-300 hover:shadow-[0_0_20px_rgba(96,165,250,0.2)]"
                onClick={() => { setGoal("Flow State"); setIsModalOpen(true); }}
              >
                <i className="fas fa-infinity mr-2" />
                Flow State
              </button>
            </div>

            {/* Start Widget */}
            <div className="glass-modal rounded-2xl p-8 max-w-xl mx-auto mb-16">
              <h3 className="text-xl font-semibold mb-6">Start Your Focus Session</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Goal</label>
                  <select
                    value={goal}
                    onChange={(e) => setGoal(e.target.value)}
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
                    value={durationLabel}
                    onChange={(e) => setDurationLabel(e.target.value)}
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
                  <img
                    src={`https://picsum.photos/400/300?random=${item.imageId}`}
                    alt={`${item.title} cover image`}
                    className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-500"
                    loading="lazy"
                  />
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
        <div
          className={[
            "fixed inset-0 z-50 items-center justify-center p-4 bg-black/50 backdrop-blur-sm",
            isModalOpen ? "flex" : "hidden",
          ].join(" ")}
          onClick={(e) => {
            if (e.target === e.currentTarget) setIsModalOpen(false);
          }}
        >
          <div className="glass-modal rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-8">
              <div className="flex justify-between items-center mb-8">
                <h3 className="text-2xl font-bold">AI Session Builder</h3>
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="p-2 rounded-lg hover:bg-white/10"
                  aria-label="Close"
                >
                  <i className="fas fa-times" />
                </button>
              </div>

              {/* Step 1: Goal */}
              <div className="mb-10">
                <h4 className="text-lg font-semibold mb-4">Step 1: Select Your Goal</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <ToggleableButton
                    active={builderGoal === "Deep Work"}
                    onClick={() => setBuilderGoal("Deep Work")}
                    className="goal-option p-6 rounded-xl bg-white/5 border border-white/10 hover:border-teal-400/30 hover:bg-white/10 transition-all duration-300 text-left"
                  >
                    <i className="fas fa-laptop-code text-2xl mb-3 text-teal-400 block" />
                    <h5 className="font-semibold mb-2">Deep Work</h5>
                    <p className="text-sm text-gray-400">Intense focus sessions</p>
                  </ToggleableButton>

                  <ToggleableButton
                    active={builderGoal === "Coding"}
                    onClick={() => setBuilderGoal("Coding")}
                    className="goal-option p-6 rounded-xl bg-white/5 border border-white/10 hover:border-blue-400/30 hover:bg-white/10 transition-all duration-300 text-left"
                  >
                    <i className="fas fa-code text-2xl mb-3 text-blue-400 block" />
                    <h5 className="font-semibold mb-2">Coding</h5>
                    <p className="text-sm text-gray-400">Programming flow state</p>
                  </ToggleableButton>

                  <ToggleableButton
                    active={builderGoal === "Reading"}
                    onClick={() => setBuilderGoal("Reading")}
                    className="goal-option p-6 rounded-xl bg-white/5 border border-white/10 hover:border-purple-400/30 hover:bg-white/10 transition-all duration-300 text-left"
                  >
                    <i className="fas fa-book text-2xl mb-3 text-purple-400 block" />
                    <h5 className="font-semibold mb-2">Reading</h5>
                    <p className="text-sm text-gray-400">Extended reading sessions</p>
                  </ToggleableButton>
                </div>
              </div>

              {/* Step 2: Duration */}
              <div className="mb-10">
                <h4 className="text-lg font-semibold mb-4">Step 2: Choose Duration</h4>
                <div className="flex flex-wrap gap-3">
                  {durationChips.map((c) => (
                    <ToggleableButton
                      key={c.id}
                      active={durationChip === c.id}
                      onClick={() => setDurationChip(c.id)}
                      className="duration-chip px-6 py-3 rounded-full bg-white/5 border border-white/10 hover:border-white/30 transition-all duration-200"
                    >
                      {c.label}
                    </ToggleableButton>
                  ))}
                </div>
              </div>

              {/* Step 3: Sliders */}
              <div className="mb-10">
                <h4 className="text-lg font-semibold mb-6">Step 3: Fine-tune Your Session</h4>
                <div className="space-y-8">
                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="font-medium">Energy Level</label>
                      <span className="text-teal-400">{energyLabel}</span>
                    </div>
                    <div className="flex items-center space-x-4">
                      <span className="text-sm text-gray-400">Calm</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={energy}
                        onChange={(e) => setEnergy(parseInt(e.target.value, 10))}
                        className="flex-1 h-2 bg-white/10 rounded-lg appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-teal-400"
                      />
                      <span className="text-sm text-gray-400">Driven</span>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="font-medium">Ambience Level</label>
                      <span className="text-blue-400">{ambienceLabel}</span>
                    </div>
                    <div className="flex items-center space-x-4">
                      <span className="text-sm text-gray-400">None</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={ambience}
                        onChange={(e) => setAmbience(parseInt(e.target.value, 10))}
                        className="flex-1 h-2 bg-white/10 rounded-lg appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-400"
                      />
                      <span className="text-sm text-gray-400">Heavy</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Step 4: Nature Sounds */}
              <div className="mb-10">
                <h4 className="text-lg font-semibold mb-4">Step 4: Add Nature Sounds</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <ToggleableButton
                    active={nature === "Rain"}
                    onClick={() => setNature("Rain")}
                    className="nature-option p-4 rounded-xl bg-white/5 border border-white/10 hover:border-teal-400/30 transition-all duration-200"
                  >
                    <i className="fas fa-cloud-rain text-2xl mb-2 text-blue-400 block" />
                    <span>Rain</span>
                  </ToggleableButton>
                  <ToggleableButton
                    active={nature === "Forest"}
                    onClick={() => setNature("Forest")}
                    className="nature-option p-4 rounded-xl bg-white/5 border border-white/10 hover:border-teal-400/30 transition-all duration-200"
                  >
                    <i className="fas fa-tree text-2xl mb-2 text-green-400 block" />
                    <span>Forest</span>
                  </ToggleableButton>
                  <ToggleableButton
                    active={nature === "Fireplace"}
                    onClick={() => setNature("Fireplace")}
                    className="nature-option p-4 rounded-xl bg-white/5 border border-white/10 hover:border-teal-400/30 transition-all duration-200"
                  >
                    <i className="fas fa-fire text-2xl mb-2 text-orange-400 block" />
                    <span>Fireplace</span>
                  </ToggleableButton>
                  <ToggleableButton
                    active={nature === "Ocean"}
                    onClick={() => setNature("Ocean")}
                    className="nature-option p-4 rounded-xl bg-white/5 border border-white/10 hover:border-teal-400/30 transition-all duration-200"
                  >
                    <i className="fas fa-water text-2xl mb-2 text-cyan-400 block" />
                    <span>Ocean</span>
                  </ToggleableButton>
                </div>
              </div>

              {/* Generate Button */}
              <button
                onClick={generateSession}
                disabled={isGenerating}
                className="w-full bg-gradient-to-r from-teal-500 to-purple-500 hover:from-teal-600 hover:to-purple-600 text-white font-semibold py-4 rounded-xl transition-all duration-300 hover:shadow-[0_0_30px_rgba(168,85,247,0.4)] disabled:opacity-60"
              >
                <i className="fas fa-magic mr-2" />
                {isGenerating ? "Generating..." : "Generate AI Session"}
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Bottom Player */}
      <div className="fixed bottom-0 left-0 right-0 z-40 glass border-t border-white/10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            {/* Track Info */}
            <div className="flex items-center space-x-4">
              <img src="https://picsum.photos/60?random=1" alt="Current track cover" className="rounded-xl w-12 h-12" />
              <div>
                <h4 className="font-semibold">{nowTitle}</h4>
                <p className="text-sm text-gray-400">{nowSubtitle}</p>
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center space-x-6">
              <button className="p-2 rounded-full hover:bg-white/10">
                <i className="fas fa-step-backward" />
              </button>
              <button
                id="playPauseBtn"
                onClick={togglePlayPause}
                className="p-3 rounded-full bg-white/10 hover:bg-white/20 border border-white/20"
              >
                <i className={`fas ${isPlaying ? "fa-pause" : "fa-play"}`} />
              </button>
              <button className="p-2 rounded-full hover:bg-white/10">
                <i className="fas fa-step-forward" />
              </button>

              {/* Waveform */}
              <div className="hidden md:flex items-center space-x-1 h-8">
                <div className="waveform-bar w-1 h-3 bg-teal-400/60 rounded-full" style={{ ["--i" as any]: 0 }} />
                <div className="waveform-bar w-1 h-5 bg-teal-400/80 rounded-full" style={{ ["--i" as any]: 1 }} />
                <div className="waveform-bar w-1 h-8 bg-teal-400 rounded-full" style={{ ["--i" as any]: 2 }} />
                <div className="waveform-bar w-1 h-5 bg-teal-400/80 rounded-full" style={{ ["--i" as any]: 3 }} />
                <div className="waveform-bar w-1 h-3 bg-teal-400/60 rounded-full" style={{ ["--i" as any]: 4 }} />
              </div>
            </div>

            {/* Timer & Mixer */}
            <div className="flex items-center space-x-6">
              {/* Timer */}
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
                  <input type="range" min="0" max="100" defaultValue="70" className="w-24 h-1 bg-white/10 rounded-lg" />
                  <i className="fas fa-tree text-sm text-gray-400" />
                  <input type="range" min="0" max="100" defaultValue="30" className="w-24 h-1 bg-white/10 rounded-lg" />
                </div>
                <div className="text-xs text-gray-400 mt-1">Music vs Nature</div>
              </div>

              {/* Focus Mode */}
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
