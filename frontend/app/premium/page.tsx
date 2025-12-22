"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Sparkles,
  Music2,
  Brain,
  Zap,
  CheckCircle2,
  Lock,
  PlayCircle,
  Pause,
  Download,
  Loader2,
  Crown,
  Headphones,
} from "lucide-react";

// ============================================================================
// TYPES
// ============================================================================

interface PremiumTrack {
  id: string;
  title: string;
  tier: string;
  date: string;
  category: string;
  genre?: string;
  bpm?: number;
  key?: string;
  durationSec: number;
  goalTags?: string[];
  natureTags?: string[];
  energyMin?: number;
  energyMax?: number;
  url?: string | null;
}

interface SignedUrlResponse {
  signedUrl: string;
  expiresInSec: number;
}

// ============================================================================
// PREMIUM FEATURES
// ============================================================================

const PREMIUM_FEATURES = [
  {
    icon: Sparkles,
    title: "AI-Generated Music",
    description: "MusicGen Stereo Large (3.3B parameters) for highest quality",
    color: "text-purple-500",
  },
  {
    icon: Brain,
    title: "Binaural Beats",
    description: "Gamma/Alpha/Theta/Delta waves for focus, creativity, meditation",
    color: "text-blue-500",
  },
  {
    icon: Zap,
    title: "Sidechain Compression",
    description: "EDM-style pumping effect for high-energy tracks",
    color: "text-yellow-500",
  },
  {
    icon: Music2,
    title: "Professional Mastering",
    description: "-14 LUFS loudness, true peak limiting, 320kbps MP3",
    color: "text-green-500",
  },
];

const CATEGORIES = [
  { id: "all", label: "All Categories", icon: Music2 },
  { id: "Deep Work", label: "Deep Work", icon: Brain },
  { id: "Study", label: "Study", icon: Headphones },
  { id: "Relax", label: "Relax", icon: Sparkles },
  { id: "Nature", label: "Nature", icon: Crown },
  { id: "Flow State", label: "Flow State", icon: Zap },
];

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function PremiumPage() {
  const [tracks, setTracks] = useState<PremiumTrack[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [playingTrackId, setPlayingTrackId] = useState<string | null>(null);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const [isPremiumUser, setIsPremiumUser] = useState(false); // Auth disabled for now

  // -------------------------------------------------------------------------
  // Load Premium Catalog
  // -------------------------------------------------------------------------

  useEffect(() => {
    loadPremiumCatalog();
  }, []);

  const loadPremiumCatalog = async () => {
    try {
      setLoading(true);

      // Try to load from general catalog first (all tracks)
      const response = await fetch("/api/catalog/all");

      if (response.ok) {
        const allTracks = await response.json();
        // Filter for premium tracks only
        const premiumTracks = allTracks.filter((t: PremiumTrack) => t.tier === "premium");
        setTracks(premiumTracks);
      } else {
        // Fallback: load premium-only catalog
        const fallbackResponse = await fetch("/api/catalog/premium");
        if (fallbackResponse.ok) {
          const premiumTracks = await fallbackResponse.json();
          setTracks(premiumTracks);
        }
      }
    } catch (error) {
      console.error("Failed to load premium catalog:", error);
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------------------------------------------------
  // Audio Playback
  // -------------------------------------------------------------------------

  const handlePlay = async (track: PremiumTrack) => {
    if (!isPremiumUser) {
      alert("Premium subscription required to play tracks.\n\nUpgrade to access high-quality AI music!");
      return;
    }

    // Stop current track
    if (audioElement) {
      audioElement.pause();
      setAudioElement(null);
    }

    if (playingTrackId === track.id) {
      setPlayingTrackId(null);
      return;
    }

    try {
      // Get signed URL for premium track
      const response = await fetch(`/api/premium/signed/${track.id}`);

      if (!response.ok) {
        throw new Error("Failed to get signed URL");
      }

      const data: SignedUrlResponse = await response.json();

      // Create and play audio
      const audio = new Audio(data.signedUrl);
      audio.play();

      setAudioElement(audio);
      setPlayingTrackId(track.id);

      // Handle audio end
      audio.onended = () => {
        setPlayingTrackId(null);
        setAudioElement(null);
      };
    } catch (error) {
      console.error("Playback error:", error);
      alert("Failed to play track. Please try again.");
    }
  };

  const handleStop = () => {
    if (audioElement) {
      audioElement.pause();
      setAudioElement(null);
    }
    setPlayingTrackId(null);
  };

  // -------------------------------------------------------------------------
  // Filtering
  // -------------------------------------------------------------------------

  const filteredTracks = selectedCategory === "all"
    ? tracks
    : tracks.filter(t => t.category === selectedCategory);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Navigation */}
      <header className="sticky top-0 z-50 glass border-b border-white/10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md">
        <nav className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-8">
              <a href="/" className="text-xl font-semibold gradient-text">
                SoundFlow AI
              </a>
              <div className="hidden md:flex items-center space-x-6">
                <a href="/" className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors duration-200">
                  Home
                </a>
                <a href="/premium" className="text-purple-600 dark:text-purple-400 font-semibold flex items-center gap-1">
                  <Crown className="w-4 h-4 text-yellow-500" />
                  Premium
                </a>
                <a href="/profile" className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors duration-200">
                  Profile
                </a>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <a href="/profile" className="hidden md:block px-4 py-2 rounded-full bg-white/50 dark:bg-white/5 hover:bg-white/70 dark:hover:bg-white/10 border border-gray-200 dark:border-white/10 transition-all duration-200">
                <i className="fas fa-user mr-2" />
                Profile
              </a>
            </div>
          </div>
        </nav>
      </header>

      <div className="p-6">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-2">
            <Crown className="w-10 h-10 text-yellow-500" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
              Premium Music
            </h1>
          </div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            AI-generated music with professional DSP enhancements for ultimate focus and relaxation
          </p>

          {!isPremiumUser && (
            <Card className="border-yellow-500/50 bg-yellow-50 dark:bg-yellow-900/20">
              <CardContent className="pt-6">
                <div className="flex items-center justify-center gap-3">
                  <Lock className="w-5 h-5 text-yellow-600" />
                  <p className="text-sm font-medium">
                    Premium features are currently locked. Authentication will be enabled soon!
                  </p>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setIsPremiumUser(true)}
                    className="ml-4"
                  >
                    Preview Mode (Dev)
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Premium Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {PREMIUM_FEATURES.map((feature, i) => {
            const Icon = feature.icon;
            return (
              <Card key={i} className="border-purple-200 dark:border-purple-800">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <Icon className={`w-6 h-6 ${feature.color}`} />
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Category Filter */}
        <Card>
          <CardHeader>
            <CardTitle>Browse Premium Tracks</CardTitle>
            <CardDescription>
              {filteredTracks.length} tracks available
              {selectedCategory !== "all" && ` in ${selectedCategory}`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2 flex-wrap">
              {CATEGORIES.map((cat) => {
                const Icon = cat.icon;
                return (
                  <Button
                    key={cat.id}
                    variant={selectedCategory === cat.id ? "default" : "outline"}
                    onClick={() => setSelectedCategory(cat.id)}
                    className="flex items-center gap-2"
                  >
                    <Icon className="w-4 h-4" />
                    {cat.label}
                  </Button>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Track List */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
            <span className="ml-3 text-muted-foreground">Loading premium catalog...</span>
          </div>
        ) : filteredTracks.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <Music2 className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">
                No premium tracks available yet.
                <br />
                Check back soon for new releases!
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredTracks.map((track) => {
              const isPlaying = playingTrackId === track.id;

              return (
                <Card key={track.id} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-lg line-clamp-2">{track.title}</CardTitle>
                        <CardDescription className="mt-1">{track.date}</CardDescription>
                      </div>
                      <Badge variant="secondary">
                        <Crown className="w-3 h-3 mr-1" />
                        Premium
                      </Badge>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {/* Metadata */}
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-muted-foreground">Category:</span>
                        <p className="font-medium">{track.category}</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Duration:</span>
                        <p className="font-medium">{Math.floor(track.durationSec / 60)}:{(track.durationSec % 60).toString().padStart(2, '0')}</p>
                      </div>
                      {track.genre && (
                        <div>
                          <span className="text-muted-foreground">Genre:</span>
                          <p className="font-medium capitalize">{track.genre}</p>
                        </div>
                      )}
                      {track.bpm && (
                        <div>
                          <span className="text-muted-foreground">BPM:</span>
                          <p className="font-medium">{track.bpm}</p>
                        </div>
                      )}
                    </div>

                    {/* Tags */}
                    {track.goalTags && track.goalTags.length > 0 && (
                      <div className="flex gap-1 flex-wrap">
                        {track.goalTags.slice(0, 3).map((tag, i) => (
                          <Badge key={i} variant="outline" className="text-xs">
                            {tag.replace('_', ' ')}
                          </Badge>
                        ))}
                      </div>
                    )}

                    {/* Playback Controls */}
                    <div className="flex gap-2">
                      <Button
                        className="flex-1"
                        variant={isPlaying ? "destructive" : "default"}
                        onClick={() => isPlaying ? handleStop() : handlePlay(track)}
                        disabled={!isPremiumUser && !isPlaying}
                      >
                        {isPlaying ? (
                          <>
                            <Pause className="w-4 h-4 mr-2" />
                            Stop
                          </>
                        ) : (
                          <>
                            <PlayCircle className="w-4 h-4 mr-2" />
                            {isPremiumUser ? "Play" : "Locked"}
                          </>
                        )}
                      </Button>

                      {isPremiumUser && (
                        <Button variant="outline" size="icon">
                          <Download className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}

        {/* Upgrade CTA */}
        {!isPremiumUser && (
          <Card className="border-purple-500/50 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20">
            <CardContent className="py-8">
              <div className="text-center space-y-4">
                <Crown className="w-12 h-12 mx-auto text-yellow-500" />
                <h3 className="text-2xl font-bold">Unlock Premium Music</h3>
                <p className="text-muted-foreground max-w-2xl mx-auto">
                  Get access to AI-generated music with professional DSP enhancements,
                  binaural beats for focus, and unlimited playback.
                </p>
                <Button size="lg" className="mt-4">
                  <Crown className="w-5 h-5 mr-2" />
                  Upgrade to Premium (Coming Soon)
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
        </div>
      </div>
    </div>
  );
}
