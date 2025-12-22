"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  User,
  Crown,
  Settings,
  Music,
  BarChart3,
  Bell,
  Lock,
  Mail,
  Key,
  Info,
} from "lucide-react";

// ============================================================================
// PROFILE PAGE (Auth Disabled, Ready for Future)
// ============================================================================

export default function ProfilePage() {
  const [authEnabled, setAuthEnabled] = useState(false);

  // Mock user data (will be replaced with real auth later)
  const mockUser = {
    name: "Guest User",
    email: "guest@soundflow.app",
    tier: "free",
    joinedDate: "2025-01-01",
    totalListeningTime: 0,
    favoriteCategory: "Deep Work",
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-800">
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
                <a href="/premium" className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors duration-200 flex items-center gap-1">
                  <Crown className="w-4 h-4 text-yellow-500" />
                  Premium
                </a>
                <a href="/profile" className="text-blue-600 dark:text-blue-400 font-semibold flex items-center gap-1">
                  <User className="w-4 h-4" />
                  Profile
                </a>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <a href="/profile" className="hidden md:block px-4 py-2 rounded-full bg-white/50 dark:bg-white/5 hover:bg-white/70 dark:hover:bg-white/10 border border-gray-200 dark:border-white/10 transition-all duration-200">
                <User className="w-4 h-4 inline mr-2" />
                Profile
              </a>
            </div>
          </div>
        </nav>
      </header>

      <div className="p-6">
        <div className="max-w-5xl mx-auto space-y-6">
          {/* Header */}
          <div className="flex items-center gap-4">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
            <User className="w-10 h-10 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">{mockUser.name}</h1>
            <p className="text-muted-foreground">{mockUser.email}</p>
            <Badge variant={mockUser.tier === "premium" ? "default" : "secondary"} className="mt-2">
              {mockUser.tier === "premium" ? (
                <>
                  <Crown className="w-3 h-3 mr-1" />
                  Premium
                </>
              ) : (
                "Free Tier"
              )}
            </Badge>
          </div>
        </div>

        {/* Auth Disabled Notice */}
        <Card className="border-yellow-500/50 bg-yellow-50 dark:bg-yellow-900/20">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <Lock className="w-5 h-5 text-yellow-600 mt-0.5" />
              <div className="flex-1">
                <h3 className="font-semibold mb-1">Authentication Currently Disabled</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  This profile page is ready for future authentication integration.
                  Sign-in and user management features will be enabled soon.
                </p>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" disabled>
                    <Mail className="w-4 h-4 mr-2" />
                    Sign In (Coming Soon)
                  </Button>
                  <Button size="sm" variant="outline" disabled>
                    <Key className="w-4 h-4 mr-2" />
                    Sign Up (Coming Soon)
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Profile Tabs */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">
              <User className="w-4 h-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="settings">
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </TabsTrigger>
            <TabsTrigger value="stats">
              <BarChart3 className="w-4 h-4 mr-2" />
              Statistics
            </TabsTrigger>
            <TabsTrigger value="subscription">
              <Crown className="w-4 h-4 mr-2" />
              Subscription
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Account Information</CardTitle>
                <CardDescription>Your profile details and account status</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Full Name</Label>
                    <Input value={mockUser.name} disabled />
                  </div>
                  <div>
                    <Label>Email Address</Label>
                    <Input value={mockUser.email} type="email" disabled />
                  </div>
                  <div>
                    <Label>Account Tier</Label>
                    <Input value={mockUser.tier === "premium" ? "Premium" : "Free"} disabled />
                  </div>
                  <div>
                    <Label>Member Since</Label>
                    <Input value={mockUser.joinedDate} disabled />
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <Button disabled>
                    <Settings className="w-4 h-4 mr-2" />
                    Edit Profile (Auth Required)
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quick Stats</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-3xl font-bold text-purple-600">{mockUser.totalListeningTime}</p>
                    <p className="text-sm text-muted-foreground">Hours Listened</p>
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-blue-600">0</p>
                    <p className="text-sm text-muted-foreground">Favorite Tracks</p>
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-green-600">{mockUser.favoriteCategory}</p>
                    <p className="text-sm text-muted-foreground">Top Category</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Preferences</CardTitle>
                <CardDescription>Customize your SoundFlow experience</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Default Music Category</Label>
                  <Input value={mockUser.favoriteCategory} disabled />
                </div>

                <div className="space-y-2">
                  <Label>Audio Quality</Label>
                  <Input value="High (320kbps)" disabled />
                </div>

                <div className="flex items-center justify-between py-2">
                  <div>
                    <p className="font-medium">Email Notifications</p>
                    <p className="text-sm text-muted-foreground">Receive updates about new tracks</p>
                  </div>
                  <Button variant="outline" size="sm" disabled>
                    <Bell className="w-4 h-4 mr-2" />
                    Configure
                  </Button>
                </div>

                <div className="pt-4 border-t">
                  <p className="text-sm text-muted-foreground flex items-center gap-2">
                    <Info className="w-4 h-4" />
                    Settings will be enabled when authentication is activated
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Statistics Tab */}
          <TabsContent value="stats" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Listening Statistics</CardTitle>
                <CardDescription>Track your music consumption over time</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold mb-3">This Week</h3>
                    <div className="grid grid-cols-4 gap-4">
                      <div className="text-center">
                        <p className="text-2xl font-bold">0</p>
                        <p className="text-xs text-muted-foreground">Sessions</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold">0</p>
                        <p className="text-xs text-muted-foreground">Hours</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold">0</p>
                        <p className="text-xs text-muted-foreground">Tracks</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold">-</p>
                        <p className="text-xs text-muted-foreground">Avg/Day</p>
                      </div>
                    </div>
                  </div>

                  <div className="pt-4 border-t">
                    <h3 className="font-semibold mb-3">Most Played Categories</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center py-2">
                        <span className="text-sm">No data yet</span>
                        <Badge variant="outline">0 plays</Badge>
                      </div>
                    </div>
                  </div>

                  <div className="pt-4 border-t">
                    <p className="text-sm text-muted-foreground flex items-center gap-2">
                      <Info className="w-4 h-4" />
                      Statistics will be tracked once you start using SoundFlow
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Subscription Tab */}
          <TabsContent value="subscription" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Subscription Plan</CardTitle>
                <CardDescription>Manage your SoundFlow subscription</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="font-semibold mb-3">Current Plan: Free</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <Music className="w-4 h-4 text-green-500" />
                      <span>Access to free catalog (synthesis-based tracks)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Music className="w-4 h-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Premium AI-generated tracks (locked)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Music className="w-4 h-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Binaural beats enhancement (locked)</span>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h3 className="font-semibold mb-3">Upgrade to Premium</h3>
                  <Card className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border-purple-200">
                    <CardContent className="pt-6">
                      <div className="flex items-start justify-between">
                        <div>
                          <h4 className="font-semibold text-lg">Premium Plan</h4>
                          <ul className="mt-2 space-y-1 text-sm">
                            <li className="flex items-center gap-2">
                              <Crown className="w-4 h-4 text-yellow-500" />
                              AI-generated music (MusicGen Stereo Large)
                            </li>
                            <li className="flex items-center gap-2">
                              <Crown className="w-4 h-4 text-yellow-500" />
                              Binaural beats for focus/meditation
                            </li>
                            <li className="flex items-center gap-2">
                              <Crown className="w-4 h-4 text-yellow-500" />
                              Professional mastering (320kbps)
                            </li>
                            <li className="flex items-center gap-2">
                              <Crown className="w-4 h-4 text-yellow-500" />
                              Unlimited playback & downloads
                            </li>
                          </ul>
                        </div>
                        <div className="text-right">
                          <p className="text-3xl font-bold">$9.99</p>
                          <p className="text-sm text-muted-foreground">/month</p>
                        </div>
                      </div>
                      <Button className="w-full mt-4" disabled>
                        <Crown className="w-4 h-4 mr-2" />
                        Upgrade (Coming Soon)
                      </Button>
                    </CardContent>
                  </Card>
                </div>

                <div className="pt-4 border-t">
                  <p className="text-sm text-muted-foreground flex items-center gap-2">
                    <Info className="w-4 h-4" />
                    Subscriptions will be available when payment integration is added
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
        </div>
      </div>
    </div>
  );
}
