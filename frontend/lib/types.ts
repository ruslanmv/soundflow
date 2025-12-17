export type Tier = "free" | "premium";

export type CreateSessionInput = {
  goal: string;
  durationMin: number;
  energy: number; // 0..100
  ambience: number; // 0..100
  nature: string;
};

export type SessionPlan = {
  tempoBpm: number;
  key: string;
  instrumentation: string;
  ambiencePrompt: string;
  durationSec: number;
};

export type CreateSessionResponse = {
  id: string;
  plan: SessionPlan;
  musicUrl: string;
  ambienceUrl: string;
  durationSec: number;
};

/**
 * Premium daily contract:
 * backend returns signed URLs (or public) for "today's track"
 */
export type PremiumDailyResponse = {
  id: string;
  title: string;
  durationSec: number;
  musicUrl: string;
  ambienceUrl?: string;
  // optional metadata
  tags?: string[];
  createdAt?: string; // ISO
};
