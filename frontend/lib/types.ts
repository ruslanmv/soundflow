export type CreateSessionInput = { goal: string; durationMin: number; energy: number; ambience: number; nature: string; };
export type SessionPlan = { tempoBpm: number; key: string; instrumentation: string; ambiencePrompt: string; durationSec: number; };
export type CreateSessionResponse = { id: string; plan: SessionPlan; musicUrl: string; ambienceUrl: string; durationSec: number; };
