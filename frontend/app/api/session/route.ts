import { NextResponse } from "next/server";
import type { CreateSessionInput } from "@/lib/types";

const PYTHON_API_URL = process.env.PYTHON_API_URL;

export async function POST(req: Request) {
  const body = (await req.json()) as CreateSessionInput;

  if (PYTHON_API_URL) {
    try {
      const pyRes = await fetch(`${PYTHON_API_URL}/generate-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (pyRes.ok) return NextResponse.json(await pyRes.json());
    } catch {}
  }

  const durationMin = Number(body.durationMin ?? 50);
  const energy = Number(body.energy ?? 50);
  const nature = body.nature ?? "Rain";

  const plan = {
    tempoBpm: Math.round(70 + (energy / 100) * 60),
    key: "C minor",
    instrumentation: "synths, minimal piano, no vocals",
    ambiencePrompt: `${nature.toLowerCase()} ambience`,
    durationSec: Math.max(60, durationMin * 60),
  };

  return NextResponse.json({
    id: `sess_${Math.random().toString(16).slice(2)}`,
    plan,
    musicUrl: "https://cdn.pixabay.com/download/audio/2022/03/15/audio_2b8b2b51aa.mp3?filename=ambient-11090.mp3",
    ambienceUrl: "https://cdn.pixabay.com/download/audio/2022/02/23/audio_0f1c2f3c28.mp3?filename=rain-ambient-11060.mp3",
    durationSec: plan.durationSec,
  });
}
