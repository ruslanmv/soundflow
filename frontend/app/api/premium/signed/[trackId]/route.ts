import { NextResponse } from "next/server";

const PYTHON_API_URL = process.env.PYTHON_API_URL;
const PREMIUM_API_KEY = process.env.PREMIUM_API_KEY;

export async function GET(
  req: Request,
  { params }: { params: Promise<{ trackId: string }> }
) {
  if (!PYTHON_API_URL) {
    return NextResponse.json(
      { error: "PYTHON_API_URL not configured" },
      { status: 503 }
    );
  }

  if (!PREMIUM_API_KEY) {
    return NextResponse.json(
      { error: "PREMIUM_API_KEY not configured" },
      { status: 503 }
    );
  }

  const { trackId } = await params;

  // Proxy to backend premium signed URL endpoint
  const upstream = await fetch(
    `${PYTHON_API_URL}/tracks/premium/${trackId}/signed`,
    {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "x-premium-key": PREMIUM_API_KEY,
      },
    }
  );

  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
