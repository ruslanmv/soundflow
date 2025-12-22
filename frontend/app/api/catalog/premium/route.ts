import { NextResponse } from "next/server";

const PYTHON_API_URL = process.env.PYTHON_API_URL;

export async function GET(req: Request) {
  if (!PYTHON_API_URL) {
    return NextResponse.json(
      { error: "PYTHON_API_URL not configured" },
      { status: 503 }
    );
  }

  // Proxy to backend premium catalog endpoint
  const upstream = await fetch(`${PYTHON_API_URL}/catalog/premium`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
