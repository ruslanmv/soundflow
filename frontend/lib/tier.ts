import type { Tier } from "@/lib/types";

/**
 * Production-ready tier gating scaffold.
 *
 * Today:
 * - default: free
 * - allow testing: ?tier=premium
 * - allow dev override: localStorage "sf_tier" = "premium"
 *
 * Later (real prod):
 * - replace with cookie/session from your auth provider
 * - or call /api/me and read entitlement from backend/Stripe
 */
export function getClientTier(): Tier {
  if (typeof window === "undefined") return "free";

  const url = new URL(window.location.href);
  const q = url.searchParams.get("tier");
  if (q === "premium") return "premium";

  const stored = window.localStorage.getItem("sf_tier");
  if (stored === "premium") return "premium";

  return "free";
}

export function setClientTier(tier: Tier) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem("sf_tier", tier);
}

export function clearClientTierOverride() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem("sf_tier");
}
