import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtPct(x: number, digits = 1) {
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtNum(x: number, digits = 2) {
  return x.toLocaleString(undefined, { maximumFractionDigits: digits });
}

/** map an activation (relative to feature max) to a violet intensity 0..1 */
export function actIntensity(act: number, maxAct: number) {
  if (maxAct <= 0) return 0;
  return Math.max(0, Math.min(1, act / maxAct));
}
