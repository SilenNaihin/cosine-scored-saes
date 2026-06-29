import { useMemo, useState } from "react";
import { PROBING, CHARTS, HEADLINE, C, type Topk } from "@/data";
import { Figure, Num, Toggle } from "@/components/ui";
import { CountUp, AnimatedBar } from "@/components/motion";
import { cn } from "@/lib/utils";

/* The headline probing result as one instrument. Three linked views of the same
   +14.9% win: (1) the scoreboard, where restricting to shared features collapses the
   gap; (2) the per-dataset race (wins 7 of 8); (3) where the win comes from — mostly
   features only the cosine encoder learns. Aggregate + per-dataset accuracies are a
   paper-pinned SAEBench sparse-probing numbers used in Table 1 and the paper
   figures; the matched-vs-full decomposition is CHARTS.DISCOVERY_SEP. */

type K = "top-1" | "top-2" | "top-5";
type Mode = "full" | "shared" | "unique";
const KKEY: Record<K, keyof Topk> = { "top-1": "top_1", "top-2": "top_2", "top-5": "top_5" };
const KIDX: Record<K, number> = { "top-1": 0, "top-2": 1, "top-5": 2 };
const SHARED = "#bdbab2"; // pale gray = shared / matched capacity
const AXMAX = 0.95;
const pctOf = (v: number) => (v / AXMAX) * 100;
const acc = (t: Topk, k: K) => (t[KKEY[k]] ?? t.top_1) as number;
const fmtAcc = (v: number) => `${(v * 100).toFixed(1)}%`;

const MODE_LABEL: Record<Mode, string> = {
  full: "full dictionary",
  shared: "shared features only",
  unique: "unique contribution",
};

export function ProbeWin() {
  const [k, setK] = useState<K>("top-1");
  const [mode, setMode] = useState<Mode>("full");
  const [hover, setHover] = useState<number | null>(null);

  const std = acc(PROBING.aggregate.standard, k);
  const cosFull = acc(PROBING.aggregate.cosine, k);
  const row = CHARTS.DISCOVERY_SEP.rows[KIDX[k]] as [string, number, number, number, number];
  const [, , matchedGapPp, uniquePct, sharedPct] = row;
  const fullGapPp = (cosFull - std) * 100;
  const cosShared = std + matchedGapPp / 100;

  const cosShown = mode === "shared" ? cosShared : cosFull;
  const gapShown = mode === "shared" ? matchedGapPp : fullGapPp;

  // Per-dataset race is always top-1, matching the paper's per-task figure.
  const race = useMemo(() => {
    return PROBING.datasets
      .map((d, i) => ({ d, i, s: acc(d.standard, "top-1"), c: acc(d.cosine, "top-1") }))
      .sort((a, b) => b.c - b.s - (a.c - a.s));
  }, []);
  const wins = race.filter((r) => r.c > r.s).length;

  const headline = (() => {
    if (hover !== null) {
      const r = race[hover];
      const dpp = (r.c - r.s) * 100;
      return r.c >= r.s ? (
        <>Cosine wins <b>{r.d.label}</b> ({r.d.name}) by <Num>+{dpp.toFixed(1)} pp</Num>.</>
      ) : (
        <>Standard wins <b>{r.d.label}</b> ({r.d.name}) by <Num>+{Math.abs(dpp).toFixed(1)} pp</Num> — magnitude carries the signal here.</>
      );
    }
    if (mode === "shared")
      return <>Restricted to features <i>both</i> encoders learn, the gap nearly vanishes — <Num>+{matchedGapPp.toFixed(1)} pp</Num>.</>;
    if (mode === "unique")
      return <>Most of the win — <Num>{uniquePct}%</Num> — comes from features only the cosine encoder learns.</>;
    return <>Across the full dictionary the cosine probe reads the concept <Num>+{fullGapPp.toFixed(1)} pp</Num> more often, and wins <Num>{wins} of {race.length}</Num> datasets.</>;
  })();

  return (
    <Figure
      n={3}
      wide
      caption={
        <>
          <b>Where the probing win comes from.</b> A one-feature linear probe reads a concept off a
          cosine feature <Num>+{fullGapPp.toFixed(1)} pp</Num> more often than off a standard one, at matched
          reconstruction. Switch to <i>shared features only</i> and the gap collapses to{" "}
          <Num>+{matchedGapPp.toFixed(1)} pp</Num>: the advantage is mostly <i>discovery</i> — features the
          inner-product encoder never learns. Aggregate and per-dataset accuracies are the paper-pinned
          Qwen3-8B L18, 500M SAEBench sparse-probing numbers.
        </>
      }
    >
      <div className="p-4 sm:p-5">
        {/* controls */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mb-4">
          <Toggle options={["top-1", "top-2", "top-5"]} value={k} onChange={(v) => setK(v as K)} label="probe" />
          <Toggle options={["full", "shared", "unique"]} value={mode} onChange={(v) => setMode(v as Mode)} label="dictionary" />
          <span className="ml-auto inline-flex items-center gap-1.5 rounded-full border border-rule bg-wash/70 px-2.5 py-1 font-sans text-[12px] text-soft">
            <span className="size-1.5 rounded-full bg-cos" /> matched reconstruction · FVE ≈ {HEADLINE.headline.fve}
          </span>
        </div>

        {/* ── Part 1 — scoreboard ─────────────────────────────────────── */}
        <div className="rounded-md border border-rule bg-white/50 px-4 py-4">
          <div className="font-sans text-[12px] uppercase tracking-[0.14em] text-muted mb-3">
            Sparse-probing accuracy · {k} · {MODE_LABEL[mode]}
          </div>
          <ScoreRow label="Standard" sub="inner product" value={std} color={C.standard} />
          <div className="h-2" />
          <ScoreRow
            label="Cosine"
            sub="direction-scored"
            value={cosShown}
            color={C.cosine}
            base={mode === "unique" ? std : undefined}
            sharedTo={mode === "unique" ? cosShared : undefined}
          />
          {/* gap bracket */}
          <div className="mt-3 flex items-center gap-2 pl-[112px]">
            <span className="inline-block h-px w-6 bg-cos" />
            <span className="font-sans text-[13px] font-semibold text-cos">
              +<CountUp value={gapShown} format={(v) => v.toFixed(1)} /> pp
            </span>
            <span className="font-sans text-[12.5px] text-muted">{headline}</span>
          </div>
        </div>

        <div className="grid md:grid-cols-[1.35fr_1fr] gap-4 mt-4">
          {/* ── Part 2 — dataset race ─────────────────────────────────── */}
          <div className="rounded-md border border-rule bg-white/50 px-4 py-3">
            <div className="font-sans text-[12px] uppercase tracking-[0.14em] text-muted mb-2.5">
              Per-dataset top-1 · cosine wins {wins} of {race.length}
            </div>
            <div className="space-y-1">
              {race.map((r, ri) => (
                <RaceRow
                  key={r.d.name}
                  label={r.d.label}
                  name={r.d.name}
                  s={r.s}
                  c={r.c}
                  active={hover === ri}
                  onHover={(on) => setHover(on ? ri : null)}
                />
              ))}
            </div>
          </div>

          {/* ── Part 3 — attribution ──────────────────────────────────── */}
          <div className="rounded-md border border-rule bg-white/50 px-4 py-3 flex flex-col">
            <div className="font-sans text-[12px] uppercase tracking-[0.14em] text-muted mb-2.5">
              Where the {k} gain comes from
            </div>
            <div className="flex h-9 w-full overflow-hidden rounded border border-rule">
              <div
                className={cn("flex items-center justify-center transition-all", mode === "unique" && "ring-2 ring-cos ring-inset")}
                style={{ width: `${uniquePct}%`, background: C.cosine }}
              >
                <span className="font-mono text-[12px] text-white">{uniquePct}%</span>
              </div>
              <div className="flex items-center justify-center" style={{ width: `${sharedPct}%`, background: SHARED }}>
                {sharedPct >= 12 && <span className="font-mono text-[12px] text-ink/70">{sharedPct}%</span>}
              </div>
            </div>
            <div className="mt-2 flex flex-col gap-1 font-sans text-[12.5px] text-soft">
              <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-[2px]" style={{ background: C.cosine }} /> cosine-only features (discovery)</span>
              <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-[2px]" style={{ background: SHARED }} /> shared-feature improvement</span>
            </div>
            <p className="mt-auto pt-3 text-[12.5px] font-sans text-muted leading-relaxed">
              Of the +{fullGapPp.toFixed(1)} pp gap, only {(matchedGapPp).toFixed(1)} pp survives when both encoders
              are restricted to the 8,661 features they share.
            </p>
          </div>
        </div>
      </div>
    </Figure>
  );
}

/* A scoreboard row. In "unique" mode the cosine bar shows its gain split into a
   shared (gray, up to `sharedTo`) and a unique (purple) segment above `base`. */
function ScoreRow({
  label,
  sub,
  value,
  color,
  base,
  sharedTo,
}: {
  label: string;
  sub: string;
  value: number;
  color: string;
  base?: number;
  sharedTo?: number;
}) {
  const split = base !== undefined && sharedTo !== undefined;
  return (
    <div className="flex items-center gap-3">
      <div className="w-[100px] shrink-0">
        <div className="font-sans font-semibold text-[14px] leading-tight" style={{ color }}>{label}</div>
        <div className="font-sans text-[11px] text-muted leading-tight">{sub}</div>
      </div>
      <div className="relative flex-1 h-7 rounded bg-wash overflow-hidden">
        {split ? (
          <>
            {/* base (shared) up to sharedTo */}
            <AnimatedBar pct={pctOf(sharedTo!)} color={color} className="absolute inset-y-0 left-0 opacity-45" rounded={false} />
            {/* unique segment from sharedTo..value */}
            <div
              className="absolute inset-y-0 transition-all"
              style={{ left: `${pctOf(sharedTo!)}%`, width: `${pctOf(value) - pctOf(sharedTo!)}%`, background: color }}
            />
            {/* standard marker */}
            <div className="absolute inset-y-0 w-px bg-ink/40" style={{ left: `${pctOf(base!)}%` }} />
          </>
        ) : (
          <AnimatedBar pct={pctOf(value)} color={color} className="absolute inset-y-0 left-0" rounded={false} />
        )}
      </div>
      <span className="w-14 shrink-0 text-right font-mono text-[13px]" style={{ color }}>
        <CountUp value={value} format={fmtAcc} />
      </span>
    </div>
  );
}

function RaceRow({
  label,
  name,
  s,
  c,
  active,
  onHover,
}: {
  label: string;
  name: string;
  s: number;
  c: number;
  active: boolean;
  onHover: (on: boolean) => void;
}) {
  const win = c >= s;
  const dpp = (c - s) * 100;
  return (
    <div
      className={cn("flex items-center gap-2.5 rounded px-1.5 py-1 transition-colors cursor-default", active && "bg-cos/[0.06]")}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
    >
      <span className="w-[88px] shrink-0 font-sans text-[12.5px] text-soft truncate" title={name}>{label}</span>
      <div className="relative flex-1 h-4">
        {/* standard bar (thin, top) */}
        <div className="absolute left-0 top-0 h-1.5 rounded-sm" style={{ width: `${pctOf(s)}%`, background: C.standard }} />
        {/* cosine bar (thin, bottom) */}
        <div className="absolute left-0 bottom-0 h-1.5 rounded-sm" style={{ width: `${pctOf(c)}%`, background: C.cosine }} />
      </div>
      <span className={cn("w-12 shrink-0 text-right font-mono text-[12px]", win ? "text-cos" : "text-soft")}>
        {win ? "+" : "−"}{Math.abs(dpp).toFixed(1)}
      </span>
    </div>
  );
}
