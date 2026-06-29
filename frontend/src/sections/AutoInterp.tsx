import { useMemo, useState } from "react";
import { AUTOINTERP, C, type VariantName } from "@/data";
import { Figure, Legend, Num, Toggle } from "@/components/ui";
import { cn } from "@/lib/utils";

/* §5 — Matched 500M describe-then-predict rates. The older 1-5 coherence
   gallery in autointerp.json is not the headline per-feature checkpoint. */

const ROWS = [
  { name: "Standard", color: C.standard, rate: 20.1, low: 22.3, medium: 21.6, high: 14.8 },
  { name: "Global a", color: "#8f8678", rate: 21.3, low: 24.3, medium: 21.4, high: 18.0 },
  { name: "Per-feature", color: C.cosine, rate: 19.2, low: 23.5, medium: 19.5, high: 14.3 },
];

const maxRate = 25;
const fmt = (v: number) => `${v.toFixed(1)}%`;

type Feat = { id: number; concept: string; coherence: number; examplesConsistent: number; maxAct: number };
const V = AUTOINTERP.variants as Record<VariantName, { features: Feat[] }>;
const COL: Record<VariantName, string> = { Standard: C.standard, Cosine: C.cosine };

export function AutoInterp() {
  const [variant, setVariant] = useState<VariantName>("Cosine");
  const [q, setQ] = useState("");
  const [minCoh, setMinCoh] = useState(0);
  const [limit, setLimit] = useState(10);

  const feats = useMemo(() => {
    const term = q.trim().toLowerCase();
    return V[variant].features
      .filter((f) => f.coherence >= minCoh && (!term || f.concept.toLowerCase().includes(term)))
      .sort((a, b) => b.coherence - a.coherence || b.examplesConsistent - a.examplesConsistent);
  }, [variant, q, minCoh]);

  return (
    <Figure
      n={11}
      wide
      caption={
        <>
          <b>Per-feature interpretability is matched.</b> Describe-then-predict scoring at the 500M headline
          setting puts all three variants in a <Num>2.1 pp</Num> band, so the sparse-probing gain is not a
          per-feature legibility gain. The concept list below shows examples from the judged-feature set.
        </>
      }
    >
      <div className="p-4">
        <div className="flex items-center justify-between gap-4 mb-4">
          <div className="font-sans text-[13px] text-soft font-semibold">
            LLM-judged interpretable rate
          </div>
          <Legend items={ROWS.map((r) => ({ label: r.name, color: r.color }))} />
        </div>

        <div className="space-y-3">
          {ROWS.map((r) => (
            <div key={r.name} className="grid grid-cols-[112px_1fr_52px] items-center gap-3 font-sans">
              <div className="text-[13px] font-semibold" style={{ color: r.color }}>{r.name}</div>
              <div className="h-7 rounded bg-wash overflow-hidden">
                <div
                  className="h-full rounded-sm transition-all"
                  style={{ width: `${(r.rate / maxRate) * 100}%`, background: r.color }}
                />
              </div>
              <div className="font-mono text-[13px] text-right" style={{ color: r.color }}>{fmt(r.rate)}</div>
            </div>
          ))}
        </div>

        <div className="mt-5 rounded-md border border-rule bg-white/50 overflow-hidden">
          <table className="w-full text-[13px] font-sans">
            <thead className="text-muted border-b border-rule">
              <tr>
                <th className="text-left font-medium px-3 py-2">Variant</th>
                <th className="text-right font-medium px-3 py-2">Low freq.</th>
                <th className="text-right font-medium px-3 py-2">Medium freq.</th>
                <th className="text-right font-medium px-3 py-2">High freq.</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => (
                <tr key={r.name} className="border-b border-rule/60 last:border-0">
                  <td className="px-3 py-2 font-semibold" style={{ color: r.color }}>{r.name}</td>
                  <td className="px-3 py-2 text-right font-mono">{fmt(r.low)}</td>
                  <td className="px-3 py-2 text-right font-mono">{fmt(r.medium)}</td>
                  <td className="px-3 py-2 text-right font-mono">{fmt(r.high)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-5 border-t border-rule pt-5">
          <div className="flex flex-wrap items-center gap-3 mb-3 font-sans text-[13px]">
            <Toggle
              options={["Standard", "Cosine"]}
              value={variant}
              onChange={(v) => { setVariant(v as VariantName); setLimit(10); }}
              label="Concepts"
            />
            <Toggle
              options={["0", "3", "4", "5"]}
              value={String(minCoh)}
              onChange={(v) => { setMinCoh(Number(v)); setLimit(10); }}
              label="min coherence"
            />
            <div className="ml-auto">
              <input
                value={q}
                onChange={(e) => { setQ(e.target.value); setLimit(10); }}
                placeholder="search concept..."
                className="bg-white border border-rule rounded px-2.5 py-1 w-44 focus:outline-none focus:border-cos/50"
              />
            </div>
          </div>

          <div className="text-[12.5px] font-sans text-muted mb-2">
            {feats.length} feature{feats.length === 1 ? "" : "s"} match.
          </div>

          <div className="space-y-1.5">
            {feats.slice(0, limit).map((f) => (
              <FeatureRow key={f.id} f={f} variant={variant} />
            ))}
          </div>

          {limit < feats.length && (
            <div className="flex justify-center mt-4">
              <button
                onClick={() => setLimit((l) => l + 10)}
                className="rounded border border-rule bg-white/60 hover:border-cos/40 px-4 py-1.5 text-[13px] font-sans text-muted hover:text-ink"
              >
                More concepts
              </button>
            </div>
          )}
        </div>
      </div>
    </Figure>
  );
}

function CoherenceDots({ value, color }: { value: number; color: string }) {
  return (
    <span className="inline-flex gap-0.5 shrink-0" title={`coherence ${value}/5`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span
          key={i}
          className="size-1.5 rounded-full"
          style={{ background: i <= value ? color : "transparent", border: `1px solid ${i <= value ? color : "#cfcec8"}` }}
        />
      ))}
    </span>
  );
}

function FeatureRow({ f, variant }: { f: Feat; variant: VariantName }) {
  return (
    <div className="flex items-center gap-3 rounded border border-rule/70 bg-white/50 px-3 py-2">
      <CoherenceDots value={f.coherence} color={COL[variant]} />
      <span className="font-mono text-[12px] text-muted shrink-0 w-14">#{f.id}</span>
      <span className={cn("font-sans text-[14px] text-ink truncate", f.coherence <= 2 && "text-soft italic")}>{f.concept}</span>
      <span className="ml-auto flex items-center gap-3 font-mono text-[12px] text-muted shrink-0">
        <span title="examples consistent (of 20)">{f.examplesConsistent}/20</span>
        <span title="max activation">act {f.maxAct.toFixed(2)}</span>
      </span>
    </div>
  );
}
