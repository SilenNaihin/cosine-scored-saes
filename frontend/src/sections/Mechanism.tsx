import { useMemo, useState } from "react";
import { ISO, type IsolationItem } from "@/data";
import { Figure, Num } from "@/components/ui";
import { TeX } from "@/components/Math";
import { cn } from "@/lib/utils";

type Demo = IsolationItem & { kind: "norm-detector" | "missed-signal" };

function buildExamples(): Demo[] {
  const fp = ISO.falsePositives.slice(0, 6).map((x) => ({ ...x, kind: "norm-detector" as const }));
  const lnm = ISO.lowNormMisses.slice(0, 6).map((x) => ({ ...x, kind: "missed-signal" as const }));
  const out: Demo[] = [];
  for (let i = 0; i < 6; i++) {
    if (fp[i]) out.push(fp[i]);
    if (lnm[i]) out.push(lnm[i]);
  }
  return out;
}

const SAE_MAX = 24;
const COS_MAX = 0.7;
const KL_PRESENT = 0.02;
const COS_PRESENT = 0.3; // cosine "fires" above this directional alignment

function Meter({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.max(0, Math.min(1, value / max));
  return (
    <div className="h-2 rounded-full bg-wash overflow-hidden">
      <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${pct * 100}%`, background: color }} />
    </div>
  );
}

export function Mechanism() {
  const examples = useMemo(buildExamples, []);
  const [i, setI] = useState(0);
  const ex = examples[i];

  const truthPresent = ex.kl >= KL_PRESENT;
  const stdSaysPresent = ex.saeAct > 3;
  const cosSaysPresent = ex.cos > COS_PRESENT;

  function go(n: number) {
    setI((p) => (p + n + examples.length) % examples.length);
  }

  return (
    <Figure
      n={2}
      wide
      caption={
        <>
          <b>Token by token pathology</b> Each example is a real token where the standard SAE and cosine disagree.
		  We compare the causal effect of removing the feature from the stream: we note that our cosine SAEs fire when causally relevant features are present, unlike the standard SAE.
		  From the published standard BatchTopK and cosine SAE checkpoint, {ISO.model} layer {ISO.layer}.
        </>
      }
    >
      <div className="p-4">
        {/* token chips */}
        <div className="flex flex-wrap gap-1.5 mb-4">
          {examples.map((e, idx) => (
            <button
              key={idx}
              onClick={() => setI(idx)}
              className={cn(
                "font-mono text-[13px] rounded px-2 py-0.5 border transition-colors",
                idx === i ? "border-cos bg-cos/10 text-ink" : "border-rule bg-white/50 text-muted hover:border-cos/50 hover:text-ink"
              )}
            >
              {e.token.replace(/\n/g, "⏎").trim() === "" ? "␣" : e.token.trim()}
            </button>
          ))}
        </div>

        <div className="rounded-md bg-wash/70 border border-rule px-4 py-3 mb-4 flex flex-wrap items-center gap-x-5 gap-y-1 font-sans text-[14px]">
          <span>token <span className="font-mono text-ink">"{ex.token}"</span></span>
          <span className="text-muted">feature <span className="font-mono">#{ex.feature}</span></span>
          <span className="text-muted"><TeX>{String.raw`\lVert x\rVert`}</TeX> <span className="font-mono text-ink">{ex.norm.toFixed(0)}</span></span>
          <span className={cn("ml-auto text-[12.5px] px-2 py-0.5 rounded-full", ex.kind === "norm-detector" ? "bg-ink/5 text-soft" : "bg-cos/10 text-cos")}>
            {ex.kind === "norm-detector" ? "high-norm, wrong direction" : "low-norm, right direction"}
          </span>
        </div>

        <div className="grid sm:grid-cols-2 gap-4">
          <div className="rounded-md border border-rule p-4">
            <Head dot="#1a1a1a" name="Standard" sub="inner product" verdict={stdSaysPresent === truthPresent ? "right" : "wrong"} />
            <div className="mt-3 text-[12.5px] font-sans text-muted flex justify-between mb-1"><span>activation</span><span className="font-mono text-ink">{ex.saeAct.toFixed(2)}</span></div>
            <Meter value={ex.saeAct} max={SAE_MAX} color="#1a1a1a" />
            <div className="mt-3 text-[13.5px] font-sans">{stdSaysPresent ? "fires — feature present" : "silent — feature absent"}</div>
          </div>
          <div className="rounded-md border border-cos/40 p-4 bg-cos/[0.03]">
            <Head dot="#7C3AED" name="Cosine" sub="direction only" verdict={cosSaysPresent === truthPresent ? "right" : "wrong"} />
            <div className="mt-3 text-[12.5px] font-sans text-muted flex justify-between mb-1"><span>cosine alignment</span><span className="font-mono text-cos">{ex.cos.toFixed(3)}</span></div>
            <Meter value={ex.cos} max={COS_MAX} color="#7C3AED" />
            <div className="mt-3 text-[13.5px] font-sans">{cosSaysPresent ? "fires — feature present" : "silent — feature absent"}</div>
          </div>
        </div>

        <div className="mt-4 rounded-md border border-rule p-4 bg-white/50">
          <div className="flex items-center justify-between mb-2 font-sans text-[14px]">
            <span className="font-semibold">Ground truth — ablate the feature, does this token actually matter?</span>
            <span className={cn("text-[12.5px] px-2 py-0.5 rounded-full", truthPresent ? "bg-cos/10 text-cos" : "bg-ink/5 text-soft")}>
              {truthPresent ? "causally important" : "no causal effect"}
            </span>
          </div>
          <Meter value={ex.kl} max={0.4} color={truthPresent ? "#7C3AED" : "#9a9a96"} />
          <p className="mt-3 text-[14px] font-sans text-soft leading-relaxed">
            {ex.kind === "norm-detector" ? (
              <>Standard fired at <Num>{ex.saeAct.toFixed(1)}</Num> on a high-norm token, but ablating it barely moves the model (KL {ex.kl.toFixed(3)}) — a norm-driven false positive. Cosine saw the low alignment and stayed silent.</>
            ) : (
              <>Standard missed this token (activation ≈ {ex.saeAct.toFixed(1)}), yet ablating the feature shifts the model by KL <Num>{ex.kl.toFixed(3)}</Num> — it matters. Cosine caught it from the directional signal.</>
            )}
          </p>
          <div className="flex justify-end mt-3">
            <button onClick={() => go(1)} className="rounded-md bg-cos hover:bg-cos/90 px-4 py-1.5 text-[14px] font-sans text-white">Next token →</button>
          </div>
        </div>
      </div>
    </Figure>
  );
}

function Head({ dot, name, sub, verdict }: { dot: string; name: string; sub: string; verdict: "right" | "wrong" | null }) {
  return (
    <div className="flex items-center gap-2">
      <span className="size-2.5 rounded-[2px]" style={{ background: dot }} />
      <span className="font-sans font-semibold text-[15px]">{name}</span>
      <span className="font-sans text-[12.5px] text-muted">{sub}</span>
      {verdict && (
        <span className={cn("ml-auto text-[12px] font-sans px-1.5 py-0.5 rounded", verdict === "right" ? "bg-cos/10 text-cos" : "bg-ink/5 text-muted")}>
          {verdict === "right" ? "✓ correct" : "✗ fooled"}
        </span>
      )}
    </div>
  );
}
