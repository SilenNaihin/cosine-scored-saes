import { useState } from "react";
import { Figure, Num } from "@/components/ui";
import { TeX } from "@/components/Math";
import fig1 from "@/data/fig1_tokens.json";

/* Figure 1 — one real layer-18 standard-SAE feature (#15189) that aligns with
   descriptive verbs. x = cosine of a token's residual to the feature direction;
   y = ‖residual‖. The feature keeps its top-k by ‖x‖^a·cos. Drag a: at a=0
   (cosine) / the learned a≈0.26 it keeps the verbs it points at; at a=1 (inner
   product) high-norm nouns it aligns with less crowd in. Data:
   experiments/modal_app.py::build_fig1 → frontend/src/data/fig1_tokens.json. */

type Tok = { t: string; cos: number; norm: number; k: "content" | "junk" };
const TOKENS = fig1.tokens as Tok[];
const K = fig1.k;
const COL = { content: "#7C3AED", junk: "#181818" };
const showTok = (t: string) => t.replace(/^ /, "␣");

const W = 540, H = 320, padL = 48, padT = 16, padB = 42;
const plotW = W - padL - 18, plotH = H - padT - padB;
const [c0, c1] = fig1.xDomain, [n0, n1] = fig1.yDomain;
const px = (c: number) => padL + ((c - c0) / (c1 - c0)) * plotW;
const py = (n: number) => padT + (1 - (n - n0) / (n1 - n0)) * plotH;

export function ScoreGeometry() {
  const [a, setA] = useState(fig1.learnedA);
  const [hover, setHover] = useState<number | null>(null);

  const scored = TOKENS.map((t) => ({ ...t, s: Math.pow(t.norm, a) * t.cos }));
  const thr = [...scored].sort((x, y) => y.s - x.s)[K - 1].s;
  const sel = scored.map((t) => t.s >= thr);
  const selVerb = scored.filter((t, i) => sel[i] && t.k === "content").length;
  const selNoun = scored.filter((t, i) => sel[i] && t.k === "junk").length;

  // selection cutoff: cos(n) = thr / n^a
  const cutoff = (() => {
    const pts: string[] = [];
    for (let n = n0; n <= n1; n += 1.5) {
      const c = thr / Math.pow(n, a);
      if (c >= c0 && c <= c1) pts.push(`${px(c).toFixed(1)},${py(n).toFixed(1)}`);
    }
    return pts.length ? "M" + pts.join(" L") : "";
  })();

  return (
    <Figure
      n={1}
      wide
      caption={
        <>
          <b>Direction first feature example:</b>
          Each point is a token scored by layer-18 feature 15,189.
          This feature aligns with descriptive verbs (e.g. <i>consists</i>, <i>found</i>,
          <i> tells</i>). It keeps the top-{K} tokens by score{" "}
          <TeX>{String.raw`\lVert x\rVert^{a}\cdot\cos`}</TeX>; the dashed line is that cutoff. At the learned{" "}
          <Num><TeX>{String.raw`a \approx 0.26`}</TeX></Num> and at <TeX>{String.raw`a = 0`}</TeX> (cosine) it keeps the
          verbs it points at. Slide to <TeX>{String.raw`a = 1`}</TeX> (inner product, a standard SAE feature) and
          high-norm nouns can be selected even when they align less well.
        </>
      }
    >
      <div className="p-4">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 360 }}>
          <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#cfcec8" strokeWidth={1} />
          <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="#cfcec8" strokeWidth={1} />
          <text x={padL + plotW / 2} y={H - 8} textAnchor="middle" fontSize={12} fontFamily="Space Grotesk" fill="#6b6b69">alignment  cos(x, w)  →</text>
          <text x={14} y={padT + plotH / 2} textAnchor="middle" fontSize={12} fontFamily="Space Grotesk" fill="#6b6b69" transform={`rotate(-90 14 ${padT + plotH / 2})`}>token norm ‖x‖  →</text>

          {cutoff && <path d={cutoff} fill="none" stroke="#7C3AED" strokeWidth={1.6} strokeDasharray="5 4" />}
          <text x={px(c1)} y={padT + 4} textAnchor="end" fontSize={10.5} fontFamily="Space Grotesk" fill="#7C3AED">top-{K} cutoff</text>

          {scored.map((t, i) => (
            <circle
              key={i}
              cx={px(t.cos)} cy={py(t.norm)} r={hover === i ? 8 : 6}
              fill={sel[i] ? COL[t.k] : "#fff"} stroke={COL[t.k]} strokeWidth={1.6}
              opacity={sel[i] ? 1 : 0.9}
              tabIndex={0} className="cursor-pointer outline-none"
              onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(i)} onBlur={() => setHover(null)}
            >
              <title>{`${showTok(t.t)}  ·  cos ${t.cos.toFixed(2)}  ·  ‖x‖ ${t.norm}  ·  ${sel[i] ? "selected" : "skipped"}`}</title>
            </circle>
          ))}

          {hover !== null && (() => {
            const t = scored[hover];
            const cx = px(t.cos), cy = py(t.norm);
            const l1 = showTok(t.t);
            const l2 = `cos ${t.cos.toFixed(2)} · ‖x‖ ${t.norm}`;
            const bw = Math.max(l1.length, l2.length) * 7 + 16, bh = 36;
            let bx = cx + 11; if (bx + bw > W - 4) bx = cx - 11 - bw;
            let by = cy - bh - 7; if (by < padT) by = cy + 11;
            return (
              <g pointerEvents="none">
                <rect x={bx} y={by} width={bw} height={bh} rx={5} fill="#fff" stroke="#cfcec8" strokeWidth={1} />
                <text x={bx + 8} y={by + 15} fontSize={12.5} fontFamily="JetBrains Mono" fill={COL[t.k]}>{l1}</text>
                <text x={bx + 8} y={by + 29} fontSize={11} fontFamily="Space Grotesk" fill="#6b6b69">{l2}</text>
              </g>
            );
          })()}
        </svg>

        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-3">
          <div className="flex items-center gap-3 grow min-w-[240px]">
            <span className="font-mono text-[13px] text-soft">a = {a.toFixed(2)}</span>
            <input
              type="range" min={0} max={1} step={0.01} value={a}
              onChange={(e) => setA(parseFloat(e.target.value))}
              className="grow accent-[#7C3AED]" aria-label="scoring exponent a"
            />
          </div>
          <div className="flex gap-1.5 font-sans text-[12.5px]">
            {([["inner product", 1], ["learned", fig1.learnedA], ["cosine", 0]] as const).map(([lbl, v]) => (
              <button
                key={lbl}
                onClick={() => setA(v)}
                className={`rounded border px-2 py-1 transition-colors ${Math.abs(a - v) < 0.005 ? "border-cos bg-cos/10 text-cos" : "border-rule text-muted hover:text-ink"}`}
              >
                {lbl} <span className="font-mono">{v.toFixed(2)}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="mt-3 text-[14px] font-sans text-soft">
          Top-{K} now keeps{" "}
          <span className="text-cos font-semibold">{selVerb} verb{selVerb === 1 ? "" : "s"}</span> and{" "}
          <span className="text-ink font-semibold">{selNoun} high-norm noun{selNoun === 1 ? "" : "s"}</span>.
          {selNoun >= 2 ? " Inner product is spending the dictionary on magnitude." : selNoun === 0 ? " The feature tracks the verbs it actually aligns with." : ""}
        </div>
      </div>
    </Figure>
  );
}
