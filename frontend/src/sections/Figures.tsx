import { CHARTS, HEADLINE, C } from "@/data";
import { Figure, Num, Legend, VariantDot } from "@/components/ui";
import { TeX } from "@/components/Math";
import { GroupedBars } from "@/components/chart";
import type { ReactNode } from "react";
import figDiscoveryVsSeparability from "@/assets/paper/fig_discovery_vs_separability.png";

export function Decomposition() {
  return (
    <Figure
      n={6}
      caption={
        <>
          <b>Discovery dominates separability.</b> Sparse-probing accuracy when each SAE uses only
          features shared with the other dictionary ("shared features") versus its full dictionary
          ("all features"). Standard's flat slope shows its unique features add no probe signal;
          cosine's steep rise shows its unique features encode interpretable concepts. The gap on
          the right is the total probing advantage, driven almost entirely by feature discovery
          rather than cleaner encoding of shared directions.
        </>
      }
    >
      <div className="bg-white p-3 sm:p-4">
        <img
          src={figDiscoveryVsSeparability}
          alt="ICML paper Figure 6: discovery dominates separability shared-features versus all-features sparse probing"
          width={1269}
          height={735}
          className="mx-auto block h-auto w-full max-w-[720px]"
          loading="lazy"
          decoding="async"
        />
      </div>
    </Figure>
  );
}

/* ── Figure 5: multi-seed table ──────────────────────────────────────── */
export function ProbingTable() {
  return (
    <Figure
      n={7}
      caption={<>Multi-seed results (n=3 SAE-training seeds), Qwen3-8B L18, 500M tokens. FVE matched within 0.4%; the learned norm exponent <i>a</i> never approaches the inner-product limit (a=1).</>}
    >
      <div className="overflow-x-auto px-4 py-3">
        <table className="w-full text-[15px] font-sans">
          <thead>
            <tr className="text-muted text-left border-b border-rule">
              <th className="py-2 pr-4 font-medium">Variant</th>
              <th className="py-2 px-4 font-medium">FVE</th>
              <th className="py-2 px-4 font-medium">Probing top-1</th>
              <th className="py-2 px-4 font-medium">Learned <i className="font-serif">a</i></th>
            </tr>
          </thead>
          <tbody className="font-mono text-[14px] tabular-nums">
            {HEADLINE.variants.map((v) => {
              const cos = v.name !== "Standard";
              return (
                <tr key={v.name} className="border-b border-rule/60">
                  <td className="py-2.5 pr-4 font-sans">
                    <VariantDot variant={cos ? "Cosine" : "Standard"} /> <span className={cos ? "text-cos" : ""}>{v.name}</span>
                  </td>
                  <td className="py-2.5 px-4">{v.fve.toFixed(4)} <span className="text-muted text-[12px]">±{v.fveSd.toFixed(4)}</span></td>
                  <td className="py-2.5 px-4"><span className={cos ? "text-cos" : ""}>{v.probing.toFixed(4)}</span> <span className="text-muted text-[12px]">±{v.probingSd.toFixed(4)}</span></td>
                  <td className="py-2.5 px-4">{v.a === null ? "—" : v.a.toFixed(4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Figure>
  );
}

/* ── Figures 8-9: gradient allocation symptom + reweighting falsifier ── */
export function GradientFig() {
  const g = CHARTS.GRAD_EQ;
  const p = CHARTS.PROBING_HEADLINE;
  const data = g.layers.map((l: string, i: number) => ({
    layer: l,
    Standard: g.q4_q1_ratio.standard[i],
    Cosine: g.q4_q1_ratio.cosine[i],
  }));
  return (
    <>
      <Figure
        n={8}
        caption={
          <>
            <b>Standard encoder gradients are high-norm biased.</b> This chart shows the
            encoder-gradient ratio between the highest-norm (Q4) and lowest-norm (Q1) token
            quartiles. A ratio above 1 means the encoder spends more gradient on high-norm tokens;
            cosine stays close to balance across layers.
          </>
        }
      >
        <div className="px-1 pb-2 pt-3">
          <GroupedBars data={data} xKey="layer" height={240} refLine={1} />
          <div className="px-3">
            <Legend items={[{ label: "Standard", color: C.standard }, { label: "Cosine", color: C.cosine }]} />
          </div>
        </div>
      </Figure>

      <Figure
        n={9}
        caption={
          <>
            <b>Equalizing those gradients does not recover cosine probing.</b> We train the
            standard inner-product SAE with per-token reconstruction loss reweighted by{" "}
            <TeX>{String.raw`1/\lVert x\rVert`}</TeX> or{" "}
            <TeX>{String.raw`1/\lVert x\rVert^2`}</TeX>. Mild reweighting closes only{" "}
            <Num>{p.exp59_closure_pct_mild.toFixed(1)}%</Num> of the probing gap in this
            gradient-equalizing experiment showing that the main lever is the score function rather than just gradient reweighing.
          </>
        }
      >
        <div className="p-4 sm:p-5">
          <GradientFalsifier />
        </div>
      </Figure>
    </>
  );
}

function GradientFalsifier() {
  const p = CHARTS.PROBING_HEADLINE;
  const gradEq = CHARTS.palette.globalA;
  const rows = [
    { label: "Standard", value: p.exp59_standard, color: C.standard, weight: 600 },
    { label: <><span>+</span><TeX>{String.raw`1/\lVert x\rVert`}</TeX></>, value: p.exp59_grad_eq, color: gradEq, weight: 600 },
    { label: <><span>+</span><TeX>{String.raw`1/\lVert x\rVert^2`}</TeX></>, value: p.exp59_grad_eq_strong, color: gradEq, weight: 500 },
    { label: "Cosine", value: p.exp59_cosine, color: C.cosine, weight: 700 },
  ];
  const min = 0.50;
  const max = 0.665;
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
  const ticks = [0.50, 0.55, 0.60, 0.65];
  const mildClosure = p.exp59_closure_pct_mild;
  const strongClosure = ((p.exp59_grad_eq_strong - p.exp59_standard) / (p.exp59_cosine - p.exp59_standard)) * 100;
  const stdPct = pct(p.exp59_standard);
  const cosPct = pct(p.exp59_cosine);

  return (
    <div className="mx-auto w-full max-w-[560px] lg:max-w-none">
      <div className="grid grid-cols-[82px_1fr_48px] gap-x-2 font-sans text-[12px] sm:grid-cols-[96px_1fr_54px]">
        <div />
        <div className="relative h-8">
          <div
            className="absolute top-5 h-px bg-cos"
            style={{ left: `${stdPct}%`, width: `${cosPct - stdPct}%` }}
          />
          <div className="absolute top-5 h-2 w-px bg-cos" style={{ left: `${stdPct}%` }} />
          <div className="absolute top-5 h-2 w-px bg-cos" style={{ left: `${cosPct}%` }} />
          <div
            className="absolute top-0 -translate-x-1/2 whitespace-nowrap text-[11px] font-semibold text-cos"
            style={{ left: `${(stdPct + cosPct) / 2}%` }}
          >
            +{p.exp59_gap_pp.toFixed(1)}pp gap
          </div>
        </div>
        <div />

        {rows.map((r, i) => (
          <FalsifierRow
            key={i}
            label={r.label}
            value={r.value}
            color={r.color}
            weight={r.weight}
            pct={pct(r.value)}
            note={
              i === 1
                ? `closes ${mildClosure.toFixed(1)}%`
                : i === 2
                  ? `closes ${strongClosure.toFixed(1)}%`
                  : undefined
            }
            accent={i === 3}
          />
        ))}

        <div />
        <div className="relative mt-1 h-7 border-t border-rule/70">
          {ticks.map((t) => (
            <div
              key={t}
              className="absolute top-1 -translate-x-1/2 text-[11px] text-muted"
              style={{ left: `${pct(t)}%` }}
            >
              {Math.round(t * 100)}%
            </div>
          ))}
        </div>
        <div />
      </div>
      <div className="mt-2 border-t border-rule pt-2 font-sans text-[11.5px] leading-snug text-muted">
        Qwen3-8B L18, 50M-token gradient-equalizing experiment; top-1 sparse probing after training a standard inner-product SAE with loss reweighting.
      </div>
    </div>
  );
}

function FalsifierRow({
  label,
  value,
  color,
  weight,
  pct,
  note,
  accent,
}: {
  label: ReactNode;
  value: number;
  color: string;
  weight: number;
  pct: number;
  note?: string;
  accent?: boolean;
}) {
  return (
    <>
      <div
        className="flex h-10 items-center justify-end gap-0.5 text-right text-[12px] leading-none"
        style={{ color: accent ? C.cosine : C.standard, fontWeight: weight }}
      >
        {label}
      </div>
      <div className="relative h-10">
        <div className="absolute left-0 right-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-rule" />
        <div
          className="absolute left-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full"
          style={{ width: `${pct}%`, background: color, opacity: accent ? 0.28 : 0.20 }}
        />
        <span
          className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white"
          style={{ left: `${pct}%`, background: color }}
        />
        {note && (
          <span
            className="absolute left-0 top-[26px] whitespace-nowrap text-[11px] text-muted"
            style={{ transform: `translateX(min(${pct}%, calc(100% - 92px)))` }}
          >
            {note}
          </span>
        )}
      </div>
      <div className="flex h-10 items-center justify-start font-mono text-[12px] font-semibold tabular-nums" style={{ color }}>
        {(value * 100).toFixed(1)}%
      </div>
    </>
  );
}
