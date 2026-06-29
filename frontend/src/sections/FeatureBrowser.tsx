import { useMemo, useState } from "react";
import { L18, type FeatureL18 } from "@/data";
import { Figure, Num } from "@/components/ui";
import { TokenWindowViz } from "@/components/TokenViz";
import { cn } from "@/lib/utils";

type VariantKey = "Standard" | "Cosine" | "Cosine-global";

const VARIANTS: Record<VariantKey, { option: string; label: string; sub: string; tone: "standard" | "cosine" }> = {
  Standard: { option: "Standard", label: "Standard", sub: "inner product", tone: "standard" },
  Cosine: { option: "Per-feature cosine", label: "Per-feature cosine", sub: "per-feature", tone: "cosine" },
  "Cosine-global": { option: "Global-a cosine", label: "Global-a cosine", sub: "global a control", tone: "cosine" },
};

export function FeatureBrowser() {
  const [q, setQ] = useState("");
  const [limit, setLimit] = useState(8);
  const [variantKey, setVariantKey] = useState<VariantKey>("Cosine");
  const set = L18;
  const variantKeys = (Object.keys(VARIANTS) as VariantKey[]).filter((k) => set.variants[k]);
  const activeKey = set.variants[variantKey] ? variantKey : variantKeys[0];
  const activeMeta = VARIANTS[activeKey];
  const activeVariant = set.variants[activeKey];

  const filtered = useMemo(() => {
    const term = q.trim();
    let fs = activeVariant.features as FeatureL18[];
    if (term) fs = fs.filter((f) => String(f.id).includes(term));
    return fs;
  }, [activeVariant, q]);

  const renderCard = (f: FeatureL18) => <CardL18 key={f.id} f={f} />;

  return (
    <Figure
      n={10}
      wide
      caption={
        <>
          <b>Browse the dictionaries.</b> Top-activating contexts for individual features, standard vs
          cosine checkpoints selectable one at a time; the activating token is highlighted.{" "}
          {(L18 as any).source === "hf"
            ? "These are the published Silen/cosine-scored-saes-qwen3-8b checkpoints."
            : "Layer-18 development checkpoints."}
          <> Selected: {activeMeta.label} (<i>{activeMeta.sub}</i>), with <Num>{activeVariant.nAlive.toLocaleString()}</Num> live features at layer {set.layer}.</>
        </>
      }
    >
      <div className="p-4">
        <div className="flex flex-wrap items-center gap-3 mb-4 font-sans text-[13px]">
          <label className="flex items-center gap-2 text-muted" htmlFor="feature-browser-model">
            Model
            <select
              id="feature-browser-model"
              value={activeKey}
              onChange={(e) => {
                setVariantKey(e.target.value as VariantKey);
                setLimit(8);
              }}
              className="bg-white border border-rule rounded px-2.5 py-1 text-ink focus:outline-none focus:border-cos/50"
            >
              {variantKeys.map((k) => (
                <option key={k} value={k}>
                  {VARIANTS[k].option}
                </option>
              ))}
            </select>
          </label>
          <div className="ml-auto">
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="feature id…" className="bg-white border border-rule rounded px-2.5 py-1 w-32 focus:outline-none focus:border-cos/50" />
          </div>
        </div>
        <div className="max-w-3xl mx-auto">
          <Col tone={activeMeta.tone} label={activeMeta.label} sub={activeMeta.sub}>{filtered.slice(0, limit).map(renderCard)}</Col>
        </div>
        <div className="flex justify-center mt-5">
          <button onClick={() => setLimit((l) => l + 8)} className="rounded border border-rule bg-white/60 hover:border-cos/40 px-4 py-1.5 text-[13px] font-sans text-muted hover:text-ink">More features</button>
        </div>
      </div>
    </Figure>
  );
}

function Col({ tone, label, sub, children }: { tone: "standard" | "cosine"; label: string; sub: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-3 font-sans">
        <span className="size-2.5 rounded-[2px]" style={{ background: tone === "cosine" ? "#7C3AED" : "#1a1a1a" }} />
        <span data-testid="feature-browser-variant-title" className={cn("font-semibold text-[14px]", tone === "cosine" ? "text-cos" : "text-ink")}>{label}</span>
        <span className="text-[12.5px] text-muted">{sub}</span>
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function CardHeader({ id, freq, maxAct }: { id: number; freq: number; maxAct: number }) {
  return (
    <div className="flex items-center justify-between font-sans text-[12.5px]">
      <span className="font-mono text-ink">#{id}</span>
      <span className="flex items-center gap-3 text-muted font-mono">
        <span>act {maxAct.toFixed(1)}</span>
        <span>{freq.toLocaleString()}</span>
      </span>
    </div>
  );
}

function CardL18({ f }: { f: FeatureL18 }) {
  return (
    <div className="rounded-md border border-rule bg-white/50 p-3">
      <CardHeader id={f.id} freq={f.freq} maxAct={f.maxAct} />
      <div className="mt-2 space-y-1.5">
        {f.windows.slice(0, 3).map((w, i) => (
          <div key={i} className="rounded bg-paper border border-rule/60 px-2 py-1"><TokenWindowViz window={w} maxAct={f.maxAct} /></div>
        ))}
      </div>
    </div>
  );
}
