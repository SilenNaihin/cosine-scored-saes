import featuresL18 from "./features_l18.json";
import featuresL27 from "./features_l27.json";
import isolation from "./isolation.json";
import vignettes from "./vignettes.json";
import autointerp from "./autointerp.json";
import headline from "./headline.json";
import charts from "./charts.json";
import probingDatasets from "./probing_datasets.json";

export type VariantName = "Standard" | "Cosine";

export interface TokenWindow {
  act: number;
  pos: number;
  tokens: { t: string }[];
}
export interface FeatureL18 {
  id: number;
  freq: number;
  maxAct: number;
  windows: TokenWindow[];
}
export interface TextContext {
  act: number;
  prefix: string;
  target: string;
  suffix: string;
}
export interface FeatureL27 {
  id: number;
  freq: number;
  band: string;
  maxAct: number;
  contexts: TextContext[];
}
export interface IsolationItem {
  feature: number;
  token: string;
  cos: number;
  saeAct: number;
  norm: number;
  kl: number;
}

type L18Entry = { key: string; nAlive: number; features: FeatureL18[] };
export const L18 = featuresL18 as {
  layer: number;
  tokenizerMissing: boolean;
  source?: string;
  variants: { Standard: L18Entry; Cosine: L18Entry } & Record<string, L18Entry>;
};
type L27Entry = { key: string; nAlive: number; features: FeatureL27[] };
export const L27 = featuresL27 as {
  layer: number;
  variants: { Standard: L27Entry; Cosine: L27Entry } & Record<string, L27Entry>;
};
export const ISO = isolation as {
  model: string;
  layer: number;
  falsePositives: IsolationItem[];
  lowNormMisses: IsolationItem[];
  pairs: any[];
};
export const VIGNETTES = vignettes as {
  config: any;
  concepts: Record<string, Record<VariantName, any[]>>;
};
export const AUTOINTERP = autointerp as {
  modelId: string;
  variants: Record<VariantName, { agg: any; features: any[] }>;
};
export const HEADLINE = headline as {
  variants: { name: string; a: number | null; fve: number; fveSd: number; probing: number; probingSd: number }[];
  headline: {
    probingDeltaPct: number;
    fve: number;
    globalA: number;
    interpPerFeatureMatched: boolean;
    interpTotalRatioNoAux: number;
    firingRatioStd: number;
    firingRatioCos: number;
    reconNormStdQ4: number;
    reconNormCosQ4: number;
    dirVsNormMin: number;
    dirVsNormMax: number;
  };
  venue: string;
  award: string;
  workshop: string;
  arxiv: string;
  hf: string;
  code: string;
};
export const CHARTS = charts as any;

export type Topk = { top_1: number; top_2: number | null; top_5: number | null };
export type ProbingDataset = {
  name: string;
  label: string;
  standard: Topk;
  cosine: Topk;
  llm: number | null;
};
export const PROBING = probingDatasets as {
  model: string;
  layer: number;
  seed: number;
  source: string;
  hfRepo: string;
  aggregate: { standard: Topk; cosine: Topk };
  datasets: ProbingDataset[];
};

/** Standard = near-black ink; Cosine = paper purple. The only two series colors. */
export const C = {
  standard: "#1a1a1a",
  cosine: "#7C3AED",
  grid: "#e6e4de",
  axis: "#6b6b6b",
};
