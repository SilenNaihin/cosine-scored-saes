import Markdown from "markdown-to-jsx";
import "katex/dist/katex.min.css";
import rawMd from "./content/article.md?raw";
import { HEADLINE } from "@/data";
import { Math } from "@/components/Math";
import { Hero } from "@/sections/Hero";
import { ProbeWin } from "@/sections/ProbeWin";
import { ScoreGeometry } from "@/sections/ScoreGeometry";
import { Mechanism } from "@/sections/Mechanism";
import { AutoInterp } from "@/sections/AutoInterp";
import { FeatureBrowser } from "@/sections/FeatureBrowser";
import { Decomposition, ProbingTable, GradientFig } from "@/sections/Figures";

/**
 * Convert `$$…$$` (display) and `$…$` (inline) math into <Math/> tags before
 * markdown-to-jsx runs. The TeX is URI-encoded into an attribute so backslashes,
 * `<`, `>` and braces pass through the HTML attribute parser untouched; the Math
 * component decodes and renders it with KaTeX.
 */
function mathTag(tex: string, display: boolean): string {
  const enc = encodeURIComponent(tex.trim());
  return display ? `<Math display="true" tex="${enc}" />` : `<Math tex="${enc}" />`;
}
const md = rawMd
  .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => mathTag(t, true))
  .replace(/\$([^$\n]+?)\$/g, (_, t) => mathTag(t, false));

const SECTIONS = [
  "Scoring geometry",
  "Sparse-probing result",
  "Matched-feature decomposition",
  "Direction vs. magnitude",
  "Replication and sample efficiency",
  "Per-feature interpretability",
  "Mechanism: gradient allocation",
  "Feature browser",
  "Scope and limitations",
];

function textOf(children: React.ReactNode): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(textOf).join("");
  return "";
}

/** Numbered, purple, anchorable section heading parsed from "N  Title". */
function H2({ children }: { children: React.ReactNode }) {
  const t = textOf(children).trim();
  const m = t.match(/^(\d+)\s+(.*)$/);
  const id = m ? `sec-${m[1]}` : undefined;
  return (
    <h2 id={id} className="scroll-mt-6 flex items-baseline gap-3">
      {m ? (
        <>
          <span className="font-mono text-cos text-[1.05rem] font-medium tabular-nums">{m[1]}</span>
          <span>{m[2]}</span>
        </>
      ) : (
        children
      )}
    </h2>
  );
}

function Toc() {
  return (
    <nav className="my-7 border-y border-rule py-4 font-sans not-prose">
      <div className="text-[12px] uppercase tracking-[0.18em] text-muted mb-2.5">Contents</div>
      <ol className="grid sm:grid-cols-2 gap-x-8 gap-y-1.5 text-[14px] list-none p-0 m-0">
        {SECTIONS.map((s, i) => (
          <li key={i} className="flex gap-2.5 leading-snug">
            <span className="font-mono text-cos tabular-nums">{i + 1}</span>
            <a href={`#sec-${i + 1}`} className="text-soft no-underline hover:text-cos">{s}</a>
          </li>
        ))}
      </ol>
    </nav>
  );
}

function Abstract({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-7 border-y border-rule py-5">
      <div className="font-sans text-[12px] uppercase tracking-[0.18em] text-muted mb-2">Abstract</div>
      <div className="text-[17px] leading-relaxed text-soft prose">{children}</div>
    </div>
  );
}

function DetailNote({ children, tip }: { children: React.ReactNode; tip: string }) {
  return (
    <span className="detail-note" tabIndex={0}>
      {children}
      <span className="detail-bubble" role="tooltip">{tip}</span>
    </span>
  );
}

const overrides = {
  Abstract: { component: Abstract },
  DetailNote: { component: DetailNote },
  Toc: { component: Toc },
  Math: { component: Math },
  h2: { component: H2 },
  ScoreGeometry: { component: ScoreGeometry },
  Hero: { component: Hero },
  ProbeWin: { component: ProbeWin },
  AutoInterp: { component: AutoInterp },
  Decomposition: { component: Decomposition },
  Mechanism: { component: Mechanism },
  ProbingTable: { component: ProbingTable },
  GradientFig: { component: GradientFig },
  FeatureBrowser: { component: FeatureBrowser },
};

export default function Article() {
  return (
    <div className="min-h-screen">
      <Header />
      <article className="mx-auto max-w-[680px] px-5 prose pb-10">
        <Markdown options={{ overrides, forceBlock: true }}>{md}</Markdown>
      </article>
      <Footer />
    </div>
  );
}

function Header() {
  return (
    <header className="mx-auto max-w-[680px] px-5 pt-16 pb-2">
      <div className="mb-4 flex flex-wrap items-center gap-2 font-sans">
        <span className="text-[12.5px] uppercase tracking-[0.18em] text-cos">
          {HEADLINE.venue} · {HEADLINE.workshop}
        </span>
        <span className="rounded-full bg-cos px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-white shadow-sm shadow-cos/20">
          {HEADLINE.award}
        </span>
      </div>
      <h1 className="text-[2.6rem] leading-[1.08] tracking-[-0.02em] font-sans font-semibold">
        Size Doesn't Matter:<br />Cosine-Scored Sparse Autoencoders
      </h1>
      <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 gap-y-3 gap-x-6 font-sans text-[14px] border-t border-rule pt-4">
        <Field label="Authors" value={<>Silen Naihin<br />Lev Stambler</>} />
        <Field label="Base model" value={<>Qwen3-8B · layer 18<br />d<sub>SAE</sub> = 65,536</>} />
        <Field label="Resources" value={
          <span className="flex flex-col gap-0.5">
            <a href={HEADLINE.arxiv} target="_blank" rel="noreferrer">arXiv ↗</a>
            <a href={HEADLINE.hf} target="_blank" rel="noreferrer">Checkpoints ↗</a>
            <a href={HEADLINE.code} target="_blank" rel="noreferrer">Code ↗</a>
          </span>
        } />
      </div>
    </header>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-muted mb-1">{label}</div>
      <div className="text-ink leading-snug">{value}</div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-rule mt-6">
      <div className="mx-auto max-w-[680px] px-5 py-10 font-sans">
        <div className="text-[13px] text-muted leading-relaxed">
          <span className="text-cos">Coming to Neuronpedia.</span> These SAEs are being onboarded for hosted
          feature dashboards; until then this article self-hosts them from the published checkpoints. ·{" "}
          <a href={HEADLINE.arxiv} target="_blank" rel="noreferrer">arXiv</a> ·{" "}
          <a href={HEADLINE.hf} target="_blank" rel="noreferrer">Hugging Face</a> ·{" "}
          <a href={HEADLINE.code} target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </div>
    </footer>
  );
}
