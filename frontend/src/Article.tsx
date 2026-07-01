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
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-y-3 gap-x-6 font-sans text-[14px] border-t border-rule pt-4">
        <Field label="Authors" value={<>Silen Naihin<br />Lev Stambler</>} />
        <Field label="Base model" value={<>Qwen3-8B · layer 18 · 500M tokens<br />d<sub>SAE</sub> = 65,536</>} />
      </div>
      <div className="mt-5 flex flex-wrap gap-2.5 font-sans">
        <a
          href={HEADLINE.code}
          target="_blank"
          rel="noreferrer"
          className="group inline-flex items-center gap-2 rounded-md px-4 py-2.5 text-[15px] font-medium text-white no-underline transition-colors"
          style={{ backgroundColor: "#7C3AED" }}
        >
          <GitHubIcon />
          <span>Code</span>
          <StarIcon />
        </a>
        <a
          href={HEADLINE.arxiv}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md border border-rule px-4 py-2.5 text-[15px] font-medium text-ink no-underline transition-colors hover:border-cos hover:text-cos"
        >
          <span>arXiv</span>
          <span aria-hidden>↗</span>
        </a>
        <a
          href={HEADLINE.hf}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md border border-rule px-4 py-2.5 text-[15px] font-medium text-ink no-underline transition-colors hover:border-cos hover:text-cos"
        >
          <span>Checkpoints</span>
          <span aria-hidden>↗</span>
        </a>
      </div>
    </header>
  );
}

function GitHubIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 005.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 014 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"/>
    </svg>
  );
}

function StarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"/>
    </svg>
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
