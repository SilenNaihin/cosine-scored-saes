import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

/** A numbered, anchorable section heading (Distill style). */
export function Section({ n, id, title }: { n: number; id: string; title: string }) {
  return (
    <h2 id={id} className="scroll-mt-20 flex items-baseline gap-3">
      <span className="font-mono text-cos text-base font-medium tabular-nums">{n}</span>
      <span>{title}</span>
    </h2>
  );
}

/** A figure frame: optional wide breakout + numbered caption. */
export function Figure({
  n,
  caption,
  wide,
  children,
}: {
  n: number;
  caption: ReactNode;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <figure
      id={`fig-${n}`}
      className={cn(
        "my-9 scroll-mt-20",
        wide && "lg:relative lg:left-1/2 lg:w-[min(960px,calc(100vw-2.5rem))] lg:-translate-x-1/2"
      )}
    >
      <div className="rounded-md border border-rule bg-white/40">{children}</div>
      <figcaption className="mt-2.5 text-[13.5px] leading-relaxed text-muted font-sans">
        <span className="font-semibold text-soft">Figure {n}.</span> {caption}
      </figcaption>
    </figure>
  );
}

export function Caption({ children }: { children: ReactNode }) {
  return <div className="text-[13px] leading-relaxed text-muted font-sans">{children}</div>;
}

/** A highlighted inline number (purple) for a claim. */
export function Num({ children }: { children: ReactNode }) {
  return <span className="font-sans font-semibold text-cos tabular-nums">{children}</span>;
}

export function Mono({ children }: { children: ReactNode }) {
  return <span className="font-mono text-[0.88em]">{children}</span>;
}

/** Small two-tone legend swatch row. */
export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap gap-4 text-[13px] font-sans text-soft">
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-[2px]" style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

/** Segmented control: a small inline button group. Shared by the interactive figures. */
export function Toggle({
  options,
  value,
  onChange,
  prefix = "",
  label,
}: {
  options: string[];
  value: string;
  onChange: (v: string) => void;
  prefix?: string;
  label?: string;
}) {
  return (
    <div className="flex items-center gap-2 font-sans text-[13px]">
      {label && <span className="text-muted">{label}</span>}
      <div className="inline-flex rounded border border-rule bg-white/60 p-0.5">
        {options.map((o) => (
          <button
            key={o}
            onClick={() => onChange(o)}
            className={cn(
              "px-2.5 py-0.5 rounded-[3px] transition-colors",
              value === o ? "bg-cos/15 text-ink" : "text-muted hover:text-ink"
            )}
          >
            {prefix}
            {o}
          </button>
        ))}
      </div>
    </div>
  );
}

export function VariantDot({ variant }: { variant: "Standard" | "Cosine" }) {
  return (
    <span
      className="inline-block size-2.5 rounded-[2px] align-middle"
      style={{ background: variant === "Cosine" ? "#7C3AED" : "#1a1a1a" }}
    />
  );
}
