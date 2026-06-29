import { actIntensity } from "@/lib/utils";
import type { TokenWindow, TextContext } from "@/data";

/** A token with a purple background scaled by activation intensity. */
function Tok({ text, intensity, peak }: { text: string; intensity: number; peak?: boolean }) {
  const display = text.replace(/\n/g, "⏎");
  const bg = intensity > 0.02 ? `rgba(184, 84, 61, ${0.1 + intensity * 0.7})` : "transparent";
  return (
    <span
      className="rounded-[2px] whitespace-pre-wrap"
      style={{
        background: bg,
        color: intensity > 0.6 ? "#fff" : undefined,
        boxShadow: peak ? "inset 0 0 0 1px rgba(124,58,237,0.7)" : undefined,
      }}
    >
      {display}
    </span>
  );
}

/** Token-id window (L18): highlight the activating token; taper neighbours. */
export function TokenWindowViz({ window: w, maxAct }: { window: TokenWindow; maxAct: number }) {
  const peakI = actIntensity(w.act, maxAct);
  return (
    <div className="font-mono text-[12.5px] leading-6 break-words text-ink/85">
      {w.tokens.map((t, i) => {
        const dist = Math.abs(i - w.pos);
        const intensity = i === w.pos ? peakI : dist <= 2 ? peakI * (0.16 - dist * 0.05) : 0;
        return <Tok key={i} text={t.t} intensity={Math.max(0, intensity)} peak={i === w.pos} />;
      })}
    </div>
  );
}

/** Text context (L27): prefix + highlighted target + suffix. */
export function TextContextViz({ ctx, maxAct }: { ctx: TextContext; maxAct: number }) {
  const intensity = actIntensity(ctx.act, maxAct);
  return (
    <div className="font-mono text-[12.5px] leading-6 break-words text-soft">
      <span>{ctx.prefix}</span>
      <Tok text={ctx.target} intensity={Math.max(0.35, intensity)} peak />
      <span>{ctx.suffix}</span>
    </div>
  );
}
