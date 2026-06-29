import { Figure } from "@/components/ui";
import figHero from "@/assets/paper/fig_hero.png";

export function Hero() {
  return (
    <Figure
      n={5}
      wide
      caption={
        <>
          <b>Cosine-scored SAEs win on probing because standard features fire on token norm.</b>{" "}
          <b>(A)</b> Sparse-probing top-1 across eight tasks, Qwen3-8B L18, 500M tokens,
          d_sae = 65,536, matched FVE approx 0.77. Per-feature cosine wins on 7/8 tasks;
          sentiment is the only exception. <b>(B)</b> Standard's unmatched features fire
          22x more on the highest-norm token quartile than on the lowest, versus 4.7x for
          cosine. <b>(C)</b> On the same high-norm tokens, the standard SAE reconstructs at
          9.5x the input norm, while cosine stays close to the input scale, 0.55x.
        </>
      }
    >
      <div className="bg-white p-2 sm:p-3">
        <img
          src={figHero}
          alt="ICML paper Figure 5: sparse-probing results, unmatched-feature norm firing, and reconstruction norm inflation"
          width={2250}
          height={858}
          className="block h-auto w-full"
          loading="eager"
          decoding="async"
        />
      </div>
    </Figure>
  );
}
