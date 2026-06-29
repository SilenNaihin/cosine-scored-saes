import { test, expect, type Page } from "@playwright/test";

function trackErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
  });
  return errors;
}

test.describe("Cosine-Scored SAEs — single-page article", () => {
  test("header, spotlight, abstract, and all sections render", async ({ page }) => {
    const errors = trackErrors(page);
    await page.goto("/");

    // title + byline
    await expect(page.getByRole("heading", { level: 1 })).toContainText("Cosine-Scored Sparse Autoencoders");
    await expect(page.getByText("Spotlight").first()).toBeVisible();
    await expect(page.getByText("ICML 2026 · Mechanistic").first()).toBeVisible();
    await expect(page.getByText("Silen Naihin")).toBeVisible();
    await expect(page.getByText("Lev Stambler")).toBeVisible();

    // canonical paper/arXiv abstract
    await expect(page.getByText(/Sparse autoencoders \(SAEs\) detect features via inner product/)).toBeVisible();
    await expect(page.getByText(/cosine scoring should be the default for dictionary learning on normalized representations/)).toBeVisible();
    await expect(page.locator("body")).not.toContainText(/\bexp59\b/i);

    // the numbered section headings
    for (const h of [
      "Introduction",
      "Headline result",
      "Where the gap comes from",
      "Robustness and mechanism checks",
      "Feature browser and scope",
    ]) {
      await expect(page.getByRole("heading", { name: new RegExp(h.replace(/[.,]/g, ".")) })).toBeVisible();
    }
    expect(errors, errors.join("\n")).toEqual([]);
  });

  test("interactive figures and charts render with purple accent", async ({ page }) => {
    const errors = trackErrors(page);
    await page.goto("/");

    // recharts surface (gradient figure) with drawn bars
    await expect(page.locator(".recharts-surface").first()).toBeVisible();
    const surfaces = await page.locator(".recharts-surface").count();
    expect(surfaces).toBeGreaterThanOrEqual(1);
    await expect(page.locator(".recharts-bar-rectangle").first()).toBeVisible();

    // interactive probe-win instrument still renders, and Figures 5/6 now use paper images.
    await expect(page.getByText(/Where the probing win comes from/i)).toBeVisible();
    const fig5 = page.locator("figure#fig-5 img");
    const fig6 = page.locator("figure#fig-6 img");
    await fig5.scrollIntoViewIfNeeded();
    await expect(fig5).toBeVisible();
    await fig6.scrollIntoViewIfNeeded();
    await expect(fig6).toBeVisible();
    await expect(page.getByAltText(/ICML paper Figure 5/i)).toBeVisible();
    await expect(page.getByAltText(/ICML paper Figure 6/i)).toBeVisible();

    // numbered figure captions render (low, stable numbers; AutoInterp covered separately)
    await expect(page.getByText(/Figure 1\./)).toBeVisible();
    await expect(page.getByText(/Figure 6\./)).toBeVisible();
    const accent = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--color-cos").trim());
    expect(accent.toLowerCase()).toBe("#7c3aed");
    await expect(page.locator("link[rel='icon']")).toHaveAttribute("href", /%237C3AED/i);
    await expect(page.locator("meta[name='theme-color']")).toHaveAttribute("content", "#7C3AED");
    const html = await page.locator("body").evaluate((el) => el.innerHTML);
    expect(html).not.toContain("#b8543d");
    expect(html).not.toContain("184,84,61");
    expect(errors, errors.join("\n")).toEqual([]);
  });

  test("probe win explorer: mode toggle collapses the gap", async ({ page }) => {
    const errors = trackErrors(page);
    await page.goto("/");

    const fig = page.getByText(/Where the probing win comes from/i);
    await fig.scrollIntoViewIfNeeded();
    await expect(page.getByText(/matched reconstruction · FVE/i)).toBeVisible();

    // full dictionary: large gap headline + per-dataset race wins
    await expect(page.getByText(/wins \d of \d datasets/i).first()).toBeVisible();
    await expect(page.getByText(/cosine-only features \(discovery\)/i)).toBeVisible();

    // switching to "shared features only" surfaces the collapsed-gap message
    await page.getByRole("button", { name: "shared", exact: true }).click();
    await expect(page.getByText(/the gap nearly vanishes/i)).toBeVisible();
    expect(errors, errors.join("\n")).toEqual([]);
  });

  test("auto-interp figure: matched interpretability rates", async ({ page }) => {
    const errors = trackErrors(page);
    await page.goto("/");

    const fig = page.getByText(/LLM-judged interpretable rate/i);
    await fig.scrollIntoViewIfNeeded();
    await expect(fig).toBeVisible();
    await expect(page.getByText(/Per-feature interpretability is matched/i)).toBeVisible();
    await expect(page.getByRole("columnheader", { name: /Low freq/i })).toBeVisible();
    expect(errors, errors.join("\n")).toEqual([]);
  });

  test("interactive mechanism figure: ground truth always shown, advance", async ({ page }) => {
    const errors = trackErrors(page);
    await page.goto("/");

    // ground truth is always visible (no reveal step)
    const gt = page.getByText(/does this token actually matter/i);
    await gt.scrollIntoViewIfNeeded();
    await expect(gt).toBeVisible();

    // advancing keeps it visible
    await page.getByRole("button", { name: /Next token/i }).click();
    await expect(page.getByText(/does this token actually matter/i)).toBeVisible();
    expect(errors, errors.join("\n")).toEqual([]);
  });

  test("interactive feature browser: variant dropdown, cards, pagination", async ({ page }) => {
    const errors = trackErrors(page);
    await page.goto("/");

    const more = page.getByRole("button", { name: /More features/i });
    await more.scrollIntoViewIfNeeded();
    await expect(more).toBeVisible();
    const modelSelect = page.getByLabel(/Model/i);
    const variantTitle = page.getByTestId("feature-browser-variant-title");
    await expect(modelSelect).toBeVisible();
    await expect(variantTitle).toHaveCount(1);
    await expect(variantTitle).toHaveText("Per-feature cosine");

    // feature-id cards are shown, and "more" paginates in additional cards
    await expect(page.locator("text=/#\\d{2,}/").first()).toBeVisible();
    const before = await page.locator("text=/#\\d{2,}/").count();
    await more.click();
    await expect(async () => {
      expect(await page.locator("text=/#\\d{2,}/").count()).toBeGreaterThan(before);
    }).toPass();

    await modelSelect.selectOption("Standard");
    await expect(variantTitle).toHaveText("Standard");
    await expect(page.locator("text=/#\\d{2,}/").first()).toBeVisible();

    await modelSelect.selectOption("Cosine-global");
    await expect(variantTitle).toHaveText("Global-a cosine");
    await expect(page.locator("text=/#\\d{2,}/").first()).toBeVisible();
    expect(errors, errors.join("\n")).toEqual([]);
  });

  test("prose math renders (KaTeX, no raw $…$ in body)", async ({ page }) => {
    const errors = trackErrors(page);
    await page.goto("/");

    // KaTeX actually rendered the inline/display math in the article body
    await expect(page.locator("article .katex").first()).toBeVisible();
    const katexCount = await page.locator("article .katex").count();
    expect(katexCount).toBeGreaterThan(5);

    // No leftover TeX delimiter or literal component tag in the *prose*.
    // Interactive figures legitimately contain "$" in real token contexts
    // (e.g. prices like "$800"), so exclude <figure> content.
    const prose = await page.evaluate(() => {
      const art = document.querySelector("article");
      if (!art) return "";
      const clone = art.cloneNode(true) as HTMLElement;
      clone.querySelectorAll("figure").forEach((f) => f.remove());
      return clone.textContent || "";
    });
    expect(prose, "raw $ delimiter leaked into rendered prose").not.toContain("$");
    expect(prose).not.toContain("<Math");
    expect(errors, errors.join("\n")).toEqual([]);
  });

  test("resource links present without blog citation", async ({ page }) => {
    const errors = trackErrors(page);
    await page.goto("/");
    await expect(page.getByText(/naihin2026cosine/)).toHaveCount(0);
    await expect(page.getByRole("button", { name: "copy" })).toHaveCount(0);
    await expect(page.getByRole("link", { name: /Cite/i })).toHaveCount(0);
    await expect(page.getByRole("link", { name: /arXiv/i }).first()).toHaveAttribute("href", "https://arxiv.org/abs/2606.15054");
    await expect(page.getByRole("link", { name: /Checkpoints/i }).first()).toBeVisible();
    expect(errors, errors.join("\n")).toEqual([]);
  });
});
