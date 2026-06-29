import katex from "katex";

/** Render a TeX string to a KaTeX HTML string (errors render in-place, not thrown). */
function render(src: string, display: boolean): string {
  return katex.renderToString(src, {
    displayMode: display,
    throwOnError: false,
    strict: false,
    output: "htmlAndMathml",
  });
}

/**
 * Inline/display math for use directly in TSX. Pass the raw TeX as a string child,
 * e.g. <TeX>{String.raw`\lVert x\rVert^{a}\cos(x, w_i)`}</TeX>.
 */
export function TeX({ children, block = false }: { children: string; block?: boolean }) {
  const html = render(children, block);
  if (block) {
    return <div className="katex-block" dangerouslySetInnerHTML={{ __html: html }} />;
  }
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/**
 * Math node emitted by the markdown pipeline. `tex` arrives URI-encoded so that
 * backslashes, `<`, `>` and `$` survive markdown-to-jsx's HTML attribute parser.
 * `display` is the boolean-ish prop markdown-to-jsx produces from `display="true"`.
 */
export function Math({ tex, display }: { tex: string; display?: boolean | string }) {
  const block = display === true || display === "true";
  return <TeX block={block}>{decodeURIComponent(tex)}</TeX>;
}
