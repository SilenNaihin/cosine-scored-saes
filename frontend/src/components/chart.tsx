import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ReferenceLine,
  LabelList,
} from "recharts";

const labelStyle = { fontSize: 11, fontFamily: "Source Sans 3, sans-serif", fontWeight: 600 } as const;
import { C } from "@/data";
import type { ReactNode } from "react";

export const axis = { stroke: C.axis, fontSize: 12, tickLine: false } as const;
export const grid = { stroke: C.grid, strokeDasharray: "0" } as const;
export const tip = {
  contentStyle: {
    background: "#fff",
    border: `1px solid ${C.grid}`,
    borderRadius: 6,
    fontSize: 12,
    fontFamily: "Source Sans 3, sans-serif",
    boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
  },
  labelStyle: { color: "#1a1a1a", fontWeight: 600 },
} as const;

export function ChartBox({ height = 240, children }: { height?: number; children: ReactNode }) {
  return (
    <div style={{ height, padding: "14px 12px 6px" }}>
      <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 480, height }}>
        {children as any}
      </ResponsiveContainer>
    </div>
  );
}

/** Simple single-series bar chart, per-bar colors via `colorOf`. */
export function Bars({
  data,
  xKey,
  yKey,
  colorOf,
  domain,
  height,
  fmt,
  valueLabel,
}: {
  data: any[];
  xKey: string;
  yKey: string;
  colorOf: (d: any) => string;
  domain?: [number, number];
  height?: number;
  fmt?: (v: number) => string;
  valueLabel?: (v: number) => string;
}) {
  return (
    <ChartBox height={height}>
      <BarChart data={data} margin={{ top: 18, right: 10, bottom: 2, left: -14 }}>
        <CartesianGrid {...grid} vertical={false} />
        <XAxis dataKey={xKey} {...axis} />
        <YAxis {...axis} domain={domain as any} tickFormatter={fmt as any} />
        <Tooltip {...tip} cursor={{ fill: "rgba(124,58,237,0.06)" }} formatter={fmt as any} />
        <Bar dataKey={yKey} animationDuration={600} maxBarSize={64}>
          {data.map((d, i) => (
            <Cell key={i} fill={colorOf(d)} />
          ))}
          {valueLabel && <LabelList dataKey={yKey} position="top" formatter={valueLabel as any} style={labelStyle} />}
        </Bar>
      </BarChart>
    </ChartBox>
  );
}

/** Grouped bars: Standard (ink) vs Cosine (purple) over a category axis. */
export function GroupedBars({
  data,
  xKey,
  height,
  refLine,
  refLabel,
  valueLabel,
}: {
  data: { [k: string]: any; Standard: number; Cosine: number }[];
  xKey: string;
  height?: number;
  refLine?: number;
  refLabel?: string;
  valueLabel?: (v: number) => string;
}) {
  return (
    <ChartBox height={height}>
      <BarChart data={data} margin={{ top: 18, right: 10, bottom: 2, left: -14 }}>
        <CartesianGrid {...grid} vertical={false} />
        <XAxis dataKey={xKey} {...axis} />
        <YAxis {...axis} />
        <Tooltip {...tip} cursor={{ fill: "rgba(124,58,237,0.06)" }} />
        {refLine !== undefined && (
          <ReferenceLine y={refLine} stroke={C.axis} strokeDasharray="4 3"
            label={refLabel ? { value: refLabel, position: "insideTopRight", fill: C.axis, fontSize: 10 } : undefined} />
        )}
        <Bar dataKey="Standard" fill={C.standard} animationDuration={600} maxBarSize={26}>
          {valueLabel && <LabelList dataKey="Standard" position="top" formatter={valueLabel as any} style={labelStyle} />}
        </Bar>
        <Bar dataKey="Cosine" fill={C.cosine} animationDuration={600} maxBarSize={26}>
          {valueLabel && <LabelList dataKey="Cosine" position="top" formatter={valueLabel as any} style={labelStyle} />}
        </Bar>
      </BarChart>
    </ChartBox>
  );
}

export function Lines({
  data,
  xKey,
  xLabel,
  height,
}: {
  data: any[];
  xKey: string;
  xLabel?: string;
  height?: number;
}) {
  return (
    <ChartBox height={height}>
      <LineChart data={data} margin={{ top: 6, right: 12, bottom: xLabel ? 14 : 2, left: -12 }}>
        <CartesianGrid {...grid} />
        <XAxis dataKey={xKey} {...axis} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -4, fill: C.axis, fontSize: 11 } : undefined} />
        <YAxis {...axis} />
        <Tooltip {...tip} />
        <Line type="monotone" dataKey="Standard" stroke={C.standard} strokeWidth={2} dot={{ r: 2.5 }} animationDuration={700} />
        <Line type="monotone" dataKey="Cosine" stroke={C.cosine} strokeWidth={2.5} dot={{ r: 2.5 }} animationDuration={700} />
      </LineChart>
    </ChartBox>
  );
}
