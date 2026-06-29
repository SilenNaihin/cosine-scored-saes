import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { useEffect } from "react";

/** A number that smoothly tweens to its target whenever `value` changes. */
export function CountUp({
  value,
  format,
  className,
}: {
  value: number;
  format: (v: number) => string;
  className?: string;
}) {
  const mv = useMotionValue(value);
  const text = useTransform(mv, (v) => format(v));
  useEffect(() => {
    const controls = animate(mv, value, { duration: 0.5, ease: "easeOut" });
    return () => controls.stop();
  }, [value, mv]);
  return <motion.span className={className}>{text}</motion.span>;
}

/** A horizontal bar whose width animates to `pct` (0–100). */
export function AnimatedBar({
  pct,
  color,
  className,
  rounded = true,
}: {
  pct: number;
  color: string;
  className?: string;
  rounded?: boolean;
}) {
  return (
    <motion.div
      className={className}
      style={{ background: color, borderRadius: rounded ? 4 : 0 }}
      initial={false}
      animate={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    />
  );
}
