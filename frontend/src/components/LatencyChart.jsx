import React from "react";
import { LineChart } from "lucide-react";
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from "recharts";

const LatencyChart = ({ metrics }) => {
  if (!metrics) return null;

  const data = [
    { name: "p50", value: metrics.p50_latency_ms, label: "p50" },
    { name: "p95", value: metrics.p95_latency_ms, label: "p95" },
    { name: "p99", value: metrics.p99_latency_ms, label: "p99" },
    { name: "avg", value: metrics.avg_latency_ms, label: "avg" }
  ];

  return (
    <div
      className="bg-white border border-slate-200 rounded-sm p-5"
      data-testid="latency-chart"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium tracking-tight">Latency Distribution</h3>
        <div className="flex items-center gap-2">
          <LineChart size={16} className="text-slate-600" />
          <span className="text-sm text-slate-600">Last 24 Hours</span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-6">
        {data.map((item) => (
          <div
            key={item.name}
            className="text-center"
            data-testid={`latency-${item.name}`}
          >
            <div className="text-xs uppercase tracking-wider text-slate-500 font-medium mb-1">
              {item.label}
            </div>
            <div className="text-3xl font-light tracking-tighter text-slate-900">
              {item.value.toFixed(2)}
              <span className="text-sm text-slate-500 ml-1">ms</span>
            </div>
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <RechartsLineChart data={data}>
          <XAxis
            dataKey="label"
            tick={{ fontSize: 12, fill: "#475569" }}
            axisLine={{ stroke: "#E2E8F0" }}
          />
          <YAxis
            tick={{ fontSize: 12, fill: "#475569" }}
            axisLine={{ stroke: "#E2E8F0" }}
            label={{
              value: "Latency (ms)",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 12, fill: "#475569" }
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0F172A",
              border: "none",
              borderRadius: "4px",
              color: "#fff",
              fontSize: "12px"
            }}
            formatter={(value) => [`${value.toFixed(2)}ms`, "Latency"]}
          />
          <ReferenceLine
            y={10}
            stroke="#2563EB"
            strokeDasharray="3 3"
            label={{
              value: "Target: <10ms",
              position: "right",
              style: { fontSize: 11, fill: "#2563EB" }
            }}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#0F172A"
            strokeWidth={2}
            dot={{ fill: "#0F172A", r: 4 }}
            activeDot={{ r: 6 }}
          />
        </RechartsLineChart>
      </ResponsiveContainer>

      <div className="mt-4 p-3 bg-slate-50 border border-slate-200 rounded-sm">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-600">Average Latency</span>
          <span className="font-mono font-medium text-slate-900">
            {metrics.avg_latency_ms.toFixed(2)}ms
          </span>
        </div>
        <div className="mt-2 h-1 bg-slate-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-600 transition-all duration-500"
            style={{
              width: `${Math.min((metrics.avg_latency_ms / 10) * 100, 100)}%`
            }}
          ></div>
        </div>
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span>0ms</span>
          <span>10ms target</span>
        </div>
      </div>
    </div>
  );
};

export default LatencyChart;
