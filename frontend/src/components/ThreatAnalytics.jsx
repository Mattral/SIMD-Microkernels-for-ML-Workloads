import React from "react";
import { TrendingUp } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const ThreatAnalytics = ({ threats }) => {
  if (!threats) return null;

  const data = [
    {
      name: "Prompt Injection",
      value: threats.prompt_injection,
      color: "#E11D48"
    },
    {
      name: "PII Detection",
      value: threats.pii_detection,
      color: "#D97706"
    },
    {
      name: "Toxicity",
      value: threats.toxicity,
      color: "#DC2626"
    },
    {
      name: "Malicious Code",
      value: threats.malicious_code,
      color: "#B91C1C"
    }
  ];

  const total = data.reduce((sum, item) => sum + item.value, 0);

  return (
    <div
      className="bg-white border border-slate-200 rounded-sm p-5"
      data-testid="threat-analytics"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium tracking-tight">Threat Breakdown</h3>
        <div className="flex items-center gap-2">
          <TrendingUp size={16} className="text-slate-600" />
          <span className="text-sm text-slate-600">Last 24 Hours</span>
        </div>
      </div>

      {total === 0 ? (
        <div className="text-center py-12 text-slate-400">
          <p className="text-sm">No threats detected in the last 24 hours</p>
        </div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data}>
              <XAxis
                dataKey="name"
                tick={{ fontSize: 12, fill: "#475569" }}
                axisLine={{ stroke: "#E2E8F0" }}
              />
              <YAxis
                tick={{ fontSize: 12, fill: "#475569" }}
                axisLine={{ stroke: "#E2E8F0" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0F172A",
                  border: "none",
                  borderRadius: "4px",
                  color: "#fff",
                  fontSize: "12px"
                }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="grid grid-cols-2 gap-3 mt-6">
            {data.map((threat) => (
              <div
                key={threat.name}
                className="border border-slate-200 rounded-sm p-3"
                data-testid={`threat-${threat.name.toLowerCase().replace(/\s+/g, "-")}`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: threat.color }}
                  ></div>
                  <span className="text-xs uppercase tracking-wider font-medium text-slate-600">
                    {threat.name}
                  </span>
                </div>
                <div className="text-2xl font-light tracking-tighter text-slate-900">
                  {threat.value}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default ThreatAnalytics;
