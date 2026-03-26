import React from "react";

const MetricsCard = ({ title, value, icon, trend, color, testId }) => {
  const colorClasses = {
    slate: "border-slate-900 bg-slate-50 text-slate-900",
    rose: "border-rose-600 bg-rose-50 text-rose-900",
    amber: "border-amber-600 bg-amber-50 text-amber-900",
    emerald: "border-emerald-600 bg-emerald-50 text-emerald-900"
  };

  return (
    <div
      className={`flex flex-col gap-1 border-l-4 p-4 rounded-sm ${
        colorClasses[color] || colorClasses.slate
      }`}
      data-testid={testId}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs uppercase tracking-wider font-medium opacity-70">
          {title}
        </span>
        {icon}
      </div>
      <div className="flex items-baseline gap-3">
        <span className="text-5xl font-light tracking-tighter leading-none">
          {value}
        </span>
        {trend && (
          <span className="text-xs font-medium opacity-60">{trend}</span>
        )}
      </div>
    </div>
  );
};

export default MetricsCard;
