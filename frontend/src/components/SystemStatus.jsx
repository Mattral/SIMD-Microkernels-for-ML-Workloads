import React from "react";
import { CheckCircle, XCircle, AlertCircle } from "lucide-react";

const SystemStatus = ({ health }) => {
  if (!health) return null;

  const isHealthy = health.status === "healthy";
  const statusColor = isHealthy ? "emerald" : "amber";
  const StatusIcon = isHealthy ? CheckCircle : AlertCircle;

  const components = [
    { name: "Database", connected: health.database_connected },
    { name: "Vector DB", connected: health.qdrant_connected },
    { name: "Inference", connected: health.triton_connected }
  ];

  return (
    <div
      className="bg-white border border-slate-200 rounded-sm p-4 mb-6"
      data-testid="system-status"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <StatusIcon
            size={24}
            className={`text-${statusColor}-600`}
          />
          <div>
            <h3 className="text-sm font-medium text-slate-900">
              System Status: {health.status.toUpperCase()}
            </h3>
            <p className="text-xs text-slate-500 font-mono">
              Uptime: {Math.floor(health.uptime_seconds / 60)} minutes
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {components.map((comp) => (
            <div key={comp.name} className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  comp.connected ? "bg-emerald-600" : "bg-rose-600"
                }`}
              ></div>
              <span className="text-xs text-slate-600">{comp.name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SystemStatus;
