import React from "react";
import { Clock, Shield, AlertTriangle } from "lucide-react";

const RequestLog = ({ requests }) => {
  if (!requests || requests.length === 0) {
    return (
      <div
        className="bg-white border border-slate-200 rounded-sm p-5 h-full"
        data-testid="request-log"
      >
        <h3 className="text-lg font-medium tracking-tight mb-4">Live Request Feed</h3>
        <div className="text-center py-12 text-slate-400">
          <Shield size={48} className="mx-auto mb-3 opacity-30" />
          <p className="text-sm">No requests yet</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="bg-white border border-slate-200 rounded-sm p-5 h-full"
      data-testid="request-log"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium tracking-tight">Live Request Feed</h3>
        <span className="text-xs uppercase tracking-wider text-slate-500 font-medium">
          Last {requests.length}
        </span>
      </div>

      <div className="space-y-2 overflow-y-auto" style={{ maxHeight: "calc(100vh - 300px)" }}>
        {requests.map((req, index) => (
          <div
            key={req.request_id}
            className={`border border-slate-200 rounded-sm p-3 hover:border-slate-300 transition-colors ${
              index === 0 ? "flash-new" : ""
            }`}
            data-testid={`request-log-entry-${index}`}
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                {req.blocked ? (
                  <div className="bg-rose-100 text-rose-800 px-2 py-0.5 rounded text-xs font-mono">
                    BLOCKED
                  </div>
                ) : req.threat_detected ? (
                  <div className="bg-amber-100 text-amber-800 px-2 py-0.5 rounded text-xs font-mono">
                    WARNING
                  </div>
                ) : (
                  <div className="bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded text-xs font-mono">
                    SAFE
                  </div>
                )}
              </div>
              <span className="text-xs text-slate-500 font-mono text-right">
                {req.latency_ms.toFixed(2)}ms
              </span>
            </div>

            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs">
                <Clock size={12} className="text-slate-400" />
                <span className="text-slate-600 font-mono">
                  {new Date(req.timestamp).toLocaleTimeString()}
                </span>
              </div>

              {req.threat_type && req.threat_type !== "none" && (
                <div className="flex items-center gap-2 text-xs">
                  <AlertTriangle size={12} className="text-amber-600" />
                  <span className="text-slate-700">
                    {req.threat_type.replace("_", " ").toUpperCase()}
                  </span>
                  <span className="text-slate-500 font-mono ml-auto">
                    {(req.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              )}

              <div className="text-xs text-slate-500 font-mono truncate">
                ID: {req.request_id}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RequestLog;
