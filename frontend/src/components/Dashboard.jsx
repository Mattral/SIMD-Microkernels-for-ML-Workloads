import React, { useState, useEffect } from "react";
import axios from "axios";
import { 
  Activity, 
  Shield, 
  AlertTriangle, 
  TrendingUp, 
  Clock,
  Database,
  Zap
} from "lucide-react";
import MetricsCard from "@/components/MetricsCard";
import LatencyChart from "@/components/LatencyChart";
import RequestLog from "@/components/RequestLog";
import ThreatAnalytics from "@/components/ThreatAnalytics";
import SystemStatus from "@/components/SystemStatus";
import TestPanel from "@/components/TestPanel";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [threats, setThreats] = useState(null);
  const [requests, setRequests] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchData = async () => {
    try {
      const [metricsRes, threatsRes, requestsRes, healthRes] = await Promise.all([
        axios.get(`${API}/telemetry/metrics?hours=24`),
        axios.get(`${API}/telemetry/threats?hours=24`),
        axios.get(`${API}/telemetry/requests?limit=50`),
        axios.get(`${API}/health/`)
      ]);

      setMetrics(metricsRes.data);
      setThreats(threatsRes.data);
      setRequests(requestsRes.data);
      setHealth(healthRes.data);
      setLoading(false);
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchData, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block w-16 h-16 border-4 border-slate-200 border-t-slate-900 rounded-full animate-spin mb-4"></div>
          <p className="text-sm text-slate-600">Initializing GuardRail Studio...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white" data-testid="dashboard-container">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-50">
        <div className="w-full max-w-screen-2xl mx-auto px-4 md:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3">
                <Shield size={32} weight="bold" className="text-slate-900" />
                <div>
                  <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
                    GuardRail Studio
                  </h1>
                  <p className="text-xs uppercase tracking-wider text-slate-500 font-medium">
                    LLM Firewall & Observability
                  </p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                  autoRefresh
                    ? "bg-slate-900 text-white"
                    : "bg-white text-slate-900 border border-slate-200 hover:bg-slate-50"
                }`}
                data-testid="auto-refresh-toggle"
              >
                {autoRefresh ? "Live" : "Paused"}
              </button>
              <button
                onClick={fetchData}
                className="px-3 py-1.5 text-xs font-medium rounded bg-white text-slate-900 border border-slate-200 hover:bg-slate-50 transition-colors"
                data-testid="refresh-button"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="w-full max-w-screen-2xl mx-auto px-4 md:px-8 py-6">
        {/* System Status Banner */}
        <SystemStatus health={health} />

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricsCard
            title="Total Requests"
            value={metrics?.total_requests || 0}
            icon={<Activity size={20} weight="bold" />}
            trend="+12%"
            color="slate"
            testId="metric-total-requests"
          />
          <MetricsCard
            title="Blocked Requests"
            value={metrics?.blocked_requests || 0}
            icon={<Shield size={20} weight="bold" />}
            trend="-8%"
            color="rose"
            testId="metric-blocked-requests"
          />
          <MetricsCard
            title="Threats Detected"
            value={metrics?.threats_detected || 0}
            icon={<AlertTriangle size={20} weight="bold" />}
            trend="+5%"
            color="amber"
            testId="metric-threats-detected"
          />
          <MetricsCard
            title="p99 Latency"
            value={`${metrics?.p99_latency_ms?.toFixed(2) || 0}ms`}
            icon={<Zap size={20} weight="bold" />}
            trend="-15%"
            color="emerald"
            testId="metric-p99-latency"
          />
        </div>

        {/* Main Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Charts and Analytics */}
          <div className="lg:col-span-8 space-y-6">
            <LatencyChart metrics={metrics} />
            <ThreatAnalytics threats={threats} />
            <TestPanel onTestComplete={fetchData} />
          </div>

          {/* Right Column: Live Request Log */}
          <div className="lg:col-span-4">
            <RequestLog requests={requests} />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
