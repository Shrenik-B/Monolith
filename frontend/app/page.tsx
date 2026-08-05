'use client';

import React, { useEffect, useState, useMemo } from 'react';
import {
  TrendingUp,
  Activity,
  AlertTriangle,
  Zap,
  Calendar,
  Search,
  Bot,
  Sliders,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Info,
  CheckCircle2,
  PieChart as PieIcon,
  Layers,
  BarChart3,
  Globe,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  ShieldAlert,
  FileText
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter
} from 'recharts';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface DateInfo {
  target_idx: number;
  date: string;
  current_regime: number;
  current_regime_name: string;
  current_regime_color: string;
  confidence_score: number;
  state_probabilities: { name: string; probability: number; color: string }[];
  tomorrow_regime: number;
  tomorrow_regime_name: string;
  tomorrow_regime_color: string;
  tomorrow_confidence: number;
  tomorrow_probabilities: { name: string; probability: number; color: string }[];
  persistence_prob: number;
  transition_out_prob: number;
  top_transition_state: number;
  top_transition_regime_name: string;
  top_transition_prob: number;
  has_next_day: boolean;
  actual_next_date?: string;
  actual_next_state?: number;
  actual_next_regime_name?: string;
  actual_next_regime_color?: string;
  actual_next_confidence?: number;
  actual_regime_changed?: boolean;
  macro_row: Record<string, number | string>;
  radar_data: { feature: string; key: string; value: number; raw_value: number }[];
  regime_mapping: Record<string, string>;
}

interface DtwMatch {
  rank: number;
  start_date: string;
  end_date: string;
  similarity_score: number;
  normalized_distance: number;
  dominant_regime: number;
  dominant_regime_name: string;
  dominant_regime_color: string;
  forward_30d_return: number | null;
  vix_values: number[];
}

interface DtwData {
  query_start_date: string;
  query_end_date: string;
  window_size: number;
  query_vix: number[];
  matches: DtwMatch[];
}

interface TransitionData {
  transition_matrix: number[][];
  states_labels: string[];
  short_labels: string[];
  persistence_table: {
    state_id: number;
    regime_name: string;
    regime_color: string;
    daily_persistence: number;
    avg_days: number;
  }[];
}

interface ModelMetadata {
  n_states: number;
  n_features: number;
  training_timestamp: string;
  metrics: {
    aic: number;
    bic: number;
    log_likelihood: number;
    n_samples: number;
  };
  feature_names: string[];
  regime_mapping: Record<string, string>;
  regime_colors: Record<string, string>;
}

interface AIReport {
  overview: string;
  outlook: string;
  analogs: string;
  drivers: string;
}

const toMonochromeColor = (color?: string) => {
  if (!color) return '#ffffff';
  const normalized = color.trim();
  const hex = normalized.startsWith('#') ? normalized.slice(1) : normalized;

  if (/^[0-9a-fA-F]{3}$/.test(hex)) {
    return `#${hex.split('').map((ch) => ch + ch).join('')}`;
  }

  if (/^[0-9a-fA-F]{6}$/.test(hex)) {
    const value = parseInt(hex, 16);
    const gray = Math.round(((value >> 16) & 255) * 0.3 + ((value >> 8) & 255) * 0.59 + (value & 255) * 0.11);
    return `rgb(${gray}, ${gray}, ${gray})`;
  }

  const match = normalized.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
  if (match) {
    const gray = Math.round(Number(match[1]) * 0.3 + Number(match[2]) * 0.59 + Number(match[3]) * 0.11);
    return `rgb(${gray}, ${gray}, ${gray})`;
  }

  return '#ffffff';
};

const monochromePalette = ['#ffffff', '#e5e5e5', '#a3a3a3', '#525252', '#111111'];

export default function Dashboard() {
  // State variables
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [inference, setInference] = useState<DateInfo | null>(null);
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [regimeFrequencies, setRegimeFrequencies] = useState<any[]>([]);
  const [dtwData, setDtwData] = useState<DtwData | null>(null);
  const [transitionData, setTransitionData] = useState<TransitionData | null>(null);
  const [aiReport, setAiReport] = useState<AIReport | null>(null);

  // UI state
  const [activeTab, setActiveTab] = useState<'charts' | 'forecast' | 'dtw' | 'ai' | 'diagnostics'>('charts');
  const [selectedIndicator, setSelectedIndicator] = useState<string>('vix_percentile');
  const [showColorByRegime, setShowColorByRegime] = useState<boolean>(true);
  const [dtwWindowSize, setDtwWindowSize] = useState<number>(30);
  const [dtwTopK, setDtwTopK] = useState<number>(3);
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingAi, setLoadingAi] = useState<boolean>(false);
  const [showEduGuide, setShowEduGuide] = useState<boolean>(false);

  // Initial Boot Data Loading
  useEffect(() => {
    async function initData() {
      try {
        setLoading(true);
        const [datesRes, metaRes, histRes, transRes] = await Promise.all([
          fetch(`${API_BASE}/api/dates`).then(r => r.json()),
          fetch(`${API_BASE}/api/metadata`).then(r => r.json()),
          fetch(`${API_BASE}/api/history`).then(r => r.json()),
          fetch(`${API_BASE}/api/transition-matrix`).then(r => r.json())
        ]);

        setAvailableDates(datesRes.dates);
        const defaultDate = datesRes.max_date || datesRes.dates[datesRes.dates.length - 1];
        setSelectedDate(defaultDate);
        setMetadata(metaRes);
        setHistoryData(histRes.history);
        setFeatureNames(histRes.feature_names);
        setRegimeFrequencies(histRes.regime_frequencies);
        setTransitionData(transRes);

        // Load inference for default date
        const inferRes = await fetch(`${API_BASE}/api/inference?date=${defaultDate}`).then(r => r.json());
        setInference(inferRes);

        // Load DTW for default date
        const dtwRes = await fetch(`${API_BASE}/api/dtw?date=${defaultDate}&window_size=30&top_k=3`).then(r => r.json());
        setDtwData(dtwRes);

        setLoading(false);
      } catch (err) {
        console.error('Failed to load initial data:', err);
        setLoading(false);
      }
    }
    initData();
  }, []);

  // Update date-dependent inferences when date changes
  const handleDateChange = async (newDate: string) => {
    setSelectedDate(newDate);
    setAiReport(null); // reset commentary on date shift
    try {
      const [inferRes, dtwRes] = await Promise.all([
        fetch(`${API_BASE}/api/inference?date=${newDate}`).then(r => r.json()),
        fetch(`${API_BASE}/api/dtw?date=${newDate}&window_size=${dtwWindowSize}&top_k=${dtwTopK}`).then(r => r.json())
      ]);
      setInference(inferRes);
      setDtwData(dtwRes);
    } catch (err) {
      console.error('Error fetching date inference:', err);
    }
  };

  // Re-fetch DTW when sliders change
  const handleDtwSliderChange = async (winSize: number, topK: number) => {
    setDtwWindowSize(winSize);
    setDtwTopK(topK);
    if (!selectedDate) return;
    try {
      const res = await fetch(`${API_BASE}/api/dtw?date=${selectedDate}&window_size=${winSize}&top_k=${topK}`).then(r => r.json());
      setDtwData(res);
    } catch (err) {
      console.error('Error updating DTW:', err);
    }
  };

  // Generate AI Analyst Report
  const generateAiReport = async () => {
    if (!selectedDate) return;
    setLoadingAi(true);
    try {
      const res = await fetch(`${API_BASE}/api/ai-report?date=${selectedDate}`).then(r => r.json());
      setAiReport(res.report);
      setLoadingAi(false);
    } catch (err) {
      console.error('Failed to generate AI report:', err);
      setLoadingAi(false);
    }
  };

  // Prepare chart data for DTW VIX Trajectories
  const dtwChartData = useMemo(() => {
    if (!dtwData || !dtwData.query_vix) return [];
    const len = dtwData.query_vix.length;
    const rows = [];
    for (let i = 0; i < len; i++) {
      const row: any = { day: i + 1, query: dtwData.query_vix[i] };
      dtwData.matches.forEach((m, idx) => {
        if (m.vix_values && i < m.vix_values.length) {
          row[`match_${idx + 1}`] = m.vix_values[i];
        }
      });
      rows.push(row);
    }
    return rows;
  }, [dtwData]);

  // Downsample history data slightly for smooth scatter chart performance
  const chartHistoryData = useMemo(() => {
    if (!historyData || historyData.length === 0) return [];
    // If large, sample every 2nd or 3rd point, but keep selected date intact
    if (historyData.length < 2000) return historyData;
    return historyData.filter((_, idx) => idx % 2 === 0);
  }, [historyData]);

  if (loading || !inference) {
    return (
      <div className="min-h-screen bg-black flex flex-col items-center justify-center text-white space-y-4">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-white/20 border-t-white rounded-full animate-spin"></div>
          <Activity className="w-6 h-6 text-white absolute inset-0 m-auto animate-pulse" />
        </div>
        <div className="text-center">
          <h2 className="text-xl font-bold text-white">
            Loading Monolith Quantitative Engine...
          </h2>
          <p className="text-sm text-zinc-400 mt-1">Initializing Baum-Welch HMM & Historical DTW Datasets</p>
        </div>
      </div>
    );
  }

  // Calculate VIX & Credit Spread badges
  const vixVal = (inference.macro_row.vix_percentile as number || 0) * 100;
  const vixBadge = vixVal > 70 ? { label: '◉ Elevated Fear', cls: 'bg-white/10 text-white border-white/20' }
    : vixVal > 40 ? { label: '◌ Moderate', cls: 'bg-white/10 text-white border-white/20' }
    : { label: '● Calm', cls: 'bg-white/10 text-white border-white/20' };

  const creditVal = (inference.macro_row.credit_spread as number || 0);
  const creditBadge = creditVal > 3.0 ? { label: '◉ Wide Stress', cls: 'bg-white/10 text-white border-white/20' }
    : creditVal > 2.5 ? { label: '◌ Watch', cls: 'bg-white/10 text-white border-white/20' }
    : { label: '● Tight', cls: 'bg-white/10 text-white border-white/20' };

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 flex flex-col font-sans">
      {/* ==================================================================== */}
      {/* HEADER BAR */}
      {/* ==================================================================== */}
      <header className="bg-black/90 border-b border-white/10 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1700px] mx-auto px-4 lg:px-8 py-3 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-2xl bg-white/10 border border-white/15 flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
                Market Regime Intelligence
                <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white border border-white/15 font-mono font-medium">
                  V1 React Engine
                </span>
              </h1>
              <p className="text-xs text-zinc-400 hidden sm:block">
                K-Means++ Init &rarr; Gaussian HMM (Baum–Welch) &rarr; BIC Selection &rarr; DTW Engine &rarr; AI Analyst
              </p>
            </div>
          </div>

          {/* Date Selector Input */}
          <div className="flex items-center space-x-3 bg-black border border-white/15 px-3.5 py-1.5 rounded-full shadow-inner">
            <Calendar className="w-4 h-4 text-white" />
            <label className="text-xs font-semibold text-white">Target Date:</label>
            <select
              value={selectedDate}
              onChange={(e) => handleDateChange(e.target.value)}
              className="bg-black text-white font-mono text-sm border border-white/15 rounded-full px-2.5 py-1 focus:outline-none focus:border-white transition cursor-pointer"
            >
              {availableDates.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {/* ==================================================================== */}
      {/* MAIN CONTAINER (Sidebar + Dashboard Content) */}
      {/* ==================================================================== */}
      <div className="max-w-[1700px] mx-auto px-4 lg:px-8 py-6 flex-1 flex flex-col lg:flex-row gap-6 w-full">
        {/* SIDEBAR */}
        <aside className="w-full lg:w-72 flex flex-col gap-6 shrink-0">
          {/* Model Summary Box */}
          <div className="glass-card p-5 space-y-4">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
              <Bot className="w-5 h-5 text-white" />
              <h2 className="font-semibold text-sm text-white">Model Diagnostics</h2>
            </div>
            {metadata && (
              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between items-center text-slate-400">
                  <span>HMM States:</span>
                  <span className="font-mono text-slate-200 font-bold text-sm bg-slate-800 px-2 py-0.5 rounded">
                    {metadata.n_states}
                  </span>
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>Training Samples:</span>
                  <span className="font-mono text-slate-200">{metadata.metrics.n_samples?.toLocaleString()} days</span>
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>AIC Metric:</span>
                  <span className="font-mono text-white">{Math.round(metadata.metrics.aic).toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>BIC Metric:</span>
                  <span className="font-mono text-white">{Math.round(metadata.metrics.bic).toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>Log-Likelihood:</span>
                  <span className="font-mono text-white">{metadata.metrics.log_likelihood.toFixed(1)}</span>
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>Features Used:</span>
                  <span className="font-mono text-white">{metadata.n_features}</span>
                </div>
                <div className="border-t border-slate-800/80 pt-2 text-[11px] text-slate-500 flex justify-between">
                  <span>Trained Date:</span>
                  <span className="font-mono text-slate-400">{metadata.training_timestamp.slice(0, 10)}</span>
                </div>
              </div>
            )}
          </div>

          {/* Regime Legend Box */}
          <div className="glass-card p-5 space-y-3">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
              <Layers className="w-4 h-4 text-white" />
              <h2 className="font-semibold text-sm text-slate-200">Regime State Legend</h2>
            </div>
            <div className="space-y-2 text-xs">
              {metadata &&
                Object.entries(metadata.regime_mapping).map(([stateId, regName]) => {
                  const color = metadata.regime_colors[regName] || '#95a5a6';
                  return (
                    <div key={stateId} className="flex items-start space-x-2.5 bg-slate-900/60 p-2 rounded-lg border border-slate-800/60">
                      <span className="w-3 h-3 rounded-full mt-0.5 shrink-0" style={{ backgroundColor: color }} />
                      <div>
                        <span className="font-mono font-bold text-slate-300">State {stateId}</span>
                        <p className="text-[11px] text-slate-400 leading-tight">{regName}</p>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        </aside>

        {/* MAIN BODY CONTENT */}
        <main className="flex-1 flex flex-col gap-6 min-w-0">
          {/* ==================================================================== */}
          {/* TOP METRIC CARDS (5 METRICS) */}
          {/* ==================================================================== */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {/* Metric 1: Current Market Regime */}
            <div className="glass-card glass-card-hover p-4 border-l-4 space-y-1.5" style={{ borderLeftColor: toMonochromeColor(inference.current_regime_color) }}>
              <span className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5 text-white" /> Current Regime
              </span>
              <div className="text-sm font-bold text-slate-100 leading-snug line-clamp-2" title={inference.current_regime_name}>
                {inference.current_regime_name}
              </div>
              <span className="inline-block text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full" style={{ backgroundColor: 'rgba(255,255,255,0.08)', color: '#ffffff', border: '1px solid rgba(255,255,255,0.16)' }}>
                State {inference.current_regime}
              </span>
            </div>

            {/* Metric 2: Model Confidence */}
            <div className="glass-card glass-card-hover p-4 border-l-4 border-l-cyan-500 space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-white" /> Model Confidence
              </span>
              <div className="text-2xl font-mono font-bold text-white">
                {(inference.confidence_score * 100).toFixed(1)}%
              </div>
              <p className="text-[11px] text-slate-400">Posterior probability certainty</p>
            </div>

            {/* Metric 3: Tomorrow / Realized Outcome */}
            <div
              className="glass-card glass-card-hover p-4 border-l-4 space-y-1.5"
              style={{
                borderLeftColor: toMonochromeColor(inference.has_next_day && inference.actual_regime_changed
                  ? '#ffffff'
                  : inference.tomorrow_regime_color)
              }}
            >
              <span className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase flex items-center gap-1.5">
                {inference.has_next_day ? <Clock className="w-3.5 h-3.5 text-white" /> : <Zap className="w-3.5 h-3.5 text-white" />}
                {inference.has_next_day ? `Realized Outcome (${inference.actual_next_date})` : "Tomorrow's Forecast"}
              </span>
              <div className="text-xs font-bold text-slate-100 leading-snug line-clamp-2">
                {inference.has_next_day ? inference.actual_next_regime_name : inference.tomorrow_regime_name}
              </div>
              <div className="text-[11px]">
                {inference.has_next_day ? (
                  inference.actual_regime_changed ? (
                    <span className="text-white font-semibold font-mono">⚡ State shifted next day</span>
                  ) : (
                    <span className="text-white font-semibold font-mono">🔄 Persisted next day</span>
                  )
                ) : (
                  <span className="text-white font-mono">{(inference.tomorrow_confidence * 100).toFixed(1)}% prob</span>
                )}
              </div>
            </div>

            {/* Metric 4: VIX Percentile */}
            <div className="glass-card glass-card-hover p-4 border-l-4 border-l-amber-500 space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-white" /> VIX Percentile
              </span>
              <div className="text-2xl font-mono font-bold text-white">
                {vixVal.toFixed(1)}%
              </div>
              <span className={`inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full border ${vixBadge.cls}`}>
                {vixBadge.label}
              </span>
            </div>

            {/* Metric 5: Credit Spread */}
            <div className="glass-card glass-card-hover p-4 border-l-4 border-l-red-500 space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase flex items-center gap-1.5">
                <ShieldAlert className="w-3.5 h-3.5 text-white" /> Credit Spread
              </span>
              <div className="text-2xl font-mono font-bold text-white">
                {creditVal.toFixed(2)}%
              </div>
              <span className={`inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full border ${creditBadge.cls}`}>
                {creditBadge.label}
              </span>
            </div>
          </div>

          {/* Educational Guide Expander */}
          <div className="glass-card p-4">
            <button
              onClick={() => setShowEduGuide(!showEduGuide)}
              className="w-full flex items-center justify-between text-xs font-semibold text-white hover:text-zinc-200 transition"
            >
              <div className="flex items-center space-x-2">
                <Info className="w-4 h-4 text-white" />
                <span>📖 Educational Guide: What are Market Regimes &amp; Stress Indicators?</span>
              </div>
              {showEduGuide ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            {showEduGuide && (
              <div className="mt-4 border-t border-slate-800 pt-4 text-xs text-slate-300 space-y-3 leading-relaxed">
                <p>
                  Financial markets move through distinct statistical economic phases (<strong>Regimes</strong>) driven by inflation, volatility, growth, credit, and monetary policy.
                  A Gaussian Hidden Markov Model (HMM) detects these phases by learning which combinations of 17 macroeconomic features tend to cluster together across 5,525 trading days of history (2003–2026).
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
                  <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1">
                    <strong className="text-white font-semibold">◌ Goldilocks &amp; Bull Regimes</strong>
                    <p className="text-slate-400">Low inflation, stable growth, suppressed VIX fear index, and tight credit spreads.</p>
                  </div>
                  <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1">
                    <strong className="text-white font-semibold">◌ Inflationary Expansion &amp; Peak</strong>
                    <p className="text-slate-400">Strong economic output coupled with rising inflation expectations and rate hike pressures.</p>
                  </div>
                  <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1">
                    <strong className="text-orange-400 font-semibold">🟠 Late Cycle &amp; Stagflation</strong>
                    <p className="text-slate-400">Slowing GDP momentum, inverted yield curve, and persistent monetary tightening.</p>
                  </div>
                  <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1">
                    <strong className="text-white font-semibold">◌ Recessionary Bear &amp; Crisis</strong>
                    <p className="text-slate-400">High VIX fear percentile (&gt;70%), widening corporate credit spreads (&gt;3%), and liquidations.</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ==================================================================== */}
          {/* TAB NAVIGATION BAR */}
          {/* ==================================================================== */}
          <div className="flex border-b border-slate-800 overflow-x-auto gap-2 text-xs font-semibold scrollbar-none">
            <button
              onClick={() => setActiveTab('charts')}
              className={`flex items-center space-x-2 py-3 px-4 border-b-2 transition whitespace-nowrap ${
                activeTab === 'charts'
                  ? 'border-cyan-400 text-cyan-400 bg-cyan-950/30'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>📊 Market &amp; Regime Charts</span>
            </button>
            <button
              onClick={() => setActiveTab('forecast')}
              className={`flex items-center space-x-2 py-3 px-4 border-b-2 transition whitespace-nowrap ${
                activeTab === 'forecast'
                  ? 'border-cyan-400 text-cyan-400 bg-cyan-950/30'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Zap className="w-4 h-4" />
              <span>🔮 Tomorrow's Forecast</span>
            </button>
            <button
              onClick={() => setActiveTab('dtw')}
              className={`flex items-center space-x-2 py-3 px-4 border-b-2 transition whitespace-nowrap ${
                activeTab === 'dtw'
                  ? 'border-cyan-400 text-cyan-400 bg-cyan-950/30'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Search className="w-4 h-4" />
              <span>🔍 DTW Trajectory Search</span>
            </button>
            <button
              onClick={() => setActiveTab('ai')}
              className={`flex items-center space-x-2 py-3 px-4 border-b-2 transition whitespace-nowrap ${
                activeTab === 'ai'
                  ? 'border-cyan-400 text-cyan-400 bg-cyan-950/30'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Bot className="w-4 h-4 text-white animate-pulse" />
              <span>🤖 AI Market Analyst</span>
            </button>
            <button
              onClick={() => setActiveTab('diagnostics')}
              className={`flex items-center space-x-2 py-3 px-4 border-b-2 transition whitespace-nowrap ${
                activeTab === 'diagnostics'
                  ? 'border-cyan-400 text-cyan-400 bg-cyan-950/30'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Sliders className="w-4 h-4" />
              <span>⚙️ Macro Drivers &amp; Model</span>
            </button>
          </div>

          {/* ==================================================================== */}
          {/* TAB 1: MARKET REGIME & TREND CHARTS */}
          {/* ==================================================================== */}
          {activeTab === 'charts' && (
            <div className="space-y-6">
              {/* Chart 1: Cumulative Market Index */}
              <div className="glass-card p-5 space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="font-bold text-sm text-slate-200 flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-white" />
                      Cumulative Market Growth Index (Base 100, 2003–Present)
                    </h3>
                    <p className="text-xs text-slate-400">
                      Points are color-coded by HMM detected regime. Vertical line indicates selected date ({selectedDate}).
                    </p>
                  </div>
                </div>
                <div className="h-80 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                      <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} interval={300} />
                      <YAxis dataKey="cumulative_market_index" stroke="#64748b" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(val: any, name: any, item: any) => [
                          `${Number(val).toFixed(1)} (Index)`,
                          item.payload.regime_name
                        ]}
                      />
                      <Scatter data={chartHistoryData} fill="#ffffff">
                        {chartHistoryData.map((entry, index) => {
                          const color = metadata?.regime_colors[entry.regime_name] || '#38bdf8';
                          const isSelected = entry.date === selectedDate;
                          return (
                            <Cell
                              key={`cell-${index}`}
                              fill={color}
                              r={isSelected ? 6 : 2}
                              stroke={isSelected ? '#ffffff' : 'none'}
                              strokeWidth={isSelected ? 2 : 0}
                            />
                          );
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Chart 2: 30-Day Rolling Market Return */}
              <div className="glass-card p-5 space-y-3">
                <div className="border-b border-slate-800 pb-3">
                  <h3 className="font-bold text-sm text-slate-200 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-white" />
                    30-Day Rolling Cumulative Return (%)
                  </h3>
                  <p className="text-xs text-slate-400">30-trading-day trailing return trajectory across historical market dates.</p>
                </div>
                <div className="h-56 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartHistoryData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                      <defs>
                        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ffffff" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#ffffff" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} interval={300} />
                      <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(val: any) => [`${Number(val).toFixed(2)}%`, '30d Return']}
                      />
                      <Area type="monotone" dataKey="rolling_30d_return" stroke="#ffffff" strokeWidth={1.5} fillOpacity={1} fill="url(#areaGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Chart 3: Interactive Indicator Explorer */}
              <div className="glass-card p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="font-bold text-sm text-slate-200 flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-white" />
                      Interactive Macro Indicator Time Series
                    </h3>
                    <p className="text-xs text-slate-400">Explore any of the 17 macroeconomic indicators across time.</p>
                  </div>
                  <div className="flex items-center space-x-3 text-xs">
                    <select
                      value={selectedIndicator}
                      onChange={(e) => setSelectedIndicator(e.target.value)}
                      className="bg-black text-white font-mono border border-white/15 rounded-full px-3 py-1.5 focus:outline-none focus:border-white"
                    >
                      {featureNames.map((fn) => (
                        <option key={fn} value={fn}>
                          {fn}
                        </option>
                      ))}
                    </select>
                    <label className="flex items-center space-x-1.5 text-slate-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showColorByRegime}
                        onChange={(e) => setShowColorByRegime(e.target.checked)}
                        className="rounded border-white/15 text-white focus:ring-0 bg-black cursor-pointer"
                      />
                      <span>Color by Regime</span>
                    </label>
                  </div>
                </div>

                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    {showColorByRegime ? (
                      <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                        <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} interval={300} />
                        <YAxis dataKey={selectedIndicator} stroke="#64748b" tick={{ fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                          formatter={(val: any, name: any, item: any) => [
                            `${Number(val).toFixed(4)}`,
                            `${selectedIndicator} (${item.payload.regime_name})`
                          ]}
                        />
                        <Scatter data={chartHistoryData}>
                          {chartHistoryData.map((entry, index) => {
                            const color = toMonochromeColor(metadata?.regime_colors[entry.regime_name] || '#ffffff');
                            return <Cell key={`cell-ind-${index}`} fill={color} r={2} />;
                          })}
                        </Scatter>
                      </ScatterChart>
                    ) : (
                      <LineChart data={chartHistoryData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                        <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} interval={300} />
                        <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                          formatter={(val: any) => [`${Number(val).toFixed(4)}`, selectedIndicator]}
                        />
                        <Line type="monotone" dataKey={selectedIndicator} stroke="#ffffff" dot={false} strokeWidth={1.5} />
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Grid: Probability Bar Chart + Frequency Pie Chart */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Probabilities on selected date */}
                <div className="glass-card p-5 space-y-3">
                  <div className="border-b border-slate-800 pb-3">
                    <h3 className="font-bold text-sm text-slate-200">🎯 Regime Posterior Probabilities on {inference.date}</h3>
                    <p className="text-xs text-slate-400">Model state probability vector for the selected target date.</p>
                  </div>
                  <div className="h-64 w-full pt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={inference.state_probabilities} margin={{ top: 10, right: 10, bottom: 30, left: 10 }}>
                        <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" interval={0} />
                        <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 100]} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                          formatter={(val: any) => [`${val}%`, 'Probability']}
                        />
                        <Bar dataKey="probability" radius={[6, 6, 0, 0]}>
                          {inference.state_probabilities.map((entry, index) => (
                            <Cell key={`cell-bar-${index}`} fill={toMonochromeColor(entry.color)} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Historical Regime Frequency Donut */}
                <div className="glass-card p-5 space-y-3">
                  <div className="border-b border-slate-800 pb-3">
                    <h3 className="font-bold text-sm text-slate-200">🥧 Historical Regime Frequency Distribution (2003–2026)</h3>
                    <p className="text-xs text-slate-400">Percentage breakdown of total trading days spent in each economic regime.</p>
                  </div>
                  <div className="h-64 w-full flex items-center justify-center">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={regimeFrequencies}
                          dataKey="count"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={55}
                          outerRadius={85}
                          paddingAngle={3}
                        >
                          {regimeFrequencies.map((entry, index) => (
                            <Cell key={`pie-cell-${index}`} fill={toMonochromeColor(entry.color)} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                          formatter={(val: any, name: any) => [`${val} days`, name]}
                        />
                        <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ==================================================================== */}
          {/* TAB 2: TOMORROW'S FORECAST */}
          {/* ==================================================================== */}
          {activeTab === 'forecast' && (
            <div className="space-y-6">
              {/* Ground Truth Realized Historical Shift Banner */}
              {inference.has_next_day && (
                <div
                  className={`p-4 rounded-xl border-l-8 space-y-1 ${
                    inference.actual_regime_changed
                      ? 'bg-white/10 border-l-white border border-white/20'
                      : 'bg-white/10 border-l-white border border-white/20'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    {inference.actual_regime_changed ? (
                      <AlertTriangle className="w-5 h-5 text-white" />
                    ) : (
                      <CheckCircle2 className="w-5 h-5 text-white" />
                    )}
                    <span className="font-bold text-sm uppercase tracking-wide text-slate-100">
                      {inference.actual_regime_changed
                        ? `⚡ REALIZED HISTORICAL REGIME SHIFT ON ${inference.actual_next_date}`
                        : `🔄 REALIZED HISTORICAL PERSISTENCE ON ${inference.actual_next_date}`}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 pl-7">
                    {inference.actual_regime_changed ? (
                      <>
                        State transitioned from <strong className="text-white">{inference.current_regime_name}</strong> &rarr;{' '}
                        <strong style={{ color: inference.actual_next_regime_color }}>{inference.actual_next_regime_name}</strong>
                      </>
                    ) : (
                      <>
                        Regime persisted as <strong style={{ color: inference.actual_next_regime_color }}>{inference.actual_next_regime_name}</strong> on the next trading day.
                      </>
                    )}
                  </p>
                </div>
              )}

              {/* Markov Projection Box */}
              <div
                className="glass-card p-5 border-l-4 space-y-2"
                style={{ borderLeftColor: inference.tomorrow_regime_color }}
              >
                <div className="flex items-center space-x-2">
                  <Zap className="w-5 h-5 text-white" />
                  <h3 className="font-bold text-base text-slate-100">
                    1-Day Forward Markov Chain Projection: <span style={{ color: inference.tomorrow_regime_color }}>{inference.tomorrow_regime_name}</span>
                  </h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 text-xs">
                  <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400">Daily State Persistence:</span>
                    <div className="text-base font-mono font-bold text-white">
                      {(inference.persistence_prob * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400">Transition Shift Hazard Rate:</span>
                    <div className="text-base font-mono font-bold text-white">
                      {(inference.transition_out_prob * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400">Top Candidate Target If Shift Occurs:</span>
                    <div className="text-sm font-bold text-purple-300 truncate">
                      {inference.top_transition_regime_name} ({(inference.top_transition_prob * 100).toFixed(1)}%)
                    </div>
                  </div>
                </div>
              </div>

              {/* Grid: Tomorrow Distribution & Heatmap */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Tomorrow's Probability Distribution */}
                <div className="glass-card p-5 space-y-3">
                  <div className="border-b border-slate-800 pb-3">
                    <h3 className="font-bold text-sm text-slate-200">Tomorrow's Projected Probability Distribution</h3>
                    <p className="text-xs text-slate-400">1-step forward Markov transition calculation vector.</p>
                  </div>
                  <div className="h-64 w-full pt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={inference.tomorrow_probabilities} margin={{ top: 10, right: 10, bottom: 30, left: 10 }}>
                        <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" interval={0} />
                        <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 100]} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                          formatter={(val: any) => [`${val}%`, 'Forecast Prob']}
                        />
                        <Bar dataKey="probability" radius={[6, 6, 0, 0]}>
                          {inference.tomorrow_probabilities.map((entry, index) => (
                            <Cell key={`cell-t-bar-${index}`} fill={toMonochromeColor(entry.color)} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Transition Heatmap Matrix */}
                <div className="glass-card p-5 space-y-3">
                  <div className="border-b border-slate-800 pb-3">
                    <h3 className="font-bold text-sm text-slate-200">State Transition Probability Matrix Heatmap (%)</h3>
                    <p className="text-xs text-slate-400">Rows: Today's State &rarr; Columns: Tomorrow's State</p>
                  </div>
                  {transitionData && (
                    <div className="overflow-x-auto pt-2">
                      <table className="w-full text-xs text-center border-collapse">
                        <thead>
                          <tr>
                            <th className="p-2 border border-slate-800 bg-slate-900 text-left text-slate-400 font-medium">State</th>
                            {transitionData.short_labels.map((lbl, idx) => (
                              <th key={idx} className="p-2 border border-slate-800 bg-slate-900 text-slate-300 font-medium text-[10px]" title={transitionData.states_labels[idx]}>
                                S{idx}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {transitionData.transition_matrix.map((row, rIdx) => (
                            <tr key={rIdx}>
                              <td className="p-2 border border-slate-800 bg-slate-900/80 text-left font-mono font-semibold text-slate-300 text-[11px]" title={transitionData.states_labels[rIdx]}>
                                S{rIdx}
                              </td>
                              {row.map((val, cIdx) => {
                                const intensity = Math.min(val / 100, 1);
                                const bgStyle = rIdx === cIdx
                                  ? `rgba(56, 189, 248, ${0.15 + intensity * 0.7})`
                                  : `rgba(99, 102, 241, ${intensity * 0.5})`;
                                return (
                                  <td
                                    key={cIdx}
                                    className="p-2 border border-slate-800 font-mono text-[11px]"
                                    style={{ backgroundColor: bgStyle, color: intensity > 0.4 ? '#ffffff' : '#cbd5e1' }}
                                  >
                                    {val.toFixed(1)}%
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>

              {/* State Persistence Table */}
              <div className="glass-card p-5 space-y-3">
                <div className="border-b border-slate-800 pb-3">
                  <h3 className="font-bold text-sm text-slate-200">📐 Daily State Persistence (Diagonal of Transition Matrix)</h3>
                  <p className="text-xs text-slate-400">Statistical expected duration (days) spent in each economic regime before shifting.</p>
                </div>
                {transitionData && (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-900 text-slate-400 uppercase font-semibold text-[10px] tracking-wider">
                        <tr>
                          <th className="p-3">Economic Regime</th>
                          <th className="p-3 font-mono text-right">Daily Persistence (%)</th>
                          <th className="p-3 font-mono text-right">Expected Duration (Trading Days)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {transitionData.persistence_table.map((row) => (
                          <tr key={row.state_id} className="hover:bg-slate-900/50 transition">
                            <td className="p-3 font-medium flex items-center space-x-2">
                              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: row.regime_color }} />
                              <span>{row.regime_name}</span>
                            </td>
                            <td className="p-3 font-mono font-bold text-white text-right">{row.daily_persistence}%</td>
                            <td className="p-3 font-mono text-white text-right">{row.avg_days} days</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ==================================================================== */}
          {/* TAB 3: DTW TRAJECTORY ALIGNMENT */}
          {/* ==================================================================== */}
          {activeTab === 'dtw' && (
            <div className="space-y-6">
              <div className="bg-blue-950/40 border border-blue-800/80 p-4 rounded-xl flex items-start space-x-3 text-xs text-blue-200">
                <ShieldAlert className="w-5 h-5 text-white shrink-0 mt-0.5" />
                <div>
                  <strong className="font-semibold text-white">Strict Past-Only Trajectory Search Policy:</strong>
                  <p className="mt-0.5 text-slate-300">
                    The Dynamic Time Warping (DTW) algorithm exclusively scans historical windows that ended <em>strictly before</em> the query start date to eliminate look-ahead bias and report ground-truth 30-day forward returns.
                  </p>
                </div>
              </div>

              {/* Slider Controls */}
              <div className="glass-card p-5 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-300">Trajectory Window Size:</span>
                    <span className="font-mono text-white">{dtwWindowSize} Trading Days</span>
                  </div>
                  <input
                    type="range"
                    min={15}
                    max={90}
                    step={5}
                    value={dtwWindowSize}
                    onChange={(e) => handleDtwSliderChange(Number(e.target.value), dtwTopK)}
                    className="w-full accent-cyan-400 bg-slate-900 h-2 rounded-lg cursor-pointer"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-300">Top Historical Matches:</span>
                    <span className="font-mono text-white">{dtwTopK} Matches</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={5}
                    step={1}
                    value={dtwTopK}
                    onChange={(e) => handleDtwSliderChange(dtwWindowSize, Number(e.target.value))}
                    className="w-full accent-cyan-400 bg-slate-900 h-2 rounded-lg cursor-pointer"
                  />
                </div>
              </div>

              {/* VIX Trajectory Alignment Line Chart */}
              <div className="glass-card p-5 space-y-3">
                <div className="border-b border-slate-800 pb-3 flex justify-between items-center">
                  <div>
                    <h3 className="font-bold text-sm text-slate-200">📈 VIX Trajectory Alignment (Query vs Matches)</h3>
                    <p className="text-xs text-slate-400">
                      Query Window: <code className="text-white">{dtwData?.query_start_date}</code> &rarr; <code className="text-white">{dtwData?.query_end_date}</code>
                    </p>
                  </div>
                </div>

                <div className="h-72 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={dtwChartData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                      <XAxis dataKey="day" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: 'Day in Sequence', position: 'bottom', offset: 0, fill: '#64748b', fontSize: 11 }} />
                      <YAxis stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: 'VIX Percentile', angle: -90, position: 'left', fill: '#64748b', fontSize: 11 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                      <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                      <Line type="monotone" dataKey="query" name="Query Window" stroke="#ffffff" strokeWidth={3} dot={{ r: 3 }} />
                      {dtwData?.matches.map((m, idx) => {
                        const colors = monochromePalette;
                        return (
                          <Line
                            key={`match-${idx}`}
                            type="monotone"
                            dataKey={`match_${idx + 1}`}
                            name={`Match #${m.rank} (${m.start_date} → ${m.end_date})`}
                            stroke={colors[idx % colors.length]}
                            strokeDasharray="4 4"
                            strokeWidth={1.5}
                            dot={false}
                          />
                        );
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Historical Match Cards */}
              <div className="space-y-4">
                <h3 className="font-bold text-sm text-slate-200">🏅 Top Historical Market Period Matches</h3>
                {dtwData?.matches.map((m) => {
                  const isPositive = (m.forward_30d_return || 0) > 0;
                  return (
                    <div key={m.rank} className="glass-card glass-card-hover p-5 border border-slate-800 space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                        <div className="flex items-center space-x-3">
                          <span className="w-8 h-8 rounded-full bg-white/10 border border-white/15 text-white font-mono font-bold flex items-center justify-center text-sm">
                            #{m.rank}
                          </span>
                          <div>
                            <span className="font-bold text-slate-100 text-sm">Historical Period Match</span>
                            <p className="text-xs text-slate-400 font-mono">
                              <code>{m.start_date}</code> &rarr; <code>{m.end_date}</code>
                            </p>
                          </div>
                        </div>
                        <span className="text-xs font-semibold px-3 py-1 rounded-full" style={{ backgroundColor: `${m.dominant_regime_color}20`, color: m.dominant_regime_color, border: `1px solid ${m.dominant_regime_color}40` }}>
                          {m.dominant_regime_name}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/60">
                          <span className="text-slate-400">Similarity Score</span>
                          <div className="text-base font-mono font-bold text-white">{m.similarity_score.toFixed(2)}%</div>
                        </div>
                        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/60">
                          <span className="text-slate-400">DTW Distance</span>
                          <div className="text-base font-mono font-bold text-white">{m.normalized_distance.toFixed(4)}</div>
                        </div>
                        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/60">
                          <span className="text-slate-400">Historical Regime</span>
                          <div className="text-xs font-bold text-slate-200 truncate mt-1">{m.dominant_regime_name}</div>
                        </div>
                        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/60">
                          <span className="text-slate-400">Next 30-Day Market Outcome</span>
                          <div className={`text-base font-mono font-bold flex items-center gap-1 ${isPositive ? 'text-white' : 'text-white'}`}>
                            {isPositive ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                            {m.forward_30d_return !== null ? `${m.forward_30d_return > 0 ? '+' : ''}${m.forward_30d_return.toFixed(2)}%` : 'N/A'}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ==================================================================== */}
          {/* TAB 4: AI MARKET ANALYST REPORT */}
          {/* ==================================================================== */}
          {activeTab === 'ai' && (
            <div className="space-y-6">
              <div className="glass-card p-5 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div>
                  <h3 className="font-bold text-sm text-slate-100 flex items-center gap-2">
                    <Bot className="w-5 h-5 text-white" />
                    AI Market Analyst Macro Executive Commentary
                  </h3>
                  <p className="text-xs text-slate-400">
                    Automated macro report synthesized from HMM posterior state probabilities, macro feature drivers, and DTW matches.
                  </p>
                </div>
                <button
                  onClick={generateAiReport}
                  disabled={loadingAi}
                  className="px-5 py-2.5 rounded-full bg-black text-white border border-white/15 font-semibold text-xs transition flex items-center space-x-2 shrink-0 disabled:opacity-50 cursor-pointer"
                >
                  <RefreshCw className={`w-4 h-4 ${loadingAi ? 'animate-spin' : ''}`} />
                  <span>{loadingAi ? 'Synthesizing...' : '🤖 Generate AI Commentary'}</span>
                </button>
              </div>

              {aiReport ? (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2 space-y-4">
                    {/* Executive Overview */}
                    <div className="glass-card p-5 space-y-2 border-l-4 border-l-cyan-500">
                      <h4 className="font-bold text-sm text-white flex items-center gap-2">
                        <FileText className="w-4 h-4" /> 📌 Executive Market Overview
                      </h4>
                      <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-line">{aiReport.overview}</p>
                    </div>

                    {/* Tomorrow's Outlook */}
                    <div className="glass-card p-5 space-y-2 border-l-4 border-l-emerald-500">
                      <h4 className="font-bold text-sm text-white flex items-center gap-2">
                        <Zap className="w-4 h-4" /> 🔮 Tomorrow's Outlook &amp; Regime Persistence
                      </h4>
                      <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-line">{aiReport.outlook}</p>
                    </div>

                    {/* Historical Analogs */}
                    <div className="glass-card p-5 space-y-2 border-l-4 border-l-amber-500">
                      <h4 className="font-bold text-sm text-white flex items-center gap-2">
                        <Search className="w-4 h-4" /> 📜 Historical Analogs &amp; Precedent Analysis
                      </h4>
                      <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-line">{aiReport.analogs}</p>
                    </div>
                  </div>

                  {/* Drivers & Radar Column */}
                  <div className="space-y-6">
                    {/* Drivers List */}
                    <div className="glass-card p-5 space-y-3">
                      <h4 className="font-bold text-sm text-slate-200">📊 Macroeconomic Feature Drivers</h4>
                      <div className="text-xs text-slate-300 leading-relaxed whitespace-pre-line bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                        {aiReport.drivers}
                      </div>
                    </div>

                    {/* Radar / Spider Chart */}
                    <div className="glass-card p-5 space-y-3">
                      <h4 className="font-bold text-sm text-slate-200">🕸️ Macro Risk Radar</h4>
                      <div className="h-64 w-full pt-1">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart data={inference.radar_data}>
                            <PolarGrid stroke="#334155" />
                            <PolarAngleAxis dataKey="feature" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                            <PolarRadiusAxis angle={30} domain={[0, 1]} stroke="#475569" tick={false} />
                            <Radar name="Risk Index" dataKey="value" stroke="#ffffff" fill="#ffffff" fillOpacity={0.25} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="glass-card p-12 text-center text-slate-400 space-y-3">
                  <Bot className="w-12 h-12 text-white/50 mx-auto" />
                  <p className="text-sm">Click <strong>Generate AI Commentary</strong> above to synthesize the executive analyst report for <strong>{inference.date}</strong>.</p>
                </div>
              )}
            </div>
          )}

          {/* ==================================================================== */}
          {/* TAB 5: MACRO DRIVERS & MODEL METADATA */}
          {/* ==================================================================== */}
          {activeTab === 'diagnostics' && (
            <div className="space-y-6">
              {/* Standardized Feature Vector Bar Chart */}
              <div className="glass-card p-5 space-y-3">
                <div className="border-b border-slate-800 pb-3">
                  <h3 className="font-bold text-sm text-slate-200">🔬 Full Macroeconomic Feature Vector — {inference.date} (Standardized)</h3>
                  <p className="text-xs text-slate-400">Values represent relative macro feature deviations on the selected target date.</p>
                </div>
                <div className="h-72 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={Object.entries(inference.macro_row)
                        .filter(([k]) => featureNames.includes(k))
                        .map(([k, v]) => ({ feature: k, value: Number(v) }))}
                      margin={{ top: 10, right: 10, bottom: 40, left: 10 }}
                    >
                      <XAxis dataKey="feature" stroke="#64748b" tick={{ fontSize: 9 }} angle={-35} textAnchor="end" interval={0} />
                      <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {Object.entries(inference.macro_row)
                          .filter(([k]) => featureNames.includes(k))
                          .map((entry, index) => {
                            const val = Number(entry[1]);
                            const color = val > 0.5 ? '#ffffff' : val > -0.5 ? '#d4d4d4' : '#111111';
                            return <Cell key={`cell-feat-${index}`} fill={color} />;
                          })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Statistics & Regime List Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Dataset Stats */}
                <div className="glass-card p-5 space-y-3">
                  <h4 className="font-bold text-sm text-slate-200 border-b border-slate-800 pb-2">📋 Dataset Statistics</h4>
                  <ul className="text-xs space-y-2 text-slate-300">
                    <li className="flex justify-between">
                      <span className="text-slate-400">Total Observations:</span>
                      <span className="font-mono">{metadata?.metrics.n_samples?.toLocaleString()} trading days (2003–2026)</span>
                    </li>
                    <li className="flex justify-between">
                      <span className="text-slate-400">Total Features:</span>
                      <span className="font-mono">{metadata?.n_features} indicators</span>
                    </li>
                    <li className="flex justify-between">
                      <span className="text-slate-400">Date Range:</span>
                      <span className="font-mono">{availableDates[0]} &rarr; {availableDates[availableDates.length - 1]}</span>
                    </li>
                    <li className="flex justify-between">
                      <span className="text-slate-400">Covariance Type:</span>
                      <span className="font-mono">Full</span>
                    </li>
                  </ul>
                </div>

                {/* Empirical Regime List */}
                <div className="glass-card p-5 space-y-3">
                  <h4 className="font-bold text-sm text-slate-200 border-b border-slate-800 pb-2">🏷️ Dynamic Empirical Regime Mapping</h4>
                  <div className="space-y-2 text-xs">
                    {metadata &&
                      Object.entries(metadata.regime_mapping).map(([sId, regNm]) => (
                        <div key={sId} className="flex items-center space-x-2">
                          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: metadata.regime_colors[regNm] || '#95a5a6' }} />
                          <span className="font-mono font-bold text-slate-300">State {sId}:</span>
                          <span className="text-slate-400">{regNm}</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>

              {/* JSON Expanders */}
              <div className="space-y-3">
                <details className="glass-card p-4 rounded-xl cursor-pointer text-xs">
                  <summary className="font-semibold text-white hover:text-zinc-200">📂 Raw Feature Values for Selected Date ({inference.date})</summary>
                  <pre className="mt-3 p-3 bg-black rounded-lg border border-white/10 text-[11px] font-mono text-white overflow-x-auto">
                    {JSON.stringify(inference.macro_row, null, 2)}
                  </pre>
                </details>

                <details className="glass-card p-4 rounded-xl cursor-pointer text-xs">
                  <summary className="font-semibold text-white hover:text-zinc-200">📂 HMM Model Metadata JSON</summary>
                  <pre className="mt-3 p-3 bg-black rounded-lg border border-white/10 text-[11px] font-mono text-white overflow-x-auto">
                    {JSON.stringify(metadata, null, 2)}
                  </pre>
                </details>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* FOOTER */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 text-center text-xs text-slate-500">
        Monolith Production MVP &middot; Gaussian HMM Market Regime Intelligence &middot; React + FastAPI Integration
      </footer>
    </div>
  );
}
