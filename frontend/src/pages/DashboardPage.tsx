import React, { useEffect, useState } from 'react';
import {
  FileText,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  TrendingUp,
  Cpu,
  Layers,
  ArrowRight,
  ShieldAlert,
  Database
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { getHistory, getModelStatus } from '../services/api';
import { AnalysisHistoryItem, ModelStatus } from '../types';

interface DashboardProps {
  onNavigate: (tab: string) => void;
  onSelectScan: (id: number) => void;
}

export const DashboardPage: React.FC<DashboardProps> = ({ onNavigate, onSelectScan }) => {
  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [histData, statusData] = await Promise.all([
          getHistory(),
          getModelStatus()
        ]);
        setHistory(histData);
        setModelStatus(statusData);
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const totalScans = history.length;
  const tumorPositive = history.filter((h) => h.tumor_detected).length;
  const tumorNegative = history.filter((h) => !h.tumor_detected && !h.abstain_recommended).length;
  const highUncertainty = history.filter((h) => h.fuzzy_uncertainty_level === 'High Uncertainty').length;
  const inconclusive = history.filter((h) => h.abstain_recommended).length;

  const pieData = [
    { name: 'Tumor Positive', value: tumorPositive, color: '#dc2626' },
    { name: 'Normal / Negative', value: tumorNegative, color: '#16a34a' },
    { name: 'Inconclusive / Abstain', value: inconclusive, color: '#ea580c' },
  ].filter(d => d.value > 0);

  // Activity data
  const activityData = history.slice(0, 7).reverse().map((h, i) => ({
    name: `Scan #${h.id}`,
    confidence: Math.round(h.confidence_score * 100),
    uncertainty: Math.round(h.fuzzy_uncertainty_score * 100)
  }));

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Top Banner with Guaranteed Solid Dark Navy Background */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900 border-2 border-slate-700 text-white p-7 rounded-2xl shadow-xl">
        <div>
          <h2 className="text-2xl font-black tracking-tight text-white">Clinical AI Decision Support Dashboard</h2>
          <p className="text-slate-100 text-sm mt-1.5 max-w-2xl font-bold leading-relaxed">
            Multi-modal MRI &amp; CT deep learning tumor segmentation, Bayesian Monte Carlo Dropout uncertainty estimation,
            and Mamdani fuzzy reasoning.
          </p>
        </div>
        <button
          onClick={() => onNavigate('new-scan')}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-black px-6 py-3 rounded-xl shadow-md border border-blue-400 transition-all whitespace-nowrap self-start md:self-auto text-sm cursor-pointer"
        >
          <span>Start New Scan Analysis</span>
          <ArrowRight className="w-5 h-5 text-white" />
        </button>
      </div>

      {/* Metric Cards Grid - High Contrast Black Text */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-white p-5 rounded-xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-black text-black uppercase tracking-wider">Total Scans</span>
            <FileText className="w-5 h-5 text-blue-800" />
          </div>
          <div className="mt-4">
            <span className="text-3xl font-black text-black">{totalScans}</span>
            <p className="text-xs font-bold text-black mt-1">Processed in SQLite</p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-black text-black uppercase tracking-wider">Tumor Positive</span>
            <AlertTriangle className="w-5 h-5 text-red-700" />
          </div>
          <div className="mt-4">
            <span className="text-3xl font-black text-red-700">{tumorPositive}</span>
            <p className="text-xs font-bold text-black mt-1">Glioma / Meningioma / Pituitary</p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-black text-black uppercase tracking-wider">Normal Cases</span>
            <CheckCircle2 className="w-5 h-5 text-emerald-700" />
          </div>
          <div className="mt-4">
            <span className="text-3xl font-black text-emerald-700">{tumorNegative}</span>
            <p className="text-xs font-bold text-black mt-1">No focal lesion detected</p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-black text-black uppercase tracking-wider">High Uncertainty</span>
            <ShieldAlert className="w-5 h-5 text-amber-600" />
          </div>
          <div className="mt-4">
            <span className="text-3xl font-black text-amber-700">{highUncertainty}</span>
            <p className="text-xs font-bold text-black mt-1">Fuzzy score &gt; 0.65</p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-black text-black uppercase tracking-wider">Inconclusive / Abstain</span>
            <HelpCircle className="w-5 h-5 text-slate-800" />
          </div>
          <div className="mt-4">
            <span className="text-3xl font-black text-black">{inconclusive}</span>
            <p className="text-xs font-bold text-black mt-1">Automated diagnosis held</p>
          </div>
        </div>
      </div>

      {/* Charts & Activity Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Chart */}
        <div className="lg:col-span-2 bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-black text-black">Prediction Confidence &amp; Uncertainty Trends</h3>
              <p className="text-xs font-bold text-black">Recent sequential scan analysis metrics</p>
            </div>
            <span className="px-3 py-1 text-xs font-black rounded-md bg-slate-200 text-black border border-slate-400">
              Recent 7 Scans
            </span>
          </div>

          {activityData.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={activityData}>
                  <defs>
                    <linearGradient id="colorConf" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0284c7" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#0284c7" stopOpacity={0.05} />
                    </linearGradient>
                    <linearGradient id="colorUnc" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#d97706" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#d97706" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" stroke="#000000" fontSize={12} tick={{ fill: '#000000', fontWeight: 'bold' }} />
                  <YAxis stroke="#000000" fontSize={12} domain={[0, 100]} tick={{ fill: '#000000', fontWeight: 'bold' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#000000', color: '#000000', fontWeight: 'bold' }} />
                  <Area type="monotone" dataKey="confidence" name="Confidence %" stroke="#0369a1" strokeWidth={3} fillOpacity={1} fill="url(#colorConf)" />
                  <Area type="monotone" dataKey="uncertainty" name="Uncertainty %" stroke="#b45309" strokeWidth={3} fillOpacity={1} fill="url(#colorUnc)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-black">
              <TrendingUp className="w-10 h-10 mb-2 text-slate-800" />
              <p className="text-sm font-black text-black">No analysis activity recorded yet.</p>
              <button
                onClick={() => onNavigate('new-scan')}
                className="mt-3 text-xs text-blue-900 font-black hover:underline"
              >
                Upload a scan to view graphs →
              </button>
            </div>
          )}
        </div>

        {/* Diagnosis Distribution */}
        <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <div>
            <h3 className="text-base font-black text-black">Diagnostic Distribution</h3>
            <p className="text-xs font-bold text-black mb-4">Breakdown of analyzed findings</p>
            {pieData.length > 0 ? (
              <div className="h-48 w-full flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={65}
                      innerRadius={40}
                      paddingAngle={4}
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#000000', color: '#000000', fontWeight: 'bold' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-48 flex items-center justify-center text-black font-black text-xs">
                Zero scans analyzed.
              </div>
            )}
          </div>

          <div className="space-y-2.5 pt-4 border-t-2 border-slate-200 text-xs">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 font-black text-black"><span className="w-3 h-3 rounded-full bg-red-700"></span> Positive:</span>
              <span className="font-black text-black text-sm">{tumorPositive}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 font-black text-black"><span className="w-3 h-3 rounded-full bg-emerald-700"></span> Negative:</span>
              <span className="font-black text-black text-sm">{tumorNegative}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 font-black text-black"><span className="w-3 h-3 rounded-full bg-amber-600"></span> Inconclusive:</span>
              <span className="font-black text-black text-sm">{inconclusive}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model & Dataset Status Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Architecture Status */}
        <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-blue-900" />
              <h3 className="text-base font-black text-black">Deep Learning Model Engines</h3>
            </div>
            <button
              onClick={() => onNavigate('performance')}
              className="text-xs text-blue-900 font-black hover:underline"
            >
              View Metrics →
            </button>
          </div>

          <div className="space-y-3">
            {modelStatus?.models.map((m, idx) => (
              <div key={idx} className="p-3.5 rounded-xl border-2 border-slate-300 bg-slate-50 flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-black text-sm text-black">{m.name}</span>
                    <span className="text-[10px] font-black px-2 py-0.5 rounded bg-slate-200 text-black border border-slate-400">
                      {m.modality}
                    </span>
                  </div>
                  <p className="text-xs font-bold text-black mt-1">{m.architecture}</p>
                </div>
                <div>
                  {m.is_trained ? (
                    <span className="px-3 py-1 text-xs font-black rounded-full bg-emerald-100 text-emerald-950 border border-emerald-400">
                      Trained &amp; Ready
                    </span>
                  ) : (
                    <span className="px-3 py-1 text-xs font-black rounded-full bg-amber-100 text-amber-950 border border-amber-400">
                      Baseline Prior
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Medical Imaging Datasets */}
        <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-teal-800" />
              <h3 className="text-base font-black text-black">Integrated Medical Datasets</h3>
            </div>
            <span className="text-xs font-black text-black">Research Frameworks</span>
          </div>

          <div className="space-y-3">
            <div className="p-3.5 rounded-xl border-2 border-slate-300 bg-slate-50">
              <div className="flex items-center justify-between">
                <span className="font-black text-sm text-black">BraTS Challenge Dataset</span>
                <span className="text-[10px] font-black px-2 py-0.5 rounded bg-teal-100 text-black border border-teal-400">MRI 3D</span>
              </div>
              <p className="text-xs font-bold text-black mt-1">Multi-parametric MRI for Glioma sub-region segmentation.</p>
            </div>

            <div className="p-3.5 rounded-xl border-2 border-slate-300 bg-slate-50">
              <div className="flex items-center justify-between">
                <span className="font-black text-sm text-black">Figshare Brain Tumor Dataset</span>
                <span className="text-[10px] font-black px-2 py-0.5 rounded bg-blue-100 text-black border border-blue-400">MRI T1-CE</span>
              </div>
              <p className="text-xs font-bold text-black mt-1">3,064 T1-weighted contrast enhanced slices for Glioma, Meningioma, Pituitary.</p>
            </div>

            <div className="p-3.5 rounded-xl border-2 border-slate-300 bg-slate-50">
              <div className="flex items-center justify-between">
                <span className="font-black text-sm text-black">CQ500 Head CT Dataset</span>
                <span className="text-[10px] font-black px-2 py-0.5 rounded bg-purple-100 text-black border border-purple-400">CT Non-Contrast</span>
              </div>
              <p className="text-xs font-bold text-black mt-1">Head CT space-occupying lesions, hemorrhages, and mass effect evaluation.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Analyses Table */}
      <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-black text-black">Recent Patient Scans</h3>
            <p className="text-xs font-bold text-black">Latest analyses recorded in system database</p>
          </div>
          <button
            onClick={() => onNavigate('history')}
            className="text-xs text-blue-900 font-black hover:underline"
          >
            Full History →
          </button>
        </div>

        {history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-200 border-b-2 border-slate-400 text-black">
                <tr>
                  <th className="py-2.5 px-4 font-black">Patient ID</th>
                  <th className="py-2.5 px-4 font-black">Modality</th>
                  <th className="py-2.5 px-4 font-black">Predicted Class</th>
                  <th className="py-2.5 px-4 font-black">Confidence</th>
                  <th className="py-2.5 px-4 font-black">Fuzzy Uncertainty</th>
                  <th className="py-2.5 px-4 font-black">Expert Review</th>
                  <th className="py-2.5 px-4 font-black">Date</th>
                  <th className="py-2.5 px-4 font-black text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y border-slate-300">
                {history.slice(0, 5).map((row) => (
                  <tr key={row.id} className="hover:bg-slate-100 transition-colors">
                    <td className="py-3 px-4 font-black text-black">{row.patient_id}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded text-[11px] font-black bg-slate-200 text-black border border-slate-400">
                        {row.modality}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`font-black ${row.tumor_detected ? 'text-red-700' : 'text-emerald-700'}`}>
                        {row.predicted_class}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-black text-black">{Math.round(row.confidence_score * 100)}%</td>
                    <td className="py-3 px-4">
                      <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-black ${
                        row.fuzzy_uncertainty_level === 'Low Uncertainty'
                          ? 'bg-emerald-100 text-black border border-emerald-400'
                          : row.fuzzy_uncertainty_level === 'Moderate Uncertainty'
                          ? 'bg-amber-100 text-black border border-amber-400'
                          : 'bg-red-100 text-black border border-red-400'
                      }`}>
                        {row.fuzzy_uncertainty_level}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {row.needs_expert_review ? (
                        <span className="text-red-700 font-black">Mandatory</span>
                      ) : (
                        <span className="text-black font-bold">Routine</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-black font-bold">{row.created_at}</td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => onSelectScan(row.id)}
                        className="text-blue-900 hover:text-black font-black underline cursor-pointer"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-8 text-center text-black font-bold text-xs">
            No analyses recorded yet. Upload a scan from the "New Scan Analysis" tab.
          </div>
        )}
      </div>
    </div>
  );
};
