import React, { useState, useEffect } from 'react';
import {
  History as HistoryIcon,
  Search,
  Filter,
  Download,
  Eye,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  RefreshCw
} from 'lucide-react';
import { getHistory, getAnalysisById } from '../services/api';
import { AnalysisHistoryItem, FullAnalysisResult } from '../types';

interface HistoryPageProps {
  onSelectScan: (analysis: FullAnalysisResult) => void;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({ onSelectScan }) => {
  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalityFilter, setModalityFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [inspectingId, setInspectingId] = useState<number | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const data = await getHistory(modalityFilter, statusFilter);
      setHistory(data);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [modalityFilter, statusFilter]);

  const handleInspect = async (id: number) => {
    setInspectingId(id);
    try {
      const full = await getAnalysisById(id);
      onSelectScan(full);
    } catch (err) {
      console.error('Failed to load analysis details:', err);
    } finally {
      setInspectingId(null);
    }
  };

  const filteredHistory = history.filter((item) =>
    item.patient_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.predicted_class.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b-2 border-slate-300 pb-4">
        <div>
          <h2 className="text-2xl font-black text-black tracking-tight">Analysis Archive &amp; History</h2>
          <p className="text-sm font-bold text-black mt-1">
            Permanent SQLite database records of patient scans, classification findings, and generated PDF reports.
          </p>
        </div>

        <button
          onClick={fetchHistory}
          className="flex items-center gap-2 text-xs text-white bg-blue-700 hover:bg-blue-800 border border-blue-900 font-black px-5 py-2.5 rounded-xl transition-all self-start sm:self-auto shadow-md cursor-pointer"
        >
          <RefreshCw className="w-4 h-4 text-white" />
          <span className="text-white">Refresh Database</span>
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white p-4 rounded-2xl border-2 border-slate-300 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4 text-xs">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-black absolute left-3.5 top-3.5" />
          <input
            type="text"
            placeholder="Search patient reference ID or tumor type..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 rounded-xl border-2 border-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600 bg-white font-black text-black placeholder:text-slate-600"
          />
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-black font-black">Modality:</span>
            <select
              value={modalityFilter}
              onChange={(e) => setModalityFilter(e.target.value)}
              className="bg-slate-100 border-2 border-slate-400 rounded-lg px-3 py-2 font-black text-black"
            >
              <option value="ALL">All Modalities</option>
              <option value="MRI">Brain MRI</option>
              <option value="CT">Head CT</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-black font-black">Status:</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-100 border-2 border-slate-400 rounded-lg px-3 py-2 font-black text-black"
            >
              <option value="ALL">All Findings</option>
              <option value="positive">Tumor Positive</option>
              <option value="negative">Normal / Negative</option>
              <option value="abstain">Inconclusive / Abstain</option>
            </select>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-2xl border-2 border-slate-300 shadow-sm overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-black font-bold">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-blue-800" />
            <p className="text-xs font-black">Loading database records...</p>
          </div>
        ) : filteredHistory.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-200 border-b-2 border-slate-400 text-black">
                <tr>
                  <th className="py-3 px-4 font-black">Analysis #</th>
                  <th className="py-3 px-4 font-black">Patient Reference ID</th>
                  <th className="py-3 px-4 font-black">Modality</th>
                  <th className="py-3 px-4 font-black">Predicted Finding</th>
                  <th className="py-3 px-4 font-black">Confidence</th>
                  <th className="py-3 px-4 font-black">Fuzzy Uncertainty</th>
                  <th className="py-3 px-4 font-black">Expert Review</th>
                  <th className="py-3 px-4 font-black">Date Analyzed</th>
                  <th className="py-3 px-4 font-black text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y border-slate-300">
                {filteredHistory.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-100 transition-colors">
                    <td className="py-3.5 px-4 font-mono font-black text-black">#{item.id}</td>
                    <td className="py-3.5 px-4 font-black text-black">{item.patient_id}</td>
                    <td className="py-3.5 px-4">
                      <span className="px-2 py-0.5 rounded text-[11px] font-black bg-slate-200 text-black border border-slate-400">
                        {item.modality}
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      <span className={`font-black ${item.tumor_detected ? 'text-red-700' : 'text-emerald-700'}`}>
                        {item.predicted_class}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-black text-black">
                      {Math.round(item.confidence_score * 100)}%
                    </td>
                    <td className="py-3.5 px-4">
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-[10px] font-black border ${
                          item.fuzzy_uncertainty_level === 'Low Uncertainty'
                            ? 'bg-emerald-100 text-black border-emerald-400'
                            : item.fuzzy_uncertainty_level === 'Moderate Uncertainty'
                            ? 'bg-amber-100 text-black border-amber-400'
                            : 'bg-red-100 text-black border-red-400'
                        }`}
                      >
                        {item.fuzzy_uncertainty_level}
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      {item.needs_expert_review ? (
                        <span className="text-red-700 font-black">Mandatory</span>
                      ) : (
                        <span className="text-black font-bold">Routine</span>
                      )}
                    </td>
                    <td className="py-3.5 px-4 text-black font-bold">{item.created_at}</td>
                    <td className="py-3.5 px-4 text-right space-x-2">
                      <button
                        onClick={() => handleInspect(item.id)}
                        disabled={inspectingId === item.id}
                        className="inline-flex items-center gap-1 text-black font-black px-3 py-1.5 rounded-lg bg-blue-100 hover:bg-blue-200 border-2 border-blue-400 shadow-sm cursor-pointer"
                      >
                        <Eye className="w-3.5 h-3.5 text-blue-900" />
                        <span>Inspect</span>
                      </button>

                      {item.report_pdf_url && (
                        <a
                          href={item.report_pdf_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-black font-black px-3 py-1.5 rounded-lg bg-slate-200 hover:bg-slate-300 border-2 border-slate-400 shadow-sm cursor-pointer"
                        >
                          <Download className="w-3.5 h-3.5 text-black" />
                          <span>PDF</span>
                        </a>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-12 text-center text-black font-bold text-xs">
            No historical records match the selected filters.
          </div>
        )}
      </div>
    </div>
  );
};
