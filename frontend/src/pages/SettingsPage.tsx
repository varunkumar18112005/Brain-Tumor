import React, { useState } from 'react';
import {
  Settings as SettingsIcon,
  Sliders,
  Cpu,
  Layers,
  ShieldAlert,
  Database,
  Check,
  Info
} from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const [mcSamples, setMcSamples] = useState(15);
  const [abstentionThreshold, setAbstentionThreshold] = useState(0.55);
  const [expertReviewThreshold, setExpertReviewThreshold] = useState(0.40);
  const [gaussianKernel, setGaussianKernel] = useState(3);
  const [claheClip, setClaheClip] = useState(2.0);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      <div className="flex items-center justify-between border-b-2 border-slate-300 pb-4">
        <div>
          <h2 className="text-2xl font-black text-black tracking-tight">System Settings &amp; Configuration</h2>
          <p className="text-sm font-bold text-black mt-1">
            Tune uncertainty thresholds, Monte Carlo Dropout passes, and preprocessing filter parameters.
          </p>
        </div>

        <button
          onClick={handleSave}
          className="flex items-center gap-2 bg-blue-700 hover:bg-blue-800 text-white font-black text-xs px-6 py-3 rounded-xl border border-blue-900 shadow-md transition-all cursor-pointer"
        >
          <Check className="w-4 h-4 text-white" />
          <span className="text-white">Save Settings</span>
        </button>
      </div>

      {saved && (
        <div className="p-3.5 rounded-xl bg-emerald-100 border-2 border-emerald-400 text-black text-xs font-black flex items-center gap-2">
          <Check className="w-4 h-4 text-emerald-800" />
          <span>Configuration parameters updated for future scan analyses.</span>
        </div>
      )}

      {/* Uncertainty & Probabilistic Parameters */}
      <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-6">
        <h3 className="text-base font-black text-black flex items-center gap-2">
          <Cpu className="w-5 h-5 text-blue-900" />
          <span>Uncertainty &amp; Monte Carlo Dropout Settings</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
          <div>
            <label className="block font-black text-black mb-1">
              Monte Carlo Samples (T): <span className="text-blue-900 font-black">{mcSamples} passes</span>
            </label>
            <input
              type="range"
              min="5"
              max="30"
              value={mcSamples}
              onChange={(e) => setMcSamples(parseInt(e.target.value))}
              className="w-full accent-blue-700 mt-2"
            />
            <p className="text-xs font-bold text-black mt-1.5">
              Number of stochastic forward passes with dropout active for predictive variance.
            </p>
          </div>

          <div>
            <label className="block font-black text-black mb-1">
              Abstention Threshold: <span className="text-blue-900 font-black">{abstentionThreshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.40"
              max="0.80"
              step="0.05"
              value={abstentionThreshold}
              onChange={(e) => setAbstentionThreshold(parseFloat(e.target.value))}
              className="w-full accent-blue-700 mt-2"
            />
            <p className="text-xs font-bold text-black mt-1.5">
              If confidence &lt; threshold, the system enforces formal diagnostic abstention.
            </p>
          </div>

          <div>
            <label className="block font-black text-black mb-1">
              Expert Review Trigger: <span className="text-blue-900 font-black">{expertReviewThreshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.30"
              max="0.60"
              step="0.05"
              value={expertReviewThreshold}
              onChange={(e) => setExpertReviewThreshold(parseFloat(e.target.value))}
              className="w-full accent-blue-700 mt-2"
            />
            <p className="text-xs font-bold text-black mt-1.5">
              Flags scan for mandatory human radiological corroboration.
            </p>
          </div>
        </div>
      </div>

      {/* Preprocessing Parameters */}
      <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-6">
        <h3 className="text-base font-black text-black flex items-center gap-2">
          <Layers className="w-5 h-5 text-teal-800" />
          <span>Preprocessing Filter Parameters</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          <div>
            <label className="block font-black text-black mb-1">
              Gaussian Denoising Kernel: <span className="text-blue-900 font-black">{gaussianKernel}x{gaussianKernel}</span>
            </label>
            <select
              value={gaussianKernel}
              onChange={(e) => setGaussianKernel(parseInt(e.target.value))}
              className="w-full p-2.5 rounded-xl border-2 border-slate-400 bg-white font-black text-black"
            >
              <option value={3}>3x3 (Standard Brain MRI)</option>
              <option value={5}>5x5 (Higher Noise Suppression)</option>
            </select>
          </div>

          <div>
            <label className="block font-black text-black mb-1">
              CLAHE Contrast Clip Limit: <span className="text-blue-900 font-black">{claheClip.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min="1.0"
              max="4.0"
              step="0.5"
              value={claheClip}
              onChange={(e) => setClaheClip(parseFloat(e.target.value))}
              className="w-full accent-blue-700 mt-2"
            />
            <p className="text-xs font-bold text-black mt-1.5">
              Prevents over-amplification of background noise in dark intracranial regions.
            </p>
          </div>
        </div>
      </div>

      {/* Legal & Version Information */}
      <div className="bg-slate-900 text-white p-6 rounded-2xl border-2 border-slate-700 space-y-3 text-xs leading-relaxed">
        <h4 className="text-sm font-black text-white flex items-center gap-2">
          <Info className="w-4 h-4 text-teal-400" />
          <span className="text-white">Academic System Specifications &amp; Regulatory Status</span>
        </h4>
        <p className="font-bold text-slate-100">
          <b className="text-white">Project Title:</b> Intelligent Medical Image-Based Tumor Detection and Uncertainty Assessment System<br />
          <b className="text-white">Architecture:</b> PyTorch 2D Convolutional U-Net + Deep CNN with Monte Carlo Dropout + Mamdani Fuzzy Inference Engine.<br />
          <b className="text-white">Database:</b> SQLite with SQLAlchemy ORM.<br />
          <b className="text-white">Report Generator:</b> ReportLab Clinical PDF Engine.<br />
          <b className="text-white">Regulatory Status:</b> Academic research prototype and decision support demonstration. NOT approved as a Primary Diagnostic Device.
        </p>
      </div>
    </div>
  );
};
