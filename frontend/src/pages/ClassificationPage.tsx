import React from 'react';
import {
  BrainCircuit,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Download,
  ShieldCheck,
  ShieldAlert,
  Sliders,
  Activity,
  FileCheck,
  ArrowRight
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { FullAnalysisResult } from '../types';

interface ClassificationPageProps {
  currentAnalysis: FullAnalysisResult | null;
  onNavigate: (tab: string) => void;
}

export const ClassificationPage: React.FC<ClassificationPageProps> = ({ currentAnalysis, onNavigate }) => {
  if (!currentAnalysis) {
    return (
      <div className="p-12 max-w-4xl mx-auto text-center space-y-5">
        <div className="w-16 h-16 rounded-2xl bg-slate-200 text-black flex items-center justify-center mx-auto border-2 border-slate-400">
          <BrainCircuit className="w-8 h-8 text-black" />
        </div>
        <h3 className="text-xl font-black text-black">No Active Analysis Available</h3>
        <p className="text-sm font-bold text-black max-w-md mx-auto leading-relaxed">
          Run an analysis on a new scan to inspect CNN classification, Monte Carlo Dropout uncertainty, and fuzzy reasoning rules.
        </p>
        <button
          onClick={() => onNavigate('new-scan')}
          className="inline-flex items-center gap-2 bg-blue-700 hover:bg-blue-800 text-white px-6 py-3 rounded-xl text-sm font-black transition-all shadow-md cursor-pointer border border-blue-900"
        >
          <span>Start New Scan</span>
          <ArrowRight className="w-4 h-4 text-white" />
        </button>
      </div>
    );
  }

  const {
    patient_id,
    modality,
    tumor_detected,
    predicted_class,
    confidence_score,
    class_probabilities,
    predictive_entropy,
    epistemic_uncertainty,
    fuzzy_uncertainty_score,
    fuzzy_uncertainty_level,
    fuzzy_reasoning_explanation,
    fired_fuzzy_rules,
    needs_expert_review,
    abstain_recommended,
    extracted_features,
    decision_summary,
    report_pdf_url,
    model_status
  } = currentAnalysis;

  // Chart data for probabilities
  const chartData = Object.entries(class_probabilities || {}).map(([name, prob]) => ({
    name,
    probability: Math.round(prob * 1000) / 10
  }));

  const intensityF = extracted_features?.intensity_features || { mean_intensity: 0, std_intensity: 0 };
  const shapeF = extracted_features?.shape_features || { area_pixels: 0, circularity: 0, boundary_irregularity: 0 };
  const textureF = extracted_features?.texture_features || { glcm_contrast: 0, glcm_homogeneity: 0 };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header with Title and PDF Download */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b-2 border-slate-300 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-black px-3 py-1 rounded-full bg-slate-200 text-black border-2 border-slate-400">
              Analysis #{currentAnalysis.analysis_id}
            </span>
            <span className="text-xs font-black px-3 py-1 rounded-full bg-blue-100 text-black border-2 border-blue-400">
              {modality} Modality
            </span>
            <span className="text-xs font-black text-black">Patient: {patient_id}</span>
          </div>
          <h2 className="text-2xl font-black text-black tracking-tight mt-1.5">
            Clinical Decision Support &amp; Uncertainty Assessment
          </h2>
        </div>

        {report_pdf_url && (
          <a
            href={report_pdf_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 bg-blue-700 hover:bg-blue-800 text-white font-black text-xs px-5 py-3 rounded-xl border border-blue-900 shadow-md transition-all cursor-pointer"
          >
            <Download className="w-4 h-4 text-white" />
            <span className="text-white">Download Official PDF Report</span>
          </a>
        )}
      </div>

      {/* Safety / Abstention Banner */}
      {abstain_recommended ? (
        <div className="p-5 rounded-2xl bg-amber-100 border-2 border-amber-400 flex items-start gap-3">
          <AlertTriangle className="w-6 h-6 text-amber-800 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-black text-black">High Uncertainty: Model Abstention Enforced</h4>
            <p className="text-xs text-black font-bold mt-1 leading-relaxed">
              Confidence ({Math.round(confidence_score * 100)}%) falls below safety thresholds, or predictive entropy is high.
              The AI system formally refrains from making a diagnostic classification. Mandatory radiological evaluation is required.
            </p>
          </div>
        </div>
      ) : needs_expert_review ? (
        <div className="p-5 rounded-2xl bg-red-100 border-2 border-red-400 flex items-start gap-3">
          <ShieldAlert className="w-6 h-6 text-red-800 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-black text-black">Expert Radiologist Review Strongly Advised</h4>
            <p className="text-xs text-black font-bold mt-1 leading-relaxed">
              Fuzzy rule evaluation detected boundary ambiguity or elevated epistemic uncertainty. Secondary confirmation recommended.
            </p>
          </div>
        </div>
      ) : (
        <div className="p-4 rounded-2xl bg-emerald-100 border-2 border-emerald-400 flex items-center gap-3">
          <ShieldCheck className="w-5 h-5 text-emerald-800 flex-shrink-0" />
          <p className="text-xs text-black font-black">
            Low diagnostic uncertainty detected across both Monte Carlo Dropout variance and Mamdani fuzzy rules.
          </p>
        </div>
      )}

      {/* Top 3 Metric Highlight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Diagnostic Finding */}
        <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <span className="text-xs font-black text-black uppercase tracking-wider">Classification Finding</span>
          <div className="my-4">
            <span className={`text-3xl font-black ${tumor_detected ? 'text-red-700' : 'text-emerald-700'}`}>
              {predicted_class}
            </span>
            <p className="text-xs font-bold text-black mt-1">
              {tumor_detected ? 'Space-occupying neoplasm detected' : 'Normal parenchymal structure'}
            </p>
          </div>
          <div className="pt-3 border-t-2 border-slate-200 flex items-center justify-between text-xs text-black">
            <span className="font-bold">Model Engine:</span>
            <span className="font-black text-black">{model_status?.is_trained ? 'PyTorch Deep CNN' : 'Prior Baseline'}</span>
          </div>
        </div>

        {/* Card 2: Confidence Score */}
        <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <span className="text-xs font-black text-black uppercase tracking-wider">Softmax Confidence Score</span>
          <div className="my-4">
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-black text-black">
                {(confidence_score * 100).toFixed(1)}%
              </span>
              <span className="text-xs font-bold text-black">softmax certainty</span>
            </div>
            {/* Confidence Bar */}
            <div className="w-full h-3.5 rounded-full bg-slate-200 mt-3 overflow-hidden border border-slate-300">
              <div
                className="h-full bg-blue-700 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, confidence_score * 100)}%` }}
              ></div>
            </div>
          </div>
          <div className="pt-3 border-t-2 border-slate-200 flex items-center justify-between text-xs text-black">
            <span className="font-bold">Predictive Entropy:</span>
            <span className="font-black text-black">{predictive_entropy.toFixed(3)} (bits)</span>
          </div>
        </div>

        {/* Card 3: Fuzzy Uncertainty Level */}
        <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <span className="text-xs font-black text-black uppercase tracking-wider">Fuzzy Uncertainty Assessment</span>
          <div className="my-4">
            <span className={`inline-block px-3.5 py-1.5 rounded-full text-sm font-black border-2 ${
              fuzzy_uncertainty_level === 'Low Uncertainty'
                ? 'bg-emerald-100 text-black border-emerald-400'
                : fuzzy_uncertainty_level === 'Moderate Uncertainty'
                ? 'bg-amber-100 text-black border-amber-400'
                : 'bg-red-100 text-black border-red-400'
            }`}>
              {fuzzy_uncertainty_level}
            </span>
            <p className="text-xs font-bold text-black mt-2">
              Defuzzified Score: <span className="font-black text-black">{fuzzy_uncertainty_score.toFixed(3)}</span> (Universe [0.0, 1.0])
            </p>
          </div>
          <div className="pt-3 border-t-2 border-slate-200 flex items-center justify-between text-xs text-black">
            <span className="font-bold">Epistemic Variance:</span>
            <span className="font-black text-black">{epistemic_uncertainty.toFixed(4)}</span>
          </div>
        </div>
      </div>

      {/* Class Probabilities Chart & Visual Previews */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Class Probabilities Bar Chart */}
        <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm">
          <h3 className="text-base font-black text-black mb-1">Class Probability Distribution</h3>
          <p className="text-xs font-bold text-black mb-4">Monte Carlo stochastic pass average probability across candidate categories</p>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30 }}>
                <XAxis type="number" domain={[0, 100]} unit="%" fontSize={11} stroke="#000000" tick={{ fill: '#000000', fontWeight: 'bold' }} />
                <YAxis dataKey="name" type="category" width={110} fontSize={11} stroke="#000000" tick={{ fill: '#000000', fontWeight: 'bold' }} />
                <Tooltip formatter={(value: any) => [`${value}%`, 'Probability']} contentStyle={{ backgroundColor: '#ffffff', borderColor: '#000000', fontWeight: 'bold' }} />
                <Bar dataKey="probability" radius={[0, 6, 6, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.name === predicted_class ? '#1d4ed8' : '#64748b'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Scan & Segmentation Overlay Preview */}
        <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-base font-black text-black">Lesion Segmentation Overlay</h3>
            <button
              onClick={() => onNavigate('preprocessing')}
              className="text-xs text-blue-900 font-black hover:underline cursor-pointer"
            >
              View 7-way Pipeline →
            </button>
          </div>
          <div className="grid grid-cols-2 gap-4 my-auto">
            <div className="bg-black rounded-xl p-1 flex flex-col items-center border border-slate-400">
              <img
                src={currentAnalysis.original_url}
                alt="Original"
                className="w-full h-36 object-contain rounded-lg"
              />
              <span className="text-[11px] font-bold text-white mt-1">Normalized Scan</span>
            </div>
            <div className="bg-black rounded-xl p-1 flex flex-col items-center border border-slate-400">
              <img
                src={currentAnalysis.tumor_overlay_url}
                alt="Overlay"
                className="w-full h-36 object-contain rounded-lg"
              />
              <span className="text-[11px] font-bold text-white mt-1">U-Net Tumor Overlay</span>
            </div>
          </div>
          <div className="text-xs font-bold text-black pt-3 border-t-2 border-slate-200 flex items-center justify-between">
            <span>Tumor Area: <b className="text-black">{shapeF.area_pixels} px</b></span>
            <span>Circularity: <b className="text-black">{shapeF.circularity.toFixed(2)}</b></span>
            <span>Boundary: <b className="text-black">{shapeF.boundary_irregularity > 0.5 ? 'Infiltrative' : 'Circumscribed'}</b></span>
          </div>
        </div>
      </div>

      {/* Fuzzy Reasoning Deep-Dive */}
      <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-black text-black">Mamdani Fuzzy Logic Inference &amp; Rule Trace</h3>
            <p className="text-xs font-bold text-black">Transparent mathematical explanation of uncertainty calculation</p>
          </div>
          <span className="text-xs px-3.5 py-1.5 rounded-lg bg-teal-100 text-black font-black border-2 border-teal-400">
            Fuzzy Score: {fuzzy_uncertainty_score.toFixed(3)}
          </span>
        </div>

        {/* Fired Rules List */}
        <div className="space-y-2.5">
          {fired_fuzzy_rules && fired_fuzzy_rules.length > 0 ? (
            fired_fuzzy_rules.map((rule, idx) => (
              <div key={idx} className="p-4 rounded-xl border-2 border-slate-300 bg-slate-50 text-xs space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-black text-sm text-black">
                    Rule [{rule.rule_id}] {rule.category_name ? `(${rule.category_name})` : ''}
                  </span>
                  {rule.firing_strength && (
                    <span className="font-black text-black bg-blue-100 px-2.5 py-0.5 rounded border border-blue-400">
                      Activation Strength: {rule.firing_strength.toFixed(2)}
                    </span>
                  )}
                </div>
                <p className="text-black font-bold leading-relaxed">
                  {rule.explanation || rule.inference}
                </p>
                {rule.conditions && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {rule.conditions.map((c, i) => (
                      <span key={i} className="text-[10px] bg-white border-2 border-slate-300 px-2.5 py-0.5 rounded text-black font-mono font-black">
                        {c}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))
          ) : (
            <p className="text-xs font-bold text-black">Standard baseline uncertainty rules applied.</p>
          )}
        </div>

        <div className="p-4 rounded-xl bg-slate-900 text-white text-xs font-mono whitespace-pre-line leading-relaxed border-2 border-slate-700">
          {fuzzy_reasoning_explanation}
        </div>
      </div>

      {/* Radiomic Quantitative Feature Profile */}
      <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-4">
        <div>
          <h3 className="text-base font-black text-black">Quantitative Radiomic Feature Profile</h3>
          <p className="text-xs font-bold text-black">Objective numerical descriptors extracted from the segmented region</p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
          <div className="p-3.5 rounded-xl bg-slate-100 border-2 border-slate-300">
            <span className="text-black block font-bold">Mean Intensity</span>
            <span className="text-black font-black text-base">{intensityF.mean_intensity.toFixed(1)}</span>
            <span className="text-[10px] text-black font-bold block mt-0.5">Attenuation index</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-100 border-2 border-slate-300">
            <span className="text-black block font-bold">Std Deviation</span>
            <span className="text-black font-black text-base">{intensityF.std_intensity.toFixed(1)}</span>
            <span className="text-[10px] text-black font-bold block mt-0.5">Heterogeneity</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-100 border-2 border-slate-300">
            <span className="text-black block font-bold">GLCM Contrast</span>
            <span className="text-black font-black text-base">{textureF.glcm_contrast.toFixed(2)}</span>
            <span className="text-[10px] text-black font-bold block mt-0.5">Local gray-level variations</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-100 border-2 border-slate-300">
            <span className="text-black block font-bold">GLCM Homogeneity</span>
            <span className="text-black font-black text-base">{textureF.glcm_homogeneity.toFixed(2)}</span>
            <span className="text-[10px] text-black font-bold block mt-0.5">Texture smoothness</span>
          </div>
        </div>
      </div>
    </div>
  );
};
