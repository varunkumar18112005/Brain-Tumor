import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend
} from 'recharts';
import {
  Cpu,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  BarChart2,
  Sliders,
  ShieldCheck
} from 'lucide-react';
import { getModelMetrics, trainModel } from '../services/api';
import { ModelMetrics } from '../types';

export const ModelPerformancePage: React.FC = () => {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [epochs, setEpochs] = useState(5);
  const [trainStatus, setTrainStatus] = useState<string | null>(null);

  const fetchMetrics = async () => {
    try {
      const data = await getModelMetrics();
      setMetrics(data);
    } catch (err) {
      console.error('Failed to load metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const handleTrain = async () => {
    setTraining(true);
    setTrainStatus('Training PyTorch CNN & U-Net models on synthetic dataset...');
    try {
      await trainModel(epochs, 'MRI');
      setTrainStatus('Training successfully completed! Updated model weights saved.');
      await fetchMetrics();
      setTimeout(() => setTrainStatus(null), 5000);
    } catch (err: any) {
      console.error('Training failed:', err);
      setTrainStatus(err.response?.data?.detail || 'Training failed.');
    } finally {
      setTraining(false);
    }
  };

  if (loading) {
    return (
      <div className="p-12 text-center text-black font-bold">
        <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-blue-800" />
        <p className="text-sm font-black">Loading model performance metrics...</p>
      </div>
    );
  }

  const isModelTrained = metrics && metrics.status !== 'NOT_TRAINED';

  // Loss curves data
  const lossCurvesData = metrics?.training_history?.epochs?.map((ep, i) => ({
    epoch: `Ep ${ep}`,
    trainLoss: metrics.training_history.train_loss[i],
    valLoss: metrics.training_history.val_loss[i],
    valAcc: Math.round((metrics.training_history.val_acc[i] || 0) * 100)
  })) || [];

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b-2 border-slate-300 pb-4">
        <div>
          <h2 className="text-2xl font-black text-black tracking-tight">Model Performance &amp; Evaluation</h2>
          <p className="text-sm font-bold text-black mt-1">
            Empirical validation metrics calculated from actual model predictions and ground-truth validation splits.
          </p>
        </div>

        {/* Training Trigger Controls */}
        <div className="flex items-center gap-3 bg-white p-2.5 rounded-xl border-2 border-slate-300 shadow-sm">
          <div className="flex items-center gap-1.5 text-xs font-black text-black px-2">
            <span>Epochs:</span>
            <select
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              disabled={training}
              className="font-black text-black bg-slate-100 border-2 border-slate-400 rounded px-2.5 py-1"
            >
              <option value={3}>3 Epochs</option>
              <option value={5}>5 Epochs</option>
              <option value={10}>10 Epochs</option>
              <option value={20}>20 Epochs</option>
            </select>
          </div>

          <button
            onClick={handleTrain}
            disabled={training}
            className="flex items-center gap-2 bg-blue-700 hover:bg-blue-800 text-white font-black text-xs px-5 py-2.5 rounded-lg border border-blue-900 transition-all disabled:opacity-50 shadow-sm cursor-pointer"
          >
            {training ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-white" />
                <span className="text-white">Training Models...</span>
              </>
            ) : (
              <>
                <Cpu className="w-3.5 h-3.5 text-white" />
                <span className="text-white">Train Model Now</span>
              </>
            )}
          </button>
        </div>
      </div>

      {trainStatus && (
        <div className="p-4 rounded-xl bg-blue-100 border-2 border-blue-400 text-black text-xs font-black flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-blue-800" />
          <span>{trainStatus}</span>
        </div>
      )}

      {!isModelTrained ? (
        <div className="p-8 rounded-2xl bg-amber-100 border-2 border-amber-400 text-center space-y-3">
          <AlertCircle className="w-10 h-10 text-amber-800 mx-auto" />
          <h3 className="text-base font-black text-black">Model Weights Not Yet Trained</h3>
          <p className="text-xs font-bold text-black max-w-md mx-auto">
            The neural network checkpoint has not been generated. Click the "Train Model Now" button above to
            execute real backpropagation and generate genuine evaluation curves.
          </p>
        </div>
      ) : (
        <>
          {/* Key Classification Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="bg-white p-4 rounded-xl border-2 border-slate-300 shadow-sm">
              <span className="text-[11px] font-black text-black uppercase tracking-wider block">Accuracy</span>
              <span className="text-2xl font-black text-black mt-2 block">
                {(metrics.accuracy * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] font-bold text-black mt-1 block">Validation subset</span>
            </div>

            <div className="bg-white p-4 rounded-xl border-2 border-slate-300 shadow-sm">
              <span className="text-[11px] font-black text-black uppercase tracking-wider block">Sensitivity</span>
              <span className="text-2xl font-black text-emerald-700 mt-2 block">
                {(metrics.sensitivity * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] font-bold text-black mt-1 block">True Positive Rate</span>
            </div>

            <div className="bg-white p-4 rounded-xl border-2 border-slate-300 shadow-sm">
              <span className="text-[11px] font-black text-black uppercase tracking-wider block">Specificity</span>
              <span className="text-2xl font-black text-teal-800 mt-2 block">
                {(metrics.specificity * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] font-bold text-black mt-1 block">True Negative Rate</span>
            </div>

            <div className="bg-white p-4 rounded-xl border-2 border-slate-300 shadow-sm">
              <span className="text-[11px] font-black text-black uppercase tracking-wider block">Precision</span>
              <span className="text-2xl font-black text-blue-800 mt-2 block">
                {(metrics.precision * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] font-bold text-black mt-1 block">Positive Predictive</span>
            </div>

            <div className="bg-white p-4 rounded-xl border-2 border-slate-300 shadow-sm">
              <span className="text-[11px] font-black text-black uppercase tracking-wider block">Dice Coeff</span>
              <span className="text-2xl font-black text-purple-800 mt-2 block">
                {metrics.dice_coefficient ? metrics.dice_coefficient.toFixed(3) : 'N/A'}
              </span>
              <span className="text-[10px] font-bold text-black mt-1 block">U-Net Spatial Overlap</span>
            </div>

            <div className="bg-white p-4 rounded-xl border-2 border-slate-300 shadow-sm">
              <span className="text-[11px] font-black text-black uppercase tracking-wider block">IoU (Jaccard)</span>
              <span className="text-2xl font-black text-indigo-800 mt-2 block">
                {metrics.iou_score ? metrics.iou_score.toFixed(3) : 'N/A'}
              </span>
              <span className="text-[10px] font-bold text-black mt-1 block">Intersection over Union</span>
            </div>
          </div>

          {/* Training Loss & Validation Curves */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm">
              <h3 className="text-base font-black text-black mb-1">Training &amp; Validation Loss Curves</h3>
              <p className="text-xs font-bold text-black mb-4">Empirical cross-entropy loss across epochs</p>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lossCurvesData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                    <XAxis dataKey="epoch" stroke="#000000" fontSize={11} tick={{ fill: '#000000', fontWeight: 'bold' }} />
                    <YAxis stroke="#000000" fontSize={11} tick={{ fill: '#000000', fontWeight: 'bold' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#000000', fontWeight: 'bold' }} />
                    <Legend />
                    <Line type="monotone" dataKey="trainLoss" name="Training Loss" stroke="#0284c7" strokeWidth={3} dot={{ r: 4 }} />
                    <Line type="monotone" dataKey="valLoss" name="Validation Loss" stroke="#d97706" strokeWidth={3} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Calibration Reliability Diagram */}
            <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm">
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-base font-black text-black">Reliability Diagram (Calibration)</h3>
                <span className="text-xs font-black px-2.5 py-0.5 rounded bg-slate-200 text-black border-2 border-slate-400">
                  ECE: {metrics.ece.toFixed(4)}
                </span>
              </div>
              <p className="text-xs font-bold text-black mb-4">Expected Calibration Error: Binned Confidence vs Empirical Accuracy</p>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metrics.reliability_diagram || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
                    <XAxis dataKey="bin_range" stroke="#000000" fontSize={10} tick={{ fill: '#000000', fontWeight: 'bold' }} />
                    <YAxis stroke="#000000" fontSize={11} domain={[0, 1]} tick={{ fill: '#000000', fontWeight: 'bold' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#000000', fontWeight: 'bold' }} />
                    <Legend />
                    <Bar dataKey="confidence" name="Mean Confidence" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="accuracy" name="Actual Accuracy" fill="#0369a1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Confusion Matrix Visual Grid */}
          {metrics.confusion_matrix && (
            <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-4">
              <h3 className="text-base font-black text-black">Multi-Class Confusion Matrix (Validation Split)</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-center text-xs border-2 border-slate-400">
                  <thead>
                    <tr className="bg-slate-200 text-black border-b-2 border-slate-400">
                      <th className="p-2.5 border-2 border-slate-400 text-left font-black">Ground Truth \ Predicted</th>
                      {metrics.class_names.map((name) => (
                        <th key={name} className="p-2.5 border-2 border-slate-400 font-black text-black">{name}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.confusion_matrix.map((row, rIdx) => (
                      <tr key={rIdx}>
                        <td className="p-2.5 border-2 border-slate-400 text-left font-black bg-slate-100 text-black">
                          {metrics.class_names[rIdx]}
                        </td>
                        {row.map((val, cIdx) => (
                          <td
                            key={cIdx}
                            className={`p-2.5 border-2 border-slate-400 font-mono font-black text-sm ${
                              rIdx === cIdx ? 'bg-emerald-100 text-emerald-950 font-black' : val > 0 ? 'bg-red-100 text-red-950 font-black' : 'text-slate-600 font-bold'
                            }`}
                          >
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs font-black text-black text-right">Last Trained: {metrics.last_trained}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};
