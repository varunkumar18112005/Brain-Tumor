import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileImage,
  AlertCircle,
  Play,
  CheckCircle2,
  Sliders,
  RefreshCw,
  Info
} from 'lucide-react';
import { uploadImage, runFullAnalysis } from '../services/api';
import { UploadResponse, FullAnalysisResult } from '../types';

interface NewScanPageProps {
  onAnalysisComplete: (result: FullAnalysisResult) => void;
}

export const NewScanPage: React.FC<NewScanPageProps> = ({ onAnalysisComplete }) => {
  const [patientId, setPatientId] = useState('');
  const [modality, setModality] = useState<'MRI' | 'CT'>('MRI');
  const [file, setFile] = useState<File | null>(null);
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [selectedSlice, setSelectedSlice] = useState(0);

  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = async (selectedFile: File) => {
    setFile(selectedFile);
    setErrorMessage(null);
    setUploadData(null);
    setUploading(true);

    try {
      const pId = patientId.trim() || `PAT-${Math.floor(1000 + Math.random() * 9000)}`;
      const res = await uploadImage(selectedFile, modality, pId);
      setUploadData(res);
      setSelectedSlice(res.selected_slice || 0);
      if (!patientId) {
        setPatientId(res.patient_id);
      }
    } catch (err: any) {
      console.error('Upload failed:', err);
      setErrorMessage(err.response?.data?.detail || 'Failed to parse image file.');
      setFile(null);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const executePipeline = async () => {
    if (!uploadData) return;
    setAnalyzing(true);
    setErrorMessage(null);

    const steps = [
      'Applying Gaussian & Median Noise Removal...',
      'Executing CLAHE Contrast Enhancement & Standardization...',
      'Computing U-Net Lesion Segmentation & Contour Boundaries...',
      'Extracting Radiomic Intensity, GLCM Texture & Shape Descriptors...',
      'Evaluating Knowledge Base Profiles & Rule-Based Reasoning...',
      'Running Monte Carlo Dropout Deep CNN Classification...',
      'Evaluating Mamdani Fuzzy Uncertainty Engine...',
      'Synthesizing Clinical Decision Support & Compiling PDF Report...'
    ];

    let stepIdx = 0;
    const interval = setInterval(() => {
      if (stepIdx < steps.length - 1) {
        stepIdx++;
        setAnalysisStep(steps[stepIdx]);
      }
    }, 450);

    setAnalysisStep(steps[0]);

    try {
      const res = await runFullAnalysis(
        uploadData.file_id,
        uploadData.patient_id,
        uploadData.modality,
        selectedSlice
      );
      clearInterval(interval);
      onAnalysisComplete(res);
    } catch (err: any) {
      clearInterval(interval);
      console.error('Analysis pipeline failed:', err);
      setErrorMessage(err.response?.data?.detail || 'Analysis execution encountered an error.');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      <div className="border-b-2 border-slate-300 pb-4">
        <h2 className="text-2xl font-black text-black tracking-tight">New Scan Acquisition &amp; Analysis</h2>
        <p className="text-sm font-bold text-black mt-1">
          Upload medical scans in DICOM (.dcm), NIfTI (.nii, .nii.gz), PNG, or JPG format for automated tumor detection
          and uncertainty assessment.
        </p>
      </div>

      {errorMessage && (
        <div className="p-4 rounded-xl bg-red-100 border-2 border-red-400 text-black text-sm font-black flex items-center gap-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-700" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Patient & Modality Configuration */}
      <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-xs font-black text-black uppercase tracking-wider mb-2">
            Patient Reference ID (Anonymized)
          </label>
          <input
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            placeholder="e.g. PAT-BRATS-0492"
            className="w-full px-4 py-2.5 text-sm font-black text-black rounded-xl border-2 border-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600 bg-white placeholder:text-slate-600"
          />
          <p className="text-xs font-bold text-black mt-1.5">
            Never input real identifiable patient names; reference IDs only.
          </p>
        </div>

        <div>
          <label className="block text-xs font-black text-black uppercase tracking-wider mb-2">
            Imaging Modality
          </label>
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setModality('MRI')}
              className={`py-2.5 px-4 rounded-xl text-sm font-black border-2 flex items-center justify-center gap-2 transition-all cursor-pointer ${
                modality === 'MRI'
                  ? 'bg-blue-800 text-white border-blue-900 shadow-md ring-2 ring-blue-500'
                  : 'bg-white text-black border-slate-400 hover:bg-slate-100'
              }`}
            >
              <span>Brain MRI</span>
            </button>
            <button
              type="button"
              onClick={() => setModality('CT')}
              className={`py-2.5 px-4 rounded-xl text-sm font-black border-2 flex items-center justify-center gap-2 transition-all cursor-pointer ${
                modality === 'CT'
                  ? 'bg-blue-800 text-white border-blue-900 shadow-md ring-2 ring-blue-500'
                  : 'bg-white text-black border-slate-400 hover:bg-slate-100'
              }`}
            >
              <span>Head CT</span>
            </button>
          </div>
        </div>
      </div>

      {/* Upload Drop Zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all flex flex-col items-center justify-center ${
          uploadData
            ? 'border-emerald-600 bg-emerald-50'
            : 'border-slate-400 hover:border-blue-700 bg-white hover:bg-slate-50'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.dcm,.nii,.nii.gz"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
        />

        {uploading ? (
          <div className="flex flex-col items-center py-6">
            <RefreshCw className="w-10 h-10 text-blue-800 animate-spin mb-3" />
            <p className="text-sm font-black text-black">Converting and normalizing medical image...</p>
          </div>
        ) : uploadData ? (
          <div className="flex flex-col items-center">
            <CheckCircle2 className="w-12 h-12 text-emerald-700 mb-2" />
            <span className="text-sm font-black text-black">{uploadData.original_filename}</span>
            <p className="text-xs font-bold text-black mt-1">
              {uploadData.dimensions[0]} x {uploadData.dimensions[1]} px | {uploadData.filesize_kb} KB | Format: {uploadData.format}
            </p>
            <span className="mt-3 text-xs font-black text-blue-900 underline">
              Click to replace scan
            </span>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <div className="w-14 h-14 rounded-2xl bg-blue-100 text-blue-900 flex items-center justify-center mb-4 border-2 border-blue-300">
              <UploadCloud className="w-8 h-8 text-blue-900" />
            </div>
            <p className="text-base font-black text-black">
              Drag and drop your scan here, or <span className="text-blue-800 underline">browse files</span>
            </p>
            <p className="text-xs font-bold text-black mt-1.5 max-w-md">
              Supports MRI / CT formats: DICOM (.dcm), NIfTI (.nii, .nii.gz), and high-resolution PNG / JPEG.
            </p>
          </div>
        )}
      </div>

      {/* Upload Preview & Slice Selector */}
      {uploadData && (
        <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-6">
          <div className="flex items-center justify-between border-b-2 border-slate-200 pb-4">
            <div className="flex items-center gap-3">
              <FileImage className="w-5 h-5 text-blue-900" />
              <div>
                <h3 className="text-sm font-black text-black">Normalized Scan Preview</h3>
                <p className="text-xs font-bold text-black">Slice ready for automated segmentation and classification</p>
              </div>
            </div>
            <span className="px-3 py-1 text-xs font-black rounded-md bg-blue-100 text-black border-2 border-blue-400">
              {uploadData.modality}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            <div className="md:col-span-1 flex justify-center">
              <div className="w-48 h-48 rounded-xl overflow-hidden border-2 border-slate-400 bg-black flex items-center justify-center shadow-inner">
                <img
                  src={uploadData.relative_url}
                  alt="Scan Preview"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>

            <div className="md:col-span-2 space-y-4">
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-100 border-2 border-slate-300">
                  <span className="text-black block font-bold">Resolution</span>
                  <span className="text-black font-black text-sm">{uploadData.dimensions[0]} x {uploadData.dimensions[1]}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-100 border-2 border-slate-300">
                  <span className="text-black block font-bold">File Size</span>
                  <span className="text-black font-black text-sm">{uploadData.filesize_kb} KB</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-100 border-2 border-slate-300">
                  <span className="text-black block font-bold">Format Standard</span>
                  <span className="text-black font-black text-sm">{uploadData.format} Grayscale</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-100 border-2 border-slate-300">
                  <span className="text-black block font-bold">Assigned Patient ID</span>
                  <span className="text-black font-black text-sm">{uploadData.patient_id}</span>
                </div>
              </div>

              {uploadData.total_slices > 1 && (
                <div className="pt-2">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="font-black text-black">3D Volume Slice Selection:</span>
                    <span className="font-black text-blue-900">
                      Slice {selectedSlice} of {uploadData.total_slices - 1}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max={uploadData.total_slices - 1}
                    value={selectedSlice}
                    onChange={(e) => setSelectedSlice(parseInt(e.target.value))}
                    className="w-full accent-blue-700"
                  />
                </div>
              )}

              {/* Start Analysis Button */}
              <div className="pt-4">
                <button
                  type="button"
                  disabled={analyzing}
                  onClick={executePipeline}
                  className="w-full py-4 px-6 rounded-xl font-black text-white bg-blue-700 hover:bg-blue-800 border-2 border-blue-900 shadow-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50 text-sm cursor-pointer"
                >
                  {analyzing ? (
                    <>
                      <RefreshCw className="w-5 h-5 animate-spin text-white" />
                      <span className="text-white font-black">{analysisStep}</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 fill-current text-white" />
                      <span className="text-white font-black">Execute Complete AI Pipeline</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
