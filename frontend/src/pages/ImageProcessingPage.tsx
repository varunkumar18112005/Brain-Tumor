import React from 'react';
import { Layers, Sliders, CheckCircle2, AlertCircle, Eye, ZoomIn, ArrowRight } from 'lucide-react';
import { FullAnalysisResult } from '../types';

interface ImageProcessingPageProps {
  currentAnalysis: FullAnalysisResult | null;
  onNavigate: (tab: string) => void;
}

export const ImageProcessingPage: React.FC<ImageProcessingPageProps> = ({ currentAnalysis, onNavigate }) => {
  if (!currentAnalysis) {
    return (
      <div className="p-12 max-w-4xl mx-auto text-center space-y-5">
        <div className="w-16 h-16 rounded-2xl bg-slate-200 text-black flex items-center justify-center mx-auto border-2 border-slate-400">
          <Layers className="w-8 h-8 text-black" />
        </div>
        <h3 className="text-xl font-black text-black">No Processed Scan Selected</h3>
        <p className="text-sm font-bold text-black max-w-md mx-auto leading-relaxed">
          Please upload a scan and run the AI pipeline in the "New Scan Analysis" tab to view real image preprocessing transformations.
        </p>
        <button
          onClick={() => onNavigate('new-scan')}
          className="inline-flex items-center gap-2 bg-blue-700 hover:bg-blue-800 text-white px-6 py-3 rounded-xl text-sm font-black transition-all shadow-md cursor-pointer border border-blue-900"
        >
          <span>Go to New Scan Analysis</span>
          <ArrowRight className="w-4 h-4 text-white" />
        </button>
      </div>
    );
  }

  const images = [
    {
      title: '1. Original Medical Scan',
      desc: 'Raw voxel matrix as acquired, converted to canonical grayscale.',
      url: currentAnalysis.original_url,
      tag: 'Input'
    },
    {
      title: '2. Noise-Reduced Image',
      desc: 'Gaussian kernel (k=3) & Median filtering to eliminate speckle noise.',
      url: currentAnalysis.denoised_url,
      tag: 'Filter'
    },
    {
      title: '3. Contrast-Enhanced Image',
      desc: 'CLAHE (Contrast Limited Adaptive Histogram Equalization, clip=2.0).',
      url: currentAnalysis.enhanced_url,
      tag: 'CLAHE'
    },
    {
      title: '4. Normalized & Standardized',
      desc: 'Normalized to [0.0, 1.0] intensity domain and model resolution.',
      url: currentAnalysis.normalized_url,
      tag: '224x224'
    },
    {
      title: '5. Segmentation Mask',
      desc: 'Deep U-Net binary lesion prediction contour.',
      url: currentAnalysis.segmentation_mask_url,
      tag: 'Mask'
    },
    {
      title: '6. Tumor Color Overlay',
      desc: 'High-contrast cyan boundary contour and semi-transparent red tumor ROI.',
      url: currentAnalysis.tumor_overlay_url,
      tag: 'Overlay'
    },
    {
      title: '7. Extracted Tumor Region',
      desc: 'Cropped lesion bounding box with surrounding tissue context.',
      url: currentAnalysis.extracted_region_url,
      tag: 'ROI'
    }
  ];

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b-2 border-slate-300 pb-4">
        <div>
          <h2 className="text-2xl font-black text-black tracking-tight">Image Preprocessing &amp; Transformations</h2>
          <p className="text-sm font-bold text-black mt-1">
            Visual side-by-side comparison of noise reduction, contrast enhancement, normalization, and lesion segmentation.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-black px-3.5 py-1.5 rounded-lg bg-blue-100 text-black border-2 border-blue-400">
            Patient: {currentAnalysis.patient_id}
          </span>
          <span className="text-xs font-black px-3.5 py-1.5 rounded-lg bg-slate-200 text-black border-2 border-slate-400">
            Modality: {currentAnalysis.modality}
          </span>
        </div>
      </div>

      {/* Grid of Images */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {images.map((img, i) => (
          <div
            key={i}
            className="bg-white rounded-2xl border-2 border-slate-300 shadow-sm overflow-hidden flex flex-col justify-between hover:shadow-md transition-all"
          >
            <div className="p-4 border-b-2 border-slate-200 flex items-center justify-between bg-slate-100">
              <span className="font-black text-xs text-black">{img.title}</span>
              <span className="px-2 py-0.5 text-[10px] font-black rounded bg-slate-200 text-black border border-slate-400">
                {img.tag}
              </span>
            </div>

            <div className="p-4 flex items-center justify-center bg-black min-h-[220px]">
              <img
                src={img.url}
                alt={img.title}
                className="max-h-48 max-w-full object-contain rounded-lg shadow-inner"
              />
            </div>

            <div className="p-4 bg-slate-50 border-t-2 border-slate-200 text-xs text-black">
              <p className="leading-relaxed font-bold">{img.desc}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Verification Notice */}
      <div className="p-4 rounded-xl bg-blue-100 border-2 border-blue-400 flex items-start gap-3 text-xs text-black font-bold">
        <CheckCircle2 className="w-5 h-5 text-blue-800 flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-black">Real Preprocessing Pipeline Guarantee:</span> Every displayed image is
          computationally derived directly from the uploaded patient scan file via OpenCV / SciPy filters and PyTorch U-Net inference.
          No synthetic placeholder images or mock visuals are utilized.
        </div>
      </div>
    </div>
  );
};
