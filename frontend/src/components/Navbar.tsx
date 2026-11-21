import React from 'react';
import { ShieldCheck, Cpu, HeartPulse } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  isTrained: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, isTrained }) => {
  return (
    <header className="bg-white border-b border-slate-300 sticky top-0 z-30 px-6 py-3.5 flex items-center justify-between shadow-sm">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-900 to-teal-600 flex items-center justify-center text-white shadow-md">
          <HeartPulse className="w-6 h-6 animate-pulse" />
        </div>
        <div>
          <h1 className="text-lg font-black text-black tracking-tight leading-tight">
            NeuroMed AI <span className="text-xs font-black px-2 py-0.5 rounded-full bg-teal-100 text-black border border-teal-400 ml-1">v1.0 Pro</span>
          </h1>
          <p className="text-xs text-black font-bold">Intelligent Medical Tumor Detection &amp; Uncertainty Assessment</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Model Status Badge */}
        <div className="hidden sm:flex items-center gap-2 px-3.5 py-1.5 rounded-lg border border-slate-400 bg-slate-100 text-xs">
          <Cpu className="w-4 h-4 text-blue-900" />
          <span className="text-black font-black">Deep Engine:</span>
          {isTrained ? (
            <span className="flex items-center gap-1.5 font-black text-black bg-emerald-200 px-2.5 py-0.5 rounded border border-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-700"></span> Active (PyTorch Weights)
            </span>
          ) : (
            <span className="flex items-center gap-1.5 font-black text-black bg-amber-200 px-2.5 py-0.5 rounded border border-amber-400">
              <span className="w-2 h-2 rounded-full bg-amber-700"></span> Baseline Mode (Ready to Train)
            </span>
          )}
        </div>

        {/* Safety Disclaimer Indicator */}
        <div className="hidden md:flex items-center gap-1.5 text-xs text-black font-black bg-blue-100 border border-blue-400 px-3 py-1.5 rounded-lg">
          <ShieldCheck className="w-4 h-4 text-blue-800" />
          <span>Decision Support System</span>
        </div>
      </div>
    </header>
  );
};
