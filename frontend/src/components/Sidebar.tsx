import React from 'react';
import {
  LayoutDashboard,
  UploadCloud,
  Layers,
  BrainCircuit,
  BookOpen,
  LineChart,
  History,
  Settings as SettingsIcon
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'new-scan', label: 'New Scan Analysis', icon: UploadCloud },
    { id: 'preprocessing', label: 'Image Processing', icon: Layers },
    { id: 'results', label: 'Classification & Uncertainty', icon: BrainCircuit },
    { id: 'knowledge-base', label: 'Knowledge Base', icon: BookOpen },
    { id: 'performance', label: 'Model Performance', icon: LineChart },
    { id: 'history', label: 'Analysis History', icon: History },
    { id: 'settings', label: 'Settings', icon: SettingsIcon },
  ];

  return (
    <aside className="w-64 bg-slate-950 text-white flex flex-col justify-between border-r border-slate-800 select-none">
      <div className="p-4">
        <div className="text-[12px] font-black text-slate-300 uppercase tracking-wider px-3 mb-3">
          Clinical Navigation
        </div>
        <nav className="space-y-1.5">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-bold transition-all ${
                  isActive
                    ? 'bg-blue-700 text-white shadow-md ring-2 ring-blue-400 shadow-blue-900/50'
                    : 'text-slate-200 hover:text-white hover:bg-slate-800/90'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-300'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="p-4 border-t border-slate-800 bg-slate-950">
        <div className="bg-slate-900 rounded-xl p-3.5 border border-slate-700">
          <div className="flex items-center justify-between text-xs font-black text-white mb-1">
            <span>Research Mode</span>
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
          </div>
          <p className="text-[11px] text-slate-300 font-semibold leading-relaxed">
            Strictly for academic &amp; decision support research.
          </p>
        </div>
      </div>
    </aside>
  );
};
