import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Search,
  Plus,
  Edit2,
  Check,
  Shield,
  Layers,
  Sparkles,
  HelpCircle
} from 'lucide-react';
import { getKnowledgeBase, updateKnowledgeBase } from '../services/api';

export const KnowledgeBasePage: React.FC = () => {
  const [kbData, setKbData] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('glioma');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    const fetchKb = async () => {
      try {
        const data = await getKnowledgeBase();
        setKbData(data);
      } catch (err) {
        console.error('Failed to load KB:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchKb();
  }, []);

  const categories = kbData?.tumor_categories || [];

  const filteredCategories = categories.filter((c: any) =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const activeCat = categories.find((c: any) => c.id === selectedCategory) || categories[0];

  const handleSaveRule = async (catId: string, ruleIdx: number, newInference: string) => {
    if (!kbData) return;
    setSaving(true);
    const updated = JSON.parse(JSON.stringify(kbData));
    const targetCat = updated.tumor_categories.find((c: any) => c.id === catId);
    if (targetCat && targetCat.rules[ruleIdx]) {
      targetCat.rules[ruleIdx].inference = newInference;
    }
    try {
      await updateKnowledgeBase(updated);
      setKbData(updated);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err) {
      console.error('Failed to save rule:', err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b-2 border-slate-300 pb-4">
        <div>
          <h2 className="text-2xl font-black text-black tracking-tight">
            Medical Tumor Knowledge Base &amp; Rule Engine
          </h2>
          <p className="text-sm font-bold text-black mt-1">
            Structured radiomic profiles, imaging characteristics, and decision-support reasoning rules.
          </p>
        </div>

        {/* Search Bar */}
        <div className="relative w-full sm:w-72">
          <Search className="w-4 h-4 text-black absolute left-3.5 top-3.5" />
          <input
            type="text"
            placeholder="Search categories & rules..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 text-xs font-black text-black rounded-xl border-2 border-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600 bg-white placeholder:text-slate-600"
          />
        </div>
      </div>

      {saveSuccess && (
        <div className="p-3.5 rounded-xl bg-emerald-100 border-2 border-emerald-400 text-black text-xs font-black flex items-center gap-2">
          <Check className="w-5 h-5 text-emerald-800" />
          <span>Knowledge base rules successfully updated in system storage.</span>
        </div>
      )}

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Category List Sidebar */}
        <div className="space-y-3">
          <h3 className="text-xs font-black text-black uppercase tracking-wider px-1">
            SUPPORTED TUMOR CATEGORIES
          </h3>
          <div className="space-y-2.5">
            {filteredCategories.map((cat: any) => {
              const isSelected = cat.id === activeCat?.id;
              return (
                <button
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  className={`w-full text-left p-4 rounded-xl transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-blue-800 text-white shadow-lg ring-2 ring-blue-500 border-2 border-blue-900'
                      : 'bg-white text-black border-2 border-slate-300 hover:border-blue-600 shadow-sm'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`font-black text-sm ${isSelected ? 'text-white' : 'text-black'}`}>
                      {cat.name}
                    </span>
                    <span
                      className={`text-[10px] font-black px-2 py-0.5 rounded border ${
                        isSelected
                          ? 'bg-slate-900 text-teal-300 border-teal-400'
                          : 'bg-slate-200 text-black border-slate-400'
                      }`}
                    >
                      {cat.modality}
                    </span>
                  </div>
                  <p
                    className={`text-xs mt-1.5 line-clamp-2 font-bold ${
                      isSelected ? 'text-slate-100' : 'text-slate-800'
                    }`}
                  >
                    {cat.description}
                  </p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Selected Category Details */}
        {activeCat && (
          <div className="lg:col-span-2 space-y-6">
            {/* Category Overview Card */}
            <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b-2 border-slate-200 pb-3">
                <div>
                  <h3 className="text-xl font-black text-black">{activeCat.name}</h3>
                  <p className="text-xs font-black text-blue-900 mt-0.5">Modality: {activeCat.modality}</p>
                </div>
                <span className="px-3.5 py-1.5 rounded-full text-xs font-black bg-teal-100 text-black border-2 border-teal-400">
                  {activeCat.rules?.length || 0} Decision Rules
                </span>
              </div>
              <p className="text-xs font-bold text-black leading-relaxed">{activeCat.description}</p>
            </div>

            {/* Imaging Characteristics */}
            <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-4">
              <h4 className="text-sm font-black text-black flex items-center gap-2">
                <Layers className="w-4 h-4 text-blue-800" />
                <span>Pathological Imaging Characteristics</span>
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {Object.entries(activeCat.imaging_characteristics || {}).map(([key, value]) => (
                  <div key={key} className="p-4 rounded-xl bg-slate-100 border-2 border-slate-300">
                    <span className="text-[11px] font-black text-slate-800 uppercase tracking-wider block">
                      {key.replace(/_/g, ' ')}
                    </span>
                    <span className="font-extrabold text-black mt-1.5 block leading-relaxed text-xs">
                      {String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Radiomic Feature Profile Table */}
            {activeCat.radiomic_feature_profiles && (
              <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-4">
                <h4 className="text-sm font-black text-black flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-teal-800" />
                  <span>Expected Radiomic Feature Profiles</span>
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-200 border-b-2 border-slate-400 text-black">
                      <tr>
                        <th className="py-2.5 px-3 font-black">Feature Metric</th>
                        <th className="py-2.5 px-3 font-black">Min Bounds</th>
                        <th className="py-2.5 px-3 font-black">Typical Profile</th>
                        <th className="py-2.5 px-3 font-black">Max Bounds</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y border-slate-300">
                      {Object.entries(activeCat.radiomic_feature_profiles).map(
                        ([featName, bounds]: [string, any]) => (
                          <tr key={featName} className="hover:bg-slate-100">
                            <td className="py-2.5 px-3 font-black text-black">
                              {featName.replace(/_/g, ' ')}
                            </td>
                            <td className="py-2.5 px-3 font-extrabold text-black">{bounds.min}</td>
                            <td className="py-2.5 px-3 font-black text-blue-900 text-sm">
                              {bounds.typical}
                            </td>
                            <td className="py-2.5 px-3 font-extrabold text-black">{bounds.max}</td>
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Decision Support Rules Editor */}
            <div className="bg-white p-6 rounded-2xl border-2 border-slate-300 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-black text-black flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-blue-800" />
                  <span>Rule-Based Reasoning Rules ({activeCat.name})</span>
                </h4>
                <span className="text-[11px] font-black text-blue-900">Editable for Decision Support Tuning</span>
              </div>

              <div className="space-y-3">
                {activeCat.rules?.map((rule: any, idx: number) => (
                  <div key={idx} className="p-4 rounded-xl border-2 border-slate-300 bg-slate-50 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-black text-xs text-black bg-blue-100 px-2.5 py-0.5 rounded border border-blue-400">
                        Rule ID: {rule.rule_id}
                      </span>
                      <span className="text-[11px] font-black text-black">
                        Weight: {rule.confidence_weight}
                      </span>
                    </div>

                    <div className="text-xs">
                      <span className="text-black block font-black mb-0.5">Condition:</span>
                      <code className="text-black bg-white px-2.5 py-1 rounded border-2 border-slate-300 block font-mono text-[11px] font-black">
                        {rule.condition}
                      </code>
                    </div>

                    <div className="text-xs pt-1">
                      <span className="text-black block font-black mb-0.5">Clinical Inference:</span>
                      <input
                        type="text"
                        defaultValue={rule.inference}
                        onBlur={(e) => handleSaveRule(activeCat.id, idx, e.target.value)}
                        className="w-full text-xs font-black text-black bg-white px-3 py-2 rounded-lg border-2 border-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
