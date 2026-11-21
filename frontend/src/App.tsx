import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { DashboardPage } from './pages/DashboardPage';
import { NewScanPage } from './pages/NewScanPage';
import { ImageProcessingPage } from './pages/ImageProcessingPage';
import { ClassificationPage } from './pages/ClassificationPage';
import { KnowledgeBasePage } from './pages/KnowledgeBasePage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { HistoryPage } from './pages/HistoryPage';
import { SettingsPage } from './pages/SettingsPage';
import { getModelStatus, getAnalysisById } from './services/api';
import { FullAnalysisResult } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [currentAnalysis, setCurrentAnalysis] = useState<FullAnalysisResult | null>(null);
  const [isTrained, setIsTrained] = useState<boolean>(true);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await getModelStatus();
        const trained = res.models.some((m) => m.is_trained);
        setIsTrained(trained);
      } catch (err) {
        console.error('Failed to query model status:', err);
      }
    };
    checkStatus();
  }, []);

  const handleAnalysisComplete = (res: FullAnalysisResult) => {
    setCurrentAnalysis(res);
    setActiveTab('results');
  };

  const handleSelectHistoricalScan = async (analysisOrId: FullAnalysisResult | number) => {
    if (typeof analysisOrId === 'number') {
      try {
        const full = await getAnalysisById(analysisOrId);
        setCurrentAnalysis(full);
        setActiveTab('results');
      } catch (err) {
        console.error('Failed to load analysis:', err);
      }
    } else {
      setCurrentAnalysis(analysisOrId);
      setActiveTab('results');
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 text-slate-800">
      {/* Sidebar Navigation */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar activeTab={activeTab} isTrained={isTrained} />

        <main className="flex-1 overflow-y-auto">
          {activeTab === 'dashboard' && (
            <DashboardPage
              onNavigate={setActiveTab}
              onSelectScan={handleSelectHistoricalScan}
            />
          )}

          {activeTab === 'new-scan' && (
            <NewScanPage onAnalysisComplete={handleAnalysisComplete} />
          )}

          {activeTab === 'preprocessing' && (
            <ImageProcessingPage
              currentAnalysis={currentAnalysis}
              onNavigate={setActiveTab}
            />
          )}

          {activeTab === 'results' && (
            <ClassificationPage
              currentAnalysis={currentAnalysis}
              onNavigate={setActiveTab}
            />
          )}

          {activeTab === 'knowledge-base' && <KnowledgeBasePage />}

          {activeTab === 'performance' && <ModelPerformancePage />}

          {activeTab === 'history' && (
            <HistoryPage onSelectScan={handleSelectHistoricalScan} />
          )}

          {activeTab === 'settings' && <SettingsPage />}
        </main>
      </div>
    </div>
  );
};

export default App;
