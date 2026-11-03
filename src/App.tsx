
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useState } from 'react';
import { Sidebar, NavTab } from './components/Sidebar';
import { Header } from './components/Header';
import { OverviewView } from './views/OverviewView';
import SegmentationWorkspace from './views/SegmentationWorkspace';
import { SegmentationReliabilityView } from './views/SegmentationReliabilityView';
import { DatasetExplorerView } from './views/DatasetExplorerView';
import { ModelEvaluationView } from './views/ModelEvaluationView';
import { DocumentationView } from './views/DocumentationView';
import { SettingsView } from './views/SettingsView';
import { ExperimentHistoryView } from './views/ExperimentHistoryView';
import { BusiImage } from './types';
import { Menu, X } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000';

export default function App() {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [selectedBusiCase, setSelectedBusiCase] = useState<BusiImage | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  useEffect(() => {
    let isActive = true;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (!response.ok) {
          throw new Error(`Health check failed: ${response.status}`);
        }

        const data: { status?: string } = await response.json();

        if (isActive) {
          setIsBackendConnected(data.status === 'ok');
        }
      } catch {
        if (isActive) setIsBackendConnected(false);
      }
    };

    void checkBackend();
    const intervalId = window.setInterval(() => void checkBackend(), 10000);

    return () => {
      isActive = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const handleSelectForSegmentation = (image: BusiImage) => {
    setSelectedBusiCase(image);
    setCurrentTab('segmentation');
    setMobileMenuOpen(false);
  };

  const handleTabChange = (tab: NavTab) => {
    setCurrentTab(tab);
    setMobileMenuOpen(false);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 text-slate-900">
      <div className="hidden shrink-0 md:flex">
        <Sidebar
          currentTab={currentTab}
          onTabChange={handleTabChange}
          isBackendConnected={isBackendConnected}
        />
      </div>

      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 flex md:hidden">
          <button
            type="button"
            className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close navigation menu"
          />

          <div className="relative z-50 flex w-72 max-w-[80vw] flex-col bg-white">
            <div className="flex justify-end p-3">
              <button
                type="button"
                onClick={() => setMobileMenuOpen(false)}
                className="rounded-md p-1.5 text-slate-500 hover:text-slate-800"
                aria-label="Close menu"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <Sidebar
              currentTab={currentTab}
              onTabChange={handleTabChange}
              isBackendConnected={isBackendConnected}
            />
          </div>
        </div>
      )}

      <div className="flex h-full min-w-0 flex-1 flex-col overflow-hidden">
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4 md:hidden">
          <button
            type="button"
            onClick={() => setMobileMenuOpen(true)}
            className="rounded-md p-1.5 text-slate-600 hover:bg-slate-100"
            aria-label="Open navigation menu"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="font-mono text-xs font-bold text-slate-900">
            MedSegment AI
          </div>

          <span
            className={`rounded px-2 py-0.5 font-mono text-[10px] ${
              isBackendConnected
                ? 'bg-emerald-50 text-emerald-700'
                : 'bg-amber-50 text-amber-700'
            }`}
          >
            {isBackendConnected ? 'API CONNECTED' : 'DEMO'}
          </span>
        </div>

        <Header
          currentTab={currentTab}
          onTabChange={handleTabChange}
          datasetName="BUSI Breast Ultrasound"
          totalImages={780}
        />

        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
          {currentTab === 'overview' && (
            <OverviewView onNavigate={handleTabChange} />
          )}

          {currentTab === 'segmentation' && (
            <SegmentationWorkspace initialImage={selectedBusiCase} />
          )}

          {currentTab === 'reliability' && (
            <SegmentationReliabilityView />
          )}

          {currentTab === 'dataset' && (
            <DatasetExplorerView
              onSelectForSegmentation={handleSelectForSegmentation}
            />
          )}

          {currentTab === 'evaluation' && <ModelEvaluationView />}
          {currentTab === 'documentation' && <DocumentationView />}
          {currentTab === 'settings' && <SettingsView />}
          {currentTab === 'experiments' && <ExperimentHistoryView />}
        </main>
      </div>
    </div>
  );
}
