
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  Bell,
  ChevronRight,
  CircleHelp,
  Database,
  FileImage,
  FlaskConical,
  LayoutDashboard,
  Settings2,
  Sparkles,
  X,
  ArrowUpRight,
  ScanLine,
} from 'lucide-react';

import { NavTab } from './Sidebar';

interface HeaderProps {
  currentTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  datasetName?: string;
  totalImages?: number;
}

const PAGE_INFO: Record<
  NavTab,
  { title: string; subtitle: string; icon: React.ElementType }
> = {
  overview: {
    title: 'Overview',
    subtitle: 'Your medical imaging research at a glance.',
    icon: LayoutDashboard,
  },
  segmentation: {
    title: 'Segmentation Studio',
    subtitle: 'Analyze ultrasound images and generate lesion masks.',
    icon: FileImage,
  },
  reliability: {
    title: 'Segmentation Reliability Studio',
    subtitle: 'Inspect candidate masks, disagreement, and refinement.',
    icon: ScanLine,
  },
  dataset: {
    title: 'Dataset Explorer',
    subtitle: 'Browse and inspect your breast ultrasound dataset.',
    icon: Database,
  },
  evaluation: {
    title: 'Model Evaluation',
    subtitle: 'Review segmentation performance and evaluation metrics.',
    icon: Activity,
  },
  experiments: {
    title: 'Experiment History',
    subtitle: 'Track model configurations and research runs.',
    icon: FlaskConical,
  },
  documentation: {
    title: 'Documentation',
    subtitle: 'Explore the pipeline, methodology, and research notes.',
    icon: CircleHelp,
  },
  settings: {
    title: 'Settings',
    subtitle: 'Manage your workspace and backend connection.',
    icon: Settings2,
  },
};

const RESEARCH_UPDATES = [
  {
    id: 1,
    title: 'Dataset overview',
    description:
      'BUSI contains benign, malignant, and normal breast ultrasound cases.',
    tag: 'DATASET',
    icon: Database,
  },
  {
    id: 2,
    title: 'Automatic prompt generation',
    description:
      'AutoPromptCNNV2 predicts a bounding box for the MedSAM segmentation pipeline.',
    tag: 'PIPELINE',
    icon: Sparkles,
  },
  {
    id: 3,
    title: 'Evaluation metrics',
    description:
      'Review Dice score, IoU, precision, and recall on the evaluation page.',
    tag: 'EVALUATION',
    icon: Activity,
  },
  {
    id: 4,
    title: 'Segmentation reliability',
    description:
      'Compare candidate masks and inspect areas where predictions disagree.',
    tag: 'RELIABILITY',
    icon: ScanLine,
  },
];

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onTabChange,
  datasetName = 'BUSI Breast Ultrasound',
  totalImages = 780,
}) => {
  const [showNotifications, setShowNotifications] = useState(false);
  const notificationRef = useRef<HTMLDivElement>(null);

  const page = PAGE_INFO[currentTab];
  const PageIcon = page.icon;

  useEffect(() => {
    if (!showNotifications) return;

    const handleOutsideClick = (event: MouseEvent) => {
      const target = event.target as Node;

      if (
        notificationRef.current &&
        !notificationRef.current.contains(target)
      ) {
        setShowNotifications(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setShowNotifications(false);
      }
    };

    document.addEventListener('mousedown', handleOutsideClick);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [showNotifications]);

  return (
    <header className="relative z-30 shrink-0 border-b border-slate-200 bg-white">
      {/* Top identity strip */}
      <div className="flex h-[54px] items-center justify-between bg-[#102f35] px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-400 text-[#102f35] shadow-sm">
            <Activity className="h-[18px] w-[18px]" strokeWidth={2.5} />
          </div>

          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold tracking-tight text-white">
                MedSegment
              </span>
              <span className="rounded bg-teal-400/15 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-teal-300">
                AI
              </span>
            </div>
            <p className="hidden text-[10px] tracking-wide text-teal-100/60 sm:block">
              MEDICAL IMAGING RESEARCH
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 sm:gap-3">
          <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/[0.06] px-3 py-1.5 md:flex">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-30" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
            </span>
            <span className="text-[10px] font-medium tracking-wide text-teal-50/90">
              RESEARCH WORKSPACE
            </span>
          </div>

          {/* Research notifications */}
          <div className="relative" ref={notificationRef}>
            <button
              type="button"
              onClick={() =>
                setShowNotifications((previous) => !previous)
              }
              aria-label="Open research updates"
              aria-expanded={showNotifications}
              className={`relative flex h-9 w-9 items-center justify-center rounded-xl transition-colors ${
                showNotifications
                  ? 'bg-white/15 text-white'
                  : 'text-teal-100/75 hover:bg-white/10 hover:text-white'
              }`}
            >
              <Bell className="h-[17px] w-[17px]" />
              <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-amber-300 ring-2 ring-[#102f35]" />
            </button>

            {showNotifications && (
              <div className="absolute right-0 top-12 z-50 w-[min(360px,calc(100vw-2rem))] overflow-hidden rounded-2xl border border-slate-200 bg-white text-left shadow-2xl shadow-slate-950/15">
                <div className="flex items-start justify-between bg-[#f0f8f7] px-4 py-4">
                  <div>
                    <p className="text-sm font-bold text-slate-900">
                      Research updates
                    </p>
                    <p className="mt-1 text-[11px] text-slate-500">
                      Information about your current workspace
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() => setShowNotifications(false)}
                    aria-label="Close updates"
                    className="rounded-lg p-1.5 text-slate-400 hover:bg-white hover:text-slate-700"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>

                <div className="divide-y divide-slate-100">
                  {RESEARCH_UPDATES.map((update) => {
                    const UpdateIcon = update.icon;

                    return (
                      <div
                        key={update.id}
                        className="flex gap-3 px-4 py-3.5"
                      >
                        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-teal-50 text-teal-700">
                          <UpdateIcon className="h-4 w-4" />
                        </div>

                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <p className="text-xs font-semibold text-slate-800">
                              {update.title}
                            </p>
                            <span className="text-[9px] font-bold tracking-wider text-teal-700">
                              {update.tag}
                            </span>
                          </div>

                          <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                            {update.description}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="border-t border-slate-100 px-4 py-3">
                  <p className="text-[10px] leading-relaxed text-slate-400">
                    For research use only. Outputs are not intended for
                    clinical diagnosis.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main page heading */}
      <div className="px-4 py-4 sm:px-6 sm:py-5">
        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
          <div className="flex min-w-0 items-start gap-3.5">
            <div className="mt-0.5 flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-teal-100 bg-gradient-to-br from-teal-50 to-emerald-50 text-teal-800">
              <PageIcon className="h-5 w-5" strokeWidth={1.8} />
            </div>

            <div className="min-w-0">
              <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-medium text-slate-400">
                <span>Workspace</span>
                <ChevronRight className="h-3 w-3" />
                <span className="truncate text-teal-700">
                  {page.title}
                </span>
              </div>

              <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
                {page.title}
              </h1>

              <p className="mt-1 text-xs leading-relaxed text-slate-500 sm:text-[13px]">
                {page.subtitle}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 sm:justify-end">
            {/* Dataset badge */}
            <div className="flex h-10 items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-white text-teal-700 shadow-sm">
                <Database className="h-3.5 w-3.5" />
              </div>

              <div className="min-w-0">
                <p className="max-w-[130px] truncate text-[10px] font-semibold text-slate-700 sm:max-w-[155px]">
                  {datasetName}
                </p>
                <p className="text-[10px] text-slate-500">
                  {totalImages.toLocaleString()} images
                </p>
              </div>
            </div>

            {/* Primary action */}
            {currentTab !== 'segmentation' && (
              <button
                type="button"
                onClick={() => onTabChange('segmentation')}
                className="group flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-teal-700 to-emerald-700 px-3.5 text-xs font-semibold text-white shadow-md shadow-teal-900/10 transition-all hover:-translate-y-0.5 hover:from-teal-800 hover:to-emerald-800 hover:shadow-lg active:translate-y-0 sm:px-4"
              >
                <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-white/15">
                  <Sparkles className="h-3.5 w-3.5 text-teal-100" />
                </span>
                <span>New segmentation</span>
                <ArrowUpRight className="hidden h-3.5 w-3.5 text-teal-100 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 sm:block" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Bottom accent */}
      <div className="h-[2px] bg-gradient-to-r from-teal-700 via-emerald-400 to-transparent" />
    </header>
  );
};

export default Header;
