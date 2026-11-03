
import React from 'react';
import {
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  Activity,
  ScanLine,
  Database,
  Image as ImageIcon,
  Layers3,
  Microscope,
  ShieldCheck,
  FileText,
  ChevronRight,
  CircleDot,
  Crosshair,
  Workflow,
  MoveUpRight,
} from 'lucide-react';

import { BUSI_METRICS_SUMMARY } from '../data/busiSampleData';
import { NavTab } from '../components/Sidebar';

interface OverviewViewProps {
  onNavigate: (tab: NavTab) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  onNavigate,
}) => {
  const metrics = [
    {
      index: '01',
      label: 'DATASET VOLUME',
      value: BUSI_METRICS_SUMMARY.totalImages,
      suffix: 'images',
    },
    {
      index: '02',
      label: 'ANNOTATED',
      value: BUSI_METRICS_SUMMARY.annotatedCount,
      suffix: 'with masks',
    },
    {
      index: '03',
      label: 'TRAINING SET',
      value: BUSI_METRICS_SUMMARY.trainingImages,
      suffix: 'images',
    },
    {
      index: '04',
      label: 'VALIDATION / TEST',
      value: `${BUSI_METRICS_SUMMARY.valImages} / ${BUSI_METRICS_SUMMARY.testImages}`,
      suffix: 'images',
    },
  ];

  const pipeline = [
    {
      id: '01',
      title: 'Image input',
      detail: 'Breast ultrasound',
      icon: ImageIcon,
    },
    {
      id: '02',
      title: 'AutoPromptCNNV2',
      detail: 'Lesion localization',
      icon: Crosshair,
    },
    {
      id: '03',
      title: 'MedSAM ViT-B',
      detail: 'Mask generation',
      icon: Layers3,
    },
    {
      id: '04',
      title: 'Evaluation',
      detail: 'Dice · IoU · Recall',
      icon: Activity,
    },
  ];

  return (
    <div className="mx-auto w-full max-w-[1700px] space-y-8 pb-12 text-slate-900">

      {/* Top identity strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#103D37] text-[#B9F6D7]">
            <Microscope className="h-[18px] w-[18px]" />
          </div>
          <div>
            <p className="text-sm font-bold tracking-tight text-slate-950">
              MedSegment <span className="text-teal-700">AI</span>
            </p>
            <p className="mt-0.5 text-[10px] uppercase tracking-[0.16em] text-slate-400">
              Research environment / 01
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2">
          <span className="h-2 w-2 rounded-full bg-teal-600" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-600">
            BUSI · Ultrasound
          </span>
        </div>
      </div>

      {/* Editorial hero */}
      <section className="relative overflow-hidden rounded-[24px] bg-[#102A2B] text-white">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -right-24 -top-28 h-96 w-96 rounded-full border border-teal-200/[0.10]" />
          <div className="absolute -right-8 -top-12 h-64 w-64 rounded-full border border-teal-200/[0.12]" />
          <div className="absolute bottom-[-160px] right-[18%] h-72 w-72 rounded-full bg-teal-400/[0.12] blur-3xl" />
          <div
            className="absolute inset-0 opacity-[0.045]"
            style={{
              backgroundImage:
                'linear-gradient(rgba(255,255,255,.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.5) 1px, transparent 1px)',
              backgroundSize: '30px 30px',
            }}
          />
        </div>

        <div className="relative grid lg:grid-cols-[1.15fr_0.85fr]">
          <div className="p-7 sm:p-10 lg:p-12">
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-[#A8EACD]">
              <span className="h-1.5 w-1.5 rounded-full bg-[#A8EACD]" />
              Medical image intelligence
            </div>

            <h1 className="mt-8 max-w-2xl text-[42px] font-medium leading-[1.05] tracking-[-0.055em] sm:text-6xl xl:text-[68px]">
              Pixels into
              <br />
              <span className="font-serif italic text-[#A8EACD]">
                possibilities.
              </span>
            </h1>

            <p className="mt-6 max-w-lg text-sm leading-7 text-slate-300 sm:text-base">
              Prompt-efficient lesion segmentation for breast ultrasound.
              Investigate how automatic localization and foundation models
              can work together in a reproducible research pipeline.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              <button
                onClick={() => onNavigate('segmentation')}
                className="group inline-flex items-center gap-3 rounded-lg bg-[#B9F6D7] px-5 py-3.5 text-sm font-bold text-[#102A2B] transition hover:bg-white focus:outline-none focus:ring-4 focus:ring-teal-100/20"
              >
                <ScanLine className="h-4 w-4" />
                Open image workspace
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </button>

              <button
                onClick={() => onNavigate('documentation')}
                className="inline-flex items-center gap-2 rounded-lg border border-white/20 px-4 py-3.5 text-sm font-medium text-white transition hover:bg-white/10"
              >
                <FileText className="h-4 w-4" />
                Methodology
              </button>
            </div>

            <div className="mt-10 flex flex-wrap gap-x-6 gap-y-3 border-t border-white/15 pt-5 text-xs text-slate-300">
              <span className="inline-flex items-center gap-2">
                <CircleDot className="h-3.5 w-3.5 text-[#A8EACD]" />
                Automatic bounding-box prediction
              </span>
              <span className="inline-flex items-center gap-2">
                <CircleDot className="h-3.5 w-3.5 text-[#A8EACD]" />
                Pixel-level segmentation
              </span>
            </div>
          </div>

          {/* Abstract scientific visualization */}
          <div className="relative flex min-h-[300px] items-center justify-center overflow-hidden px-5 pb-8 pt-2 sm:px-10 lg:min-h-full lg:p-8">
            <div className="relative w-full max-w-[370px]">
              <div className="absolute -inset-8 rounded-full border border-teal-200/[0.08]" />
              <div className="absolute -inset-3 rounded-full border border-dashed border-teal-200/[0.15]" />

              <div className="relative overflow-hidden rounded-2xl border border-white/15 bg-[#0A2021]/90 p-5 shadow-2xl">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.18em] text-teal-200">
                      Pipeline overview
                    </p>
                    <p className="mt-2 text-lg font-semibold">
                      From image to mask
                    </p>
                  </div>
                  <Workflow className="h-5 w-5 text-teal-200" />
                </div>

                <div className="mt-6 space-y-0">
                  {pipeline.map((stage, index) => {
                    const Icon = stage.icon;

                    return (
                      <React.Fragment key={stage.id}>
                        <div className="group flex items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.035] p-3 transition hover:border-teal-200/30 hover:bg-white/[0.07]">
                          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#B9F6D7]/10 text-[#B9F6D7]">
                            <Icon className="h-5 w-5" />
                          </div>

                          <div className="min-w-0 flex-1">
                            <p className="text-sm font-semibold text-white">
                              {stage.title}
                            </p>
                            <p className="mt-1 text-xs text-slate-400">
                              {stage.detail}
                            </p>
                          </div>

                          <span className="font-mono text-[10px] text-slate-500">
                            {stage.id}
                          </span>
                        </div>

                        {index < pipeline.length - 1 && (
                          <div className="ml-8 flex h-5 items-center">
                            <div className="h-full border-l border-dashed border-teal-200/30" />
                            <ArrowDownRight className="ml-1 h-3 w-3 text-teal-200/60" />
                          </div>
                        )}
                      </React.Fragment>
                    );
                  })}
                </div>

                <div className="mt-5 flex items-center justify-between border-t border-white/10 pt-4">
                  <span className="text-[10px] uppercase tracking-wider text-slate-500">
                    Research prototype
                  </span>
                  <span className="text-[10px] font-semibold text-[#A8EACD]">
                    NOT FOR DIAGNOSIS
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Data ledger */}
      <section>
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-teal-800">
              Research inventory
            </p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight">
              Dataset at a glance
            </h2>
          </div>
          <button
            onClick={() => onNavigate('dataset')}
            className="group inline-flex items-center gap-1.5 text-sm font-semibold text-teal-800 hover:text-teal-950"
          >
            Browse all images
            <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </button>
        </div>

        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
          <div className="grid grid-cols-2 divide-x divide-y divide-slate-200 sm:grid-cols-4 sm:divide-y-0">
            {metrics.map((metric) => (
              <div
                key={metric.index}
                className="relative p-5 transition-colors hover:bg-[#F1F8F5] sm:p-6"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] tracking-widest text-slate-400">
                    {metric.index}
                  </span>
                  <ArrowUpRight className="h-3.5 w-3.5 text-slate-300" />
                </div>
                <p className="mt-5 text-[10px] font-semibold tracking-[0.14em] text-slate-500">
                  {metric.label}
                </p>
                <p className="mt-2 break-words text-3xl font-semibold tracking-[-0.05em] text-[#102A2B] sm:text-4xl">
                  {metric.value}
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  {metric.suffix}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technical flow */}
      <section>
        <div className="mb-5">
          <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-teal-800">
            System architecture
          </p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">
            The path to segmentation
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            Each stage contributes to the automatic inference workflow. The
            current implementation uses the trained prompt generator and
            pretrained MedSAM foundation model.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {pipeline.map((stage, index) => {
            const Icon = stage.icon;

            return (
              <div
                key={stage.id}
                className="group relative border-l-2 border-slate-200 bg-white py-5 pl-5 pr-4 transition hover:border-teal-600 hover:bg-[#F7FBF9]"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-teal-800">
                    STAGE / {stage.id}
                  </span>
                  <Icon className="h-5 w-5 text-slate-400 transition group-hover:text-teal-700" />
                </div>
                <h3 className="mt-5 text-base font-semibold">{stage.title}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-500">
                  {stage.detail}
                </p>
                {index < pipeline.length - 1 && (
                  <ArrowRight className="mt-5 h-4 w-4 text-slate-300 group-hover:text-teal-700" />
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* Research action */}
      <section className="relative overflow-hidden rounded-2xl border border-[#D5E8DF] bg-[#EDF7F1] p-6 sm:p-8">
        <div className="absolute -right-10 -top-16 h-48 w-48 rounded-full border border-teal-900/[0.06]" />
        <div className="relative flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-4">
            <div className="rounded-xl bg-[#103D37] p-3 text-[#B9F6D7]">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-teal-800">
                Model validation
              </p>
              <h2 className="mt-1.5 text-xl font-semibold tracking-tight">
                Explore the measured results.
              </h2>
              <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600">
                Review the completed test evaluation, compare localization
                performance, and inspect segmentation metrics.
              </p>
            </div>
          </div>

          <button
            onClick={() => onNavigate('evaluation')}
            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-[#103D37] px-5 py-3 text-sm font-semibold text-white transition hover:bg-[#18564D]"
          >
            Open evaluation
            <MoveUpRight className="h-4 w-4" />
          </button>
        </div>
      </section>

      <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-5 text-xs text-slate-400">
        <span>MedSegment AI / Research prototype</span>
        <span>BUSI · Automated medical image segmentation</span>
      </footer>
    </div>
  );
};
