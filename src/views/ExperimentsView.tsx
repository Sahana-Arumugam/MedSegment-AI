
import React, { useMemo, useState } from 'react';
import {
  FlaskConical,
  Plus,
  Trash2,
  X,
  Search,
  Clock3,
  CheckCircle2,
  CircleDashed,
  Play,
  Layers3,
  ChevronDown,
  SlidersHorizontal,
  Info,
  CalendarDays,
  FileText,
  Target,
} from 'lucide-react';
import { EXPERIMENTS_HISTORY } from '../data/experimentsData';
import { Experiment, ModelArchitecture, PromptMode } from '../types';

type ExperimentStatus = Experiment['status'];
type DatasetSplit = 'busi_80_20' | 'busi_70_30' | 'busi_5_fold';
type AdaptationMethod =
  | 'zero_shot'
  | 'lora'
  | 'full_finetune'
  | 'active_learning';

const statusStyles: Record<ExperimentStatus, string> = {
  Completed: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20',
  Running: 'bg-blue-50 text-blue-700 ring-blue-600/20',
  Planned: 'bg-slate-100 text-slate-600 ring-slate-500/20',
  Failed: 'bg-rose-50 text-rose-700 ring-rose-600/20',
};

const formatLabel = (value: string) =>
  value
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

const getStatusIcon = (status: ExperimentStatus) => {
  switch (status) {
    case 'Completed':
      return <CheckCircle2 size={14} />;
    case 'Running':
      return <Play size={13} />;
    case 'Failed':
      return <X size={13} />;
    default:
      return <CircleDashed size={14} />;
  }
};

export const ExperimentsView: React.FC = () => {
  const [experiments, setExperiments] =
    useState<Experiment[]>(EXPERIMENTS_HISTORY);

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [statusFilter, setStatusFilter] = useState('All');
  const [architectureFilter, setArchitectureFilter] = useState('All');
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest');

  const [name, setName] = useState('');
  const [modelArchitecture, setModelArchitecture] =
    useState<ModelArchitecture>('medsam_lora');
  const [promptMode, setPromptMode] = useState<PromptMode>('auto_bbox');
  const [datasetSplit, setDatasetSplit] =
    useState<DatasetSplit>('busi_80_20');
  const [adaptationMethod, setAdaptationMethod] =
    useState<AdaptationMethod>('lora');
  const [labeledImagesCount, setLabeledImagesCount] = useState(200);
  const [description, setDescription] = useState('');
  const [formError, setFormError] = useState('');

  const completedCount = experiments.filter(
    (experiment) => experiment.status === 'Completed',
  ).length;

  const runningCount = experiments.filter(
    (experiment) => experiment.status === 'Running',
  ).length;

  const plannedCount = experiments.filter(
    (experiment) => experiment.status === 'Planned',
  ).length;

  const filteredExperiments = useMemo(() => {
    return experiments
      .filter(
        (experiment) =>
          statusFilter === 'All' || experiment.status === statusFilter,
      )
      .filter(
        (experiment) =>
          architectureFilter === 'All' ||
          experiment.modelArchitecture === architectureFilter,
      )
      .sort((a, b) => {
        const comparison = a.date.localeCompare(b.date);
        return sortOrder === 'newest' ? -comparison : comparison;
      });
  }, [experiments, statusFilter, architectureFilter, sortOrder]);

  const resetForm = () => {
    setName('');
    setDescription('');
    setModelArchitecture('medsam_lora');
    setPromptMode('auto_bbox');
    setDatasetSplit('busi_80_20');
    setAdaptationMethod('lora');
    setLabeledImagesCount(200);
    setFormError('');
  };

  const handleCreateExperiment = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!name.trim()) {
      setFormError('Enter a name for this experiment.');
      return;
    }

    if (
      !Number.isInteger(labeledImagesCount) ||
      labeledImagesCount < 10 ||
      labeledImagesCount > 546
    ) {
      setFormError('Enter a labeled image count between 10 and 546.');
      return;
    }

    const newExperiment: Experiment = {
      id: `exp-${Date.now()}`,
      name: name.trim(),
      modelArchitecture,
      promptMode,
      datasetSplit,
      adaptationMethod,
      labeledImagesCount,
      // These values are placeholders until an actual training run
      // computes and stores the metrics.
      trainableParamsMillion: 0,
      totalParamsMillion: 0,
      status: 'Planned',
      diceScore: 0,
      iouScore: 0,
      hd95: 0,
      trainingTimeMin: 0,
      inferenceLatencyMs: 0,
      date: new Date().toISOString().slice(0, 10),
      description:
        description.trim() || 'No experiment notes have been added.',
    };

    setExperiments((current) => [newExperiment, ...current]);
    setShowCreateModal(false);
    resetForm();
  };

  const handleDelete = (id: string) => {
    setExperiments((current) =>
      current.filter((experiment) => experiment.id !== id),
    );
  };

  const inputClass =
    'mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-800 outline-none transition focus:border-teal-600 focus:ring-2 focus:ring-teal-600/10';

  const labelClass =
    'text-xs font-semibold text-slate-700';

  return (
    <div className="mx-auto max-w-7xl space-y-7 pb-12">
      {/* Header */}
      <header className="flex flex-col justify-between gap-5 rounded-2xl border border-slate-200 bg-white p-6 sm:flex-row sm:items-center sm:p-8">
        <div>
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-teal-100 bg-teal-50 px-3 py-1.5 text-[10px] font-bold uppercase tracking-[0.14em] text-teal-800">
            <FlaskConical size={13} />
            Research workspace
          </div>

          <h1 className="text-2xl font-bold tracking-tight text-slate-950 sm:text-3xl">
            Experiments
          </h1>

          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-500">
            Configure research trials, track experiment status, and review
            recorded evaluation results.
          </p>
        </div>

        <button
          type="button"
          onClick={() => {
            resetForm();
            setShowCreateModal(true);
          }}
          className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-teal-700 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-teal-800 focus:outline-none focus:ring-2 focus:ring-teal-600 focus:ring-offset-2"
        >
          <Plus size={17} />
          New experiment
        </button>
      </header>

      {/* Important research notice */}
      <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4">
        <Info size={18} className="mt-0.5 shrink-0 text-amber-700" />
        <div>
          <p className="text-sm font-semibold text-amber-950">
            Registering an experiment does not start model training
          </p>
          <p className="mt-1 text-xs leading-relaxed text-amber-900">
            New experiments are stored in the current frontend session.
            Evaluation metrics will remain unavailable until a training or
            evaluation backend returns actual results.
          </p>
        </div>
      </div>

      {/* Summary cards */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">
              Total experiments
            </span>
            <div className="rounded-lg bg-slate-100 p-2 text-slate-700">
              <Layers3 size={17} />
            </div>
          </div>
          <p className="mt-4 text-3xl font-bold tracking-tight text-slate-950">
            {experiments.length}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Registered research trials
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">
              Completed
            </span>
            <div className="rounded-lg bg-emerald-50 p-2 text-emerald-700">
              <CheckCircle2 size={17} />
            </div>
          </div>
          <p className="mt-4 text-3xl font-bold tracking-tight text-slate-950">
            {completedCount}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Marked as completed
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">
              Running
            </span>
            <div className="rounded-lg bg-blue-50 p-2 text-blue-700">
              <Play size={17} />
            </div>
          </div>
          <p className="mt-4 text-3xl font-bold tracking-tight text-slate-950">
            {runningCount}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Marked as running
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">
              Planned
            </span>
            <div className="rounded-lg bg-violet-50 p-2 text-violet-700">
              <Clock3 size={17} />
            </div>
          </div>
          <p className="mt-4 text-3xl font-bold tracking-tight text-slate-950">
            {plannedCount}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Awaiting execution
          </p>
        </div>
      </section>

      {/* Experiment list */}
      <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
        <div className="flex flex-col justify-between gap-4 border-b border-slate-200 p-5 sm:flex-row sm:items-center sm:p-6">
          <div>
            <h2 className="text-base font-bold text-slate-900">
              Experiment registry
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              {filteredExperiments.length} of {experiments.length} experiments
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2">
              <SlidersHorizontal size={14} className="text-slate-400" />
              <select
                aria-label="Filter by status"
                value={statusFilter}
                onChange={(event) => setStatusFilter(event.target.value)}
                className="max-w-32 bg-transparent text-xs font-medium text-slate-700 outline-none"
              >
                <option value="All">All statuses</option>
                <option value="Planned">Planned</option>
                <option value="Running">Running</option>
                <option value="Completed">Completed</option>
                <option value="Failed">Failed</option>
              </select>
            </div>

            <div className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2">
              <Layers3 size={14} className="text-slate-400" />
              <select
                aria-label="Filter by architecture"
                value={architectureFilter}
                onChange={(event) =>
                  setArchitectureFilter(event.target.value)
                }
                className="max-w-40 bg-transparent text-xs font-medium text-slate-700 outline-none"
              >
                <option value="All">All architectures</option>
                <option value="medsam_lora">MedSAM + LoRA</option>
                <option value="medsam_baseline">MedSAM baseline</option>
                <option value="medsam_autoprompt">MedSAM + Auto-Prompt</option>
                <option value="full_finetune">Full fine-tuning</option>
              </select>
            </div>

            <button
              type="button"
              onClick={() =>
                setSortOrder((current) =>
                  current === 'newest' ? 'oldest' : 'newest',
                )
              }
              className="rounded-lg border border-slate-200 px-3 py-2 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
            >
              {sortOrder === 'newest' ? 'Newest first' : 'Oldest first'}
            </button>
          </div>
        </div>

        {filteredExperiments.length === 0 ? (
          <div className="flex flex-col items-center px-6 py-16 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100 text-slate-500">
              <FlaskConical size={23} />
            </div>
            <h3 className="mt-4 text-sm font-bold text-slate-900">
              No experiments found
            </h3>
            <p className="mt-1 max-w-sm text-xs leading-relaxed text-slate-500">
              Try changing your filters or register a new experiment.
            </p>
            <button
              type="button"
              onClick={() => {
                setStatusFilter('All');
                setArchitectureFilter('All');
              }}
              className="mt-4 text-xs font-semibold text-teal-700 hover:text-teal-900"
            >
              Clear filters
            </button>
          </div>
        ) : (
          <>
            {/* Desktop table */}
            <div className="hidden overflow-x-auto lg:block">
              <table className="w-full border-collapse text-left">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/80 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    <th className="px-5 py-3.5">Experiment</th>
                    <th className="px-4 py-3.5">Architecture</th>
                    <th className="px-4 py-3.5">Prompt mode</th>
                    <th className="px-4 py-3.5">Labeled images</th>
                    <th className="px-4 py-3.5">Dice</th>
                    <th className="px-4 py-3.5">IoU</th>
                    <th className="px-4 py-3.5">Status</th>
                    <th className="px-4 py-3.5">Date</th>
                    <th className="px-4 py-3.5 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {filteredExperiments.map((experiment) => {
                    const hasMetrics =
                      experiment.status === 'Completed' ||
                      experiment.status === 'Running';

                    return (
                      <tr
                        key={experiment.id}
                        className="transition-colors hover:bg-slate-50/70"
                      >
                        <td className="max-w-64 px-5 py-4">
                          <p className="truncate text-sm font-semibold text-slate-900">
                            {experiment.name}
                          </p>
                          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-slate-500">
                            {experiment.description}
                          </p>
                        </td>
                        <td className="px-4 py-4 text-xs text-slate-600">
                          {formatLabel(experiment.modelArchitecture)}
                        </td>
                        <td className="px-4 py-4 text-xs text-slate-600">
                          {formatLabel(experiment.promptMode)}
                        </td>
                        <td className="px-4 py-4 text-xs font-medium tabular-nums text-slate-800">
                          {experiment.labeledImagesCount}
                        </td>
                        <td className="px-4 py-4 text-xs font-semibold tabular-nums text-slate-800">
                          {hasMetrics
                            ? experiment.diceScore.toFixed(3)
                            : '—'}
                        </td>
                        <td className="px-4 py-4 text-xs font-semibold tabular-nums text-slate-800">
                          {hasMetrics
                            ? experiment.iouScore.toFixed(3)
                            : '—'}
                        </td>
                        <td className="px-4 py-4">
                          <span
                            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ring-inset ${
                              statusStyles[experiment.status]
                            }`}
                          >
                            {getStatusIcon(experiment.status)}
                            {experiment.status}
                          </span>
                        </td>
                        <td className="whitespace-nowrap px-4 py-4 text-xs tabular-nums text-slate-500">
                          {experiment.date}
                        </td>
                        <td className="px-4 py-4 text-right">
                          <button
                            type="button"
                            onClick={() => handleDelete(experiment.id)}
                            title="Delete experiment"
                            aria-label={`Delete ${experiment.name}`}
                            className="rounded-lg p-2 text-slate-400 transition hover:bg-rose-50 hover:text-rose-600"
                          >
                            <Trash2 size={15} />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Responsive cards for smaller screens */}
            <div className="grid grid-cols-1 gap-3 p-4 sm:p-5 lg:hidden">
              {filteredExperiments.map((experiment) => {
                const hasMetrics =
                  experiment.status === 'Completed' ||
                  experiment.status === 'Running';

                return (
                  <article
                    key={experiment.id}
                    className="rounded-xl border border-slate-200 p-4"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <h3 className="break-words text-sm font-bold text-slate-900">
                          {experiment.name}
                        </h3>
                        <p className="mt-1 text-xs leading-relaxed text-slate-500">
                          {experiment.description}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleDelete(experiment.id)}
                        aria-label={`Delete ${experiment.name}`}
                        className="shrink-0 rounded-lg p-2 text-slate-400 hover:bg-rose-50 hover:text-rose-600"
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>

                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <span
                        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ring-inset ${
                          statusStyles[experiment.status]
                        }`}
                      >
                        {getStatusIcon(experiment.status)}
                        {experiment.status}
                      </span>
                      <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-medium text-slate-600">
                        {formatLabel(experiment.modelArchitecture)}
                      </span>
                    </div>

                    <div className="mt-4 grid grid-cols-2 gap-3 border-t border-slate-100 pt-3">
                      <div>
                        <p className="text-[10px] text-slate-400">
                          Labeled images
                        </p>
                        <p className="mt-1 text-sm font-semibold text-slate-800">
                          {experiment.labeledImagesCount}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-slate-400">
                          Created
                        </p>
                        <p className="mt-1 text-sm font-semibold text-slate-800">
                          {experiment.date}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-slate-400">
                          Dice score
                        </p>
                        <p className="mt-1 text-sm font-semibold text-slate-800">
                          {hasMetrics
                            ? experiment.diceScore.toFixed(3)
                            : 'Not available'}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-slate-400">
                          IoU score
                        </p>
                        <p className="mt-1 text-sm font-semibold text-slate-800">
                          {hasMetrics
                            ? experiment.iouScore.toFixed(3)
                            : 'Not available'}
                        </p>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </>
        )}
      </section>

      {/* Create experiment modal */}
      {showCreateModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-slate-950/50 p-4 backdrop-blur-sm"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setShowCreateModal(false);
            }
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="create-experiment-title"
            className="my-auto max-h-[92vh] w-full max-w-2xl overflow-y-auto rounded-2xl border border-slate-200 bg-white shadow-2xl"
          >
            <div className="sticky top-0 z-10 flex items-start justify-between border-b border-slate-200 bg-white p-5 sm:p-6">
              <div>
                <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-xl bg-teal-50 text-teal-700">
                  <FlaskConical size={18} />
                </div>
                <h2
                  id="create-experiment-title"
                  className="text-lg font-bold text-slate-950"
                >
                  Configure experiment
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  Define the setup for a planned research trial.
                </p>
              </div>

              <button
                type="button"
                onClick={() => setShowCreateModal(false)}
                aria-label="Close dialog"
                className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
              >
                <X size={19} />
              </button>
            </div>

            <form onSubmit={handleCreateExperiment} className="space-y-5 p-5 sm:p-6">
              <div>
                <label htmlFor="experiment-name" className={labelClass}>
                  Experiment name *
                </label>
                <input
                  id="experiment-name"
                  type="text"
                  required
                  maxLength={100}
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="e.g. Auto-prompt baseline on BUSI"
                  className={inputClass}
                />
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label htmlFor="experiment-architecture" className={labelClass}>
                    Model architecture
                  </label>
                  <select
                    id="experiment-architecture"
                    value={modelArchitecture}
                    onChange={(event) => {
                      const value = event.target.value as ModelArchitecture;
                      setModelArchitecture(value);

                      if (value === 'medsam_baseline') {
                        setAdaptationMethod('zero_shot');
                      } else if (value === 'full_finetune') {
                        setAdaptationMethod('full_finetune');
                      } else if (value === 'medsam_lora') {
                        setAdaptationMethod('lora');
                      }
                    }}
                    className={inputClass}
                  >
                    <option value="medsam_lora">MedSAM + LoRA</option>
                    <option value="medsam_baseline">MedSAM baseline</option>
                    <option value="medsam_autoprompt">
                      MedSAM + Auto-Prompt
                    </option>
                    <option value="full_finetune">Full fine-tuning</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="experiment-prompt" className={labelClass}>
                    Prompt mode
                  </label>
                  <select
                    id="experiment-prompt"
                    value={promptMode}
                    onChange={(event) =>
                      setPromptMode(event.target.value as PromptMode)
                    }
                    className={inputClass}
                  >
                    <option value="auto_bbox">Automatic bounding box</option>
                    <option value="reference_bbox">Ground-truth box</option>
                    <option value="point_prompts">Point prompts</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="experiment-adaptation" className={labelClass}>
                    Adaptation method
                  </label>
                  <select
                    id="experiment-adaptation"
                    value={adaptationMethod}
                    onChange={(event) =>
                      setAdaptationMethod(event.target.value as AdaptationMethod)
                    }
                    className={inputClass}
                  >
                    <option value="zero_shot">Zero-shot</option>
                    <option value="lora">LoRA</option>
                    <option value="full_finetune">Full fine-tuning</option>
                    <option value="active_learning">Active learning</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="experiment-split" className={labelClass}>
                    Evaluation split
                  </label>
                  <select
                    id="experiment-split"
                    value={datasetSplit}
                    onChange={(event) =>
                      setDatasetSplit(event.target.value as DatasetSplit)
                    }
                    className={inputClass}
                  >
                    <option value="busi_80_20">80/20 train/test</option>
                    <option value="busi_70_30">70/30 split configuration</option>
                    <option value="busi_5_fold">5-fold cross-validation</option>
                  </select>
                </div>
              </div>

              <div>
                <label htmlFor="labeled-images" className={labelClass}>
                  Number of labeled images
                </label>
                <input
                  id="labeled-images"
                  type="number"
                  min={10}
                  max={546}
                  step={1}
                  required
                  value={labeledImagesCount}
                  onChange={(event) =>
                    setLabeledImagesCount(
                      event.target.value === ''
                        ? 0
                        : Number(event.target.value),
                    )
                  }
                  className={inputClass}
                />
                <p className="mt-1.5 text-[11px] text-slate-400">
                  Enter a value between 10 and 546. Confirm the actual
                  training split before running an experiment.
                </p>
              </div>

              <div>
                <label htmlFor="experiment-description" className={labelClass}>
                  Research hypothesis and notes
                </label>
                <textarea
                  id="experiment-description"
                  rows={3}
                  maxLength={1000}
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="Describe the hypothesis, baseline, or configuration to investigate..."
                  className={`${inputClass} resize-y`}
                />
              </div>

              <div className="flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
                <Info size={15} className="mt-0.5 shrink-0 text-slate-500" />
                <p className="text-xs leading-relaxed text-slate-600">
                  This form registers a planned experiment only. It does not
                  launch training, allocate a GPU, or generate evaluation
                  metrics.
                </p>
              </div>

              {formError && (
                <p role="alert" className="text-xs font-medium text-rose-600">
                  {formError}
                </p>
              )}

              <div className="flex flex-col-reverse gap-2 border-t border-slate-100 pt-4 sm:flex-row sm:justify-end">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="rounded-lg border border-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-teal-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-teal-800 focus:outline-none focus:ring-2 focus:ring-teal-600 focus:ring-offset-2"
                >
                  <Plus size={16} />
                  Register experiment
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
