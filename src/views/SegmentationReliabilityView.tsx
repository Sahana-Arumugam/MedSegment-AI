
import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Download,
  ImagePlus,
  LoaderCircle,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react';

const API = 'http://127.0.0.1:8000';

type ReliabilityResult = {
  filename: string;
  width: number;
  height: number;
  bbox_xyxy: number[];
  candidate_masks: string[];
  candidate_scores: number[];
  uncertainty_map: string;
  original_mask: string;
  refined_mask: string;
  original_foreground_pixels: number;
  refined_foreground_pixels: number;
  refinement: string;
  note: string;
};

type ViewMode =
  | 'original'
  | 'candidate-1'
  | 'candidate-2'
  | 'candidate-3'
  | 'uncertainty'
  | 'before'
  | 'after';

function pngUrl(base64: string): string {
  return `data:image/png;base64,${base64}`;
}

function downloadBase64(base64: string, filename: string) {
  const link = document.createElement('a');
  link.href = pngUrl(base64);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export function SegmentationReliabilityView() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState<ReliabilityResult | null>(null);
  const [mode, setMode] = useState<ViewMode>('original');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    return () => {
      if (previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const chooseFile = (nextFile?: File) => {
    if (!nextFile) return;

    if (!/\.(png|jpe?g|bmp|tiff?)$/i.test(nextFile.name)) {
      setError('Choose a PNG, JPG, JPEG, BMP, or TIFF image.');
      return;
    }

    if (nextFile.size > 15 * 1024 * 1024) {
      setError('The image must be 15 MB or smaller.');
      return;
    }

    setError('');
    setResult(null);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setMode('original');

    if (fileInput.current) fileInput.current.value = '';
  };

  const runAnalysis = async () => {
    if (!file) {
      setError('Upload an ultrasound image first.');
      return;
    }

    setBusy(true);
    setError('');
    setResult(null);

    try {
      const body = new FormData();
      body.append('file', file);

      const response = await fetch(`${API}/api/reliability`, {
        method: 'POST',
        body,
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          data?.detail || `Reliability analysis failed (${response.status}).`
        );
      }

      setResult(data as ReliabilityResult);
      setMode('uncertainty');
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not complete reliability analysis.'
      );
    } finally {
      setBusy(false);
    }
  };

  const getDisplayImage = () => {
    if (!result) return previewUrl;

    switch (mode) {
      case 'candidate-1':
        return pngUrl(result.candidate_masks[0]);
      case 'candidate-2':
        return pngUrl(result.candidate_masks[1]);
      case 'candidate-3':
        return pngUrl(result.candidate_masks[2]);
      case 'uncertainty':
        return pngUrl(result.uncertainty_map);
      case 'before':
        return pngUrl(result.original_mask);
      case 'after':
        return pngUrl(result.refined_mask);
      default:
        return previewUrl;
    }
  };

  const modeOptions: { id: ViewMode; label: string }[] = [
    { id: 'original', label: 'Original image' },
    { id: 'candidate-1', label: 'Candidate 1' },
    { id: 'candidate-2', label: 'Candidate 2' },
    { id: 'candidate-3', label: 'Candidate 3' },
    { id: 'uncertainty', label: 'Disagreement map' },
    { id: 'before', label: 'Before refinement' },
    { id: 'after', label: 'After refinement' },
  ];

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-teal-700">
            <Activity className="h-4 w-4" />
            Research tools
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            Segmentation Reliability Studio
          </h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Compare MedSAM candidate masks, inspect where their predictions
            disagree, and review a post-processed mask beside the original.
          </p>
        </div>

        <span className="inline-flex w-fit items-center gap-2 rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-800">
          <ShieldAlert className="h-4 w-4" />
          Experimental analysis
        </span>
      </header>

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <h2 className="font-semibold text-slate-900">Input image</h2>
            <p className="mt-1 text-sm text-slate-500">
              Upload a BUSI ultrasound image or another supported image.
            </p>
            {file && (
              <p className="mt-2 truncate text-xs text-slate-600">
                Selected: {file.name}
              </p>
            )}
          </div>

          <div className="flex flex-wrap gap-2">
            <input
              ref={fileInput}
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff"
              className="hidden"
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />

            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              disabled={busy}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-300 px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              <ImagePlus className="h-4 w-4" />
              Choose image
            </button>

            <button
              type="button"
              onClick={runAnalysis}
              disabled={!file || busy}
              className="inline-flex items-center gap-2 rounded-xl bg-teal-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? (
                <LoaderCircle className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              {busy ? 'Analyzing…' : 'Analyze reliability'}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </section>

      <section className="grid gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(280px,0.8fr)]">
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="border-b border-slate-200 p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="font-semibold text-slate-900">
                  Visual inspection
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  {modeOptions.find((item) => item.id === mode)?.label}
                </p>
              </div>

              <select
                value={mode}
                onChange={(event) => setMode(event.target.value as ViewMode)}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700"
              >
                {modeOptions.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex min-h-[320px] items-center justify-center bg-slate-950 p-4 sm:min-h-[460px]">
            {previewUrl ? (
              <img
                src={getDisplayImage()}
                alt={modeOptions.find((item) => item.id === mode)?.label}
                className="max-h-[650px] max-w-full object-contain"
              />
            ) : (
              <div className="text-center text-slate-400">
                <ImagePlus className="mx-auto mb-3 h-10 w-10" />
                <p className="text-sm">Choose an image to begin.</p>
              </div>
            )}
          </div>

          {result && (
            <div className="flex flex-wrap gap-2 border-t border-slate-200 p-4">
              <button
                type="button"
                onClick={() =>
                  downloadBase64(
                    result.uncertainty_map,
                    'uncertainty-map.png'
                  )
                }
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                <Download className="h-4 w-4" />
                Download disagreement map
              </button>

              <button
                type="button"
                onClick={() =>
                  downloadBase64(result.refined_mask, 'refined-mask.png')
                }
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                <Download className="h-4 w-4" />
                Download refined mask
              </button>
            </div>
          )}
        </div>

        <div className="space-y-5">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="font-semibold text-slate-900">Analysis summary</h2>

            {!result ? (
              <p className="mt-3 text-sm leading-6 text-slate-500">
                Run reliability analysis to view the candidate scores,
                disagreement visualization, and refinement statistics.
              </p>
            ) : (
              <div className="mt-4 space-y-4">
                <div className="rounded-xl bg-slate-50 p-3">
                  <p className="text-xs text-slate-500">Image dimensions</p>
                  <p className="mt-1 font-semibold text-slate-900">
                    {result.width} × {result.height}
                  </p>
                </div>

                <div>
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Candidate model scores
                  </p>
                  <div className="space-y-2">
                    {result.candidate_scores.map((score, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between rounded-lg border border-slate-200 px-3 py-2"
                      >
                        <span className="text-sm text-slate-700">
                          Candidate {index + 1}
                        </span>
                        <span className="font-mono text-sm font-semibold text-slate-900">
                          {score.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-xl border border-teal-100 bg-teal-50 p-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-teal-900">
                    <CheckCircle2 className="h-4 w-4" />
                    Refinement completed
                  </div>
                  <p className="mt-2 text-xs leading-5 text-teal-900">
                    {result.refinement}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl border border-slate-200 p-3">
                    <p className="text-xs text-slate-500">Before</p>
                    <p className="mt-1 text-lg font-semibold text-slate-900">
                      {result.original_foreground_pixels.toLocaleString()}
                    </p>
                    <p className="text-xs text-slate-500">foreground pixels</p>
                  </div>

                  <div className="rounded-xl border border-slate-200 p-3">
                    <p className="text-xs text-slate-500">After</p>
                    <p className="mt-1 text-lg font-semibold text-slate-900">
                      {result.refined_foreground_pixels.toLocaleString()}
                    </p>
                    <p className="text-xs text-slate-500">foreground pixels</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4">
            <div className="flex items-start gap-3">
              <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-700" />
              <div>
                <h3 className="text-sm font-semibold text-amber-950">
                  Research interpretation
                </h3>
                <p className="mt-1 text-xs leading-5 text-amber-900">
                  Candidate disagreement is a diagnostic visualization, not
                  calibrated uncertainty. Morphological refinement may remove
                  artifacts but can also remove true lesion details. Validate
                  against ground-truth masks before drawing conclusions.
                  This tool is not for clinical diagnosis.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {result && (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-semibold text-slate-900">Workflow</h2>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            {[
              {
                title: '1. Candidate generation',
                text: 'MedSAM returns multiple masks for the automatically predicted bounding box.',
              },
              {
                title: '2. Disagreement estimation',
                text: 'Pixels where candidates differ are highlighted for inspection.',
              },
              {
                title: '3. Mask refinement',
                text: 'A post-processing operation is applied to a copy of the selected mask.',
              },
            ].map((step) => (
              <div
                key={step.title}
                className="rounded-xl border border-slate-200 p-4"
              >
                <h3 className="text-sm font-semibold text-slate-900">
                  {step.title}
                </h3>
                <p className="mt-2 text-xs leading-5 text-slate-600">
                  {step.text}
                </p>
              </div>
            ))}
          </div>

          <p className="mt-4 text-xs leading-5 text-slate-500">{result.note}</p>
        </section>
      )}
    </div>
  );
}
