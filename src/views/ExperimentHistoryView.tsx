
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Experiment History
 * - Saved experiment history
 * - Original AI prediction and reviewed mask thumbnails
 * - Review dates
 * - Dice, IoU, precision and recall comparison
 */

import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  Activity,
  Clock3,
  RefreshCw,
  Search,
  Image as ImageIcon,
  Layers3,
  Cpu,
  ChevronLeft,
  ExternalLink,
  AlertCircle,
  CheckCircle2,
  FileImage,
  Download,
  FileSpreadsheet,
  GitCompareArrows,
  CalendarClock,
  UserRoundCheck,
  TrendingUp,
  TrendingDown,
} from "lucide-react";

const API = "http://127.0.0.1:8000";

type Metrics = {
  dice: number;
  iou: number;
  precision: number;
  recall: number;
  true_positive_pixels?: number;
  false_positive_pixels?: number;
  false_negative_pixels?: number;
};

type Experiment = {
  id: string;
  created_at: string;
  status: string;
  filename: string;
  model: string;
  prompt_checkpoint?: string | null;
  medsam_checkpoint?: string | null;
  device?: string | null;
  inference_seconds?: number | null;
  medsam_score?: number | null;
  dice?: number | null;
  iou?: number | null;
  precision?: number | null;
  recall?: number | null;
  error?: string | null;
  bbox?: number[] | null;
  config?: Record<string, unknown> | null;
  input_url?: string | null;
  prediction_url?: string | null;
};

type ReviewedMask = {
  review_id: string;
  experiment_id: string;
  filename?: string;
  created_at: string;
  mask_url?: string | null;
  review_date?: string | null;
};

type ReviewEvaluation = {
  review_id: string;
  experiment_id: string;
  filename: string;
  ground_truth_file: string;
  original_ai_metrics: Metrics;
  reviewed_metrics: Metrics;
  changes: {
    dice: number;
    iou: number;
    precision: number;
    recall: number;
  };
};

type HistoryResponse = {
  experiments: Experiment[];
};

type ReviewsResponse = {
  reviews?: ReviewedMask[];
  reviewed_masks?: ReviewedMask[];
};

function formatDate(value?: string | null) {
  if (!value) return "Date unavailable";

  const date = new Date(value);

  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString();
}

function metric(
  value: number | null | undefined,
  digits = 4
) {
  return typeof value === "number" &&
    Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function apiUrl(path?: string | null) {
  if (!path) return "";

  return /^https?:\/\//i.test(path)
    ? path
    : `${API}${path.startsWith("/") ? "" : "/"}${path}`;
}

function Stat({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
        <span>{label}</span>
        <span className="text-teal-700">{icon}</span>
      </div>

      <div className="break-words text-xl font-semibold text-slate-900">
        {value}
      </div>
    </div>
  );
}

function Detail({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-slate-100 py-2 last:border-0">
      <span className="text-xs text-slate-500">{label}</span>
      <span className="max-w-[65%] break-words text-right text-xs font-medium text-slate-800">
        {value}
      </span>
    </div>
  );
}

function ArtifactImage({
  title,
  url,
  subtitle,
}: {
  title: string;
  url: string;
  subtitle?: string;
}) {
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [url]);

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
      <div className="flex items-center justify-between gap-2 border-b border-slate-100 bg-slate-50 px-3 py-3">
        <div className="min-w-0">
          <h3 className="text-xs font-semibold text-slate-700">
            {title}
          </h3>
          {subtitle && (
            <p className="mt-1 text-[11px] text-slate-500">
              {subtitle}
            </p>
          )}
        </div>

        {url && (
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            className="shrink-0 text-teal-700 hover:text-teal-900"
            title="Open image in new tab"
          >
            <ExternalLink size={14} />
          </a>
        )}
      </div>

      <div className="flex min-h-[220px] items-center justify-center bg-slate-950 p-3 sm:min-h-[260px]">
        {url && !failed ? (
          <img
            src={url}
            alt={title}
            onError={() => setFailed(true)}
            className="max-h-[360px] max-w-full object-contain"
          />
        ) : (
          <div className="px-4 text-center text-xs leading-5 text-slate-400">
            <ImageIcon size={22} className="mx-auto mb-2" />
            {failed
              ? "Could not load this saved image."
              : "No image is available."}
          </div>
        )}
      </div>
    </div>
  );
}

function ComparisonTable({
  evaluation,
}: {
  evaluation: ReviewEvaluation;
}) {
  const rows = [
    ["Dice score", "dice"],
    ["IoU", "iou"],
    ["Precision", "precision"],
    ["Recall", "recall"],
  ] as const;

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
      <div className="border-b border-slate-100 p-4">
        <h3 className="font-semibold text-slate-900">
          AI vs reviewed mask
        </h3>
        <p className="mt-1 text-xs leading-5 text-slate-500">
          Both masks are compared with the same ground-truth annotation.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-left text-sm">
          <thead className="bg-slate-50 text-xs text-slate-500">
            <tr>
              <th className="px-4 py-3 font-medium">Metric</th>
              <th className="px-4 py-3 font-medium">Original AI</th>
              <th className="px-4 py-3 font-medium">Reviewed</th>
              <th className="px-4 py-3 font-medium">Change</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-100">
            {rows.map(([label, key]) => {
              const original =
                evaluation.original_ai_metrics[key];
              const reviewed =
                evaluation.reviewed_metrics[key];
              const change = evaluation.changes[key];

              return (
                <tr key={key}>
                  <td className="px-4 py-3 font-medium text-slate-700">
                    {label}
                  </td>

                  <td className="px-4 py-3 tabular-nums text-slate-600">
                    {metric(original)}
                  </td>

                  <td className="px-4 py-3 font-semibold tabular-nums text-slate-900">
                    {metric(reviewed)}
                  </td>

                  <td
                    className={`px-4 py-3 font-semibold tabular-nums ${
                      change > 0
                        ? "text-emerald-700"
                        : change < 0
                          ? "text-rose-700"
                          : "text-slate-500"
                    }`}
                  >
                    <span className="inline-flex items-center gap-1">
                      {change > 0 ? (
                        <TrendingUp size={14} />
                      ) : change < 0 ? (
                        <TrendingDown size={14} />
                      ) : null}

                      {change > 0 ? "+" : ""}
                      {metric(change)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="border-t border-slate-100 bg-slate-50/70 px-4 py-3">
        <p className="text-xs leading-5 text-slate-500">
          Ground-truth annotation:{" "}
          <span className="font-medium text-slate-700">
            {evaluation.ground_truth_file}
          </span>
        </p>
        <p className="mt-1 text-xs leading-5 text-slate-500">
          Positive changes indicate higher metric values for the reviewed
          mask. Manual editing does not guarantee an improvement.
        </p>
      </div>
    </div>
  );
}

export function ExperimentHistoryView() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [reviews, setReviews] = useState<ReviewedMask[]>([]);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedReviewId, setSelectedReviewId] = useState<string | null>(
    null
  );

  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const [evaluation, setEvaluation] =
    useState<ReviewEvaluation | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");

  const loadHistory = useCallback(async (quiet = false) => {
    if (quiet) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    setError("");
    setReviewError("");

    const results = await Promise.allSettled([
      fetch(`${API}/api/experiments?limit=100`),
      fetch(`${API}/api/reviewed-masks?limit=100`),
    ]);

    try {
      const experimentsResult = results[0];

      if (experimentsResult.status === "rejected") {
        throw new Error(
          "Could not connect to the experiment history API."
        );
      }

      const response = experimentsResult.value;
      const body = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          body?.detail ||
            `Could not load experiments (${response.status}).`
        );
      }

      const rows: Experiment[] = Array.isArray(body?.experiments)
        ? body.experiments
        : [];

      setExperiments(rows);

      setSelectedId((current) =>
        current && rows.some((item) => item.id === current)
          ? current
          : rows[0]?.id ?? null
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not load experiment history."
      );
    }

    try {
      const reviewsResult = results[1];

      if (reviewsResult.status === "rejected") {
        throw new Error("Could not connect to the reviewed-mask API.");
      }

      const response = reviewsResult.value;
      const body = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          body?.detail ||
            `Could not load reviewed masks (${response.status}).`
        );
      }

      const rows: ReviewedMask[] = Array.isArray(body?.reviews)
        ? body.reviews
        : Array.isArray(body?.reviewed_masks)
          ? body.reviewed_masks
          : [];

      setReviews(rows);

      setSelectedReviewId((current) =>
        current && rows.some((item) => item.review_id === current)
          ? current
          : rows[0]?.review_id ?? null
      );
    } catch (err) {
      setReviews([]);
      setSelectedReviewId(null);
      setReviewError(
        err instanceof Error
          ? err.message
          : "Could not load reviewed annotations."
      );
    }

    setLoading(false);
    setRefreshing(false);
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) return experiments;

    return experiments.filter((item) =>
      [
        item.filename,
        item.id,
        item.model,
        item.status,
        item.device,
      ].some((value) =>
        String(value ?? "").toLowerCase().includes(query)
      )
    );
  }, [experiments, search]);

  const selected =
    experiments.find((item) => item.id === selectedId) ?? null;

  const selectedReview =
    reviews.find((item) => item.review_id === selectedReviewId) ?? null;

  const selectedReviewExperiment = selectedReview
    ? experiments.find(
        (item) => item.id === selectedReview.experiment_id
      ) ?? null
    : null;

  const completedCount = experiments.filter(
    (item) => item.status === "completed"
  ).length;

  const timedRuns = experiments.filter(
    (item) => typeof item.inference_seconds === "number"
  );

  const averageTime = timedRuns.length
    ? timedRuns.reduce(
        (sum, item) => sum + (item.inference_seconds ?? 0),
        0
      ) / timedRuns.length
    : null;

  useEffect(() => {
    if (!selectedReview) {
      setEvaluation(null);
      setEvaluationError("");
      return;
    }

    const controller = new AbortController();

    async function loadEvaluation() {
      setEvaluationLoading(true);
      setEvaluation(null);
      setEvaluationError("");

      try {
        const response = await fetch(
          `${API}/api/reviewed-masks/${encodeURIComponent(
            selectedReview!.review_id
          )}/evaluate`,
          { signal: controller.signal }
        );

        const body = await response.json().catch(() => null);

        if (!response.ok) {
          throw new Error(
            body?.detail ||
              `Metric evaluation failed (${response.status}).`
          );
        }

        setEvaluation(body as ReviewEvaluation);
      } catch (err) {
        if (
          err instanceof Error &&
          err.name === "AbortError"
        ) {
          return;
        }

        setEvaluationError(
          err instanceof Error
            ? err.message
            : "Could not calculate segmentation metrics."
        );
      } finally {
        if (!controller.signal.aborted) {
          setEvaluationLoading(false);
        }
      }
    }

    void loadEvaluation();

    return () => controller.abort();
  }, [selectedReview?.review_id]);

  return (
    <div className="mx-auto max-w-[1500px] space-y-6 text-slate-900">
      <header className="flex flex-col justify-between gap-4 rounded-3xl border border-white bg-white/90 p-6 shadow-sm sm:flex-row sm:items-end">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-teal-700">
            <Activity size={15} />
            Research workspace
          </div>

          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Experiment history
            <span className="text-teal-700">.</span>
          </h1>

          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            Inspect saved segmentation runs, original predictions, reviewed
            annotations, and their quantitative evaluation.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <a
            href={`${API}/api/experiments/export/history.csv`}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:border-teal-300 hover:bg-teal-50"
            title="Download all saved experiment records as CSV"
          >
            <FileSpreadsheet size={16} />
            Export CSV
          </a>
          <button
            type="button"
            onClick={() => void loadHistory(true)}
            disabled={refreshing || loading}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:border-teal-300 hover:bg-teal-50 disabled:opacity-50"
          >
            <RefreshCw
              size={16}
              className={refreshing ? "animate-spin" : ""}
            />
            Refresh history
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat
          label="Saved experiments"
          value={String(experiments.length)}
          icon={<Layers3 size={17} />}
        />

        <Stat
          label="Completed runs"
          value={String(completedCount)}
          icon={<CheckCircle2 size={17} />}
        />

        <Stat
          label="Reviewed masks"
          value={String(reviews.length)}
          icon={<UserRoundCheck size={17} />}
        />

        <Stat
          label="Mean inference time"
          value={
            averageTime === null
              ? "—"
              : `${averageTime.toFixed(2)} s`
          }
          icon={<Clock3 size={17} />}
        />
      </div>

      {error && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800"
        >
          <AlertCircle size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">
              Unable to load experiment history
            </p>
            <p className="mt-1">{error}</p>
            <button
              type="button"
              onClick={() => void loadHistory()}
              className="mt-2 font-semibold underline"
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {/* Saved experiments */}
      <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-col gap-3 border-b border-slate-100 p-4 sm:flex-row sm:items-center sm:justify-between sm:px-5">
          <div>
            <h2 className="font-semibold">Saved runs</h2>
            <p className="mt-1 text-xs text-slate-500">
              Select an experiment to inspect its artifacts and configuration.
            </p>
          </div>

          <label className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
            <Search size={16} className="text-slate-400" />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search filename, ID, model..."
              className="w-full bg-transparent text-sm outline-none sm:w-64"
            />
          </label>
        </div>

        {loading ? (
          <div className="p-12 text-center text-sm text-slate-500">
            <RefreshCw
              size={20}
              className="mx-auto mb-3 animate-spin text-teal-700"
            />
            Loading saved experiments...
          </div>
        ) : filtered.length === 0 ? (
          <div className="p-12 text-center">
            <FileImage
              size={24}
              className="mx-auto mb-3 text-slate-400"
            />
            <h3 className="font-semibold text-slate-800">
              {search
                ? "No matching experiments"
                : "No experiments saved yet"}
            </h3>
            <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">
              Run segmentation in Segmentation Studio, then refresh this page.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="bg-slate-50 text-[11px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="px-5 py-3 font-semibold">Image / ID</th>
                  <th className="px-4 py-3 font-semibold">Status</th>
                  <th className="px-4 py-3 font-semibold">Inference</th>
                  <th className="px-4 py-3 font-semibold">MedSAM score</th>
                  <th className="px-4 py-3 font-semibold">Device</th>
                  <th className="px-4 py-3 font-semibold">Created</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-100">
                {filtered.map((item) => (
                  <tr
                    key={item.id}
                    onClick={() => setSelectedId(item.id)}
                    className={`cursor-pointer transition hover:bg-teal-50/60 ${
                      selectedId === item.id ? "bg-teal-50/70" : ""
                    }`}
                  >
                    <td className="px-5 py-4">
                      <p className="max-w-[240px] truncate font-semibold text-slate-800">
                        {item.filename || "Unnamed image"}
                      </p>
                      <p className="mt-1 font-mono text-[10px] text-slate-400">
                        {item.id}
                      </p>
                    </td>

                    <td className="px-4 py-4">
                      <span
                        className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
                          item.status === "completed"
                            ? "bg-emerald-50 text-emerald-700"
                            : item.status === "failed"
                              ? "bg-rose-50 text-rose-700"
                              : "bg-amber-50 text-amber-700"
                        }`}
                      >
                        {item.status}
                      </span>
                    </td>

                    <td className="px-4 py-4 tabular-nums text-slate-700">
                      {typeof item.inference_seconds === "number"
                        ? `${item.inference_seconds.toFixed(2)} s`
                        : "—"}
                    </td>

                    <td className="px-4 py-4 tabular-nums text-slate-700">
                      {metric(item.medsam_score)}
                    </td>

                    <td className="px-4 py-4 text-slate-600">
                      {item.device || "—"}
                    </td>

                    <td className="whitespace-nowrap px-4 py-4 text-xs text-slate-500">
                      {formatDate(item.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Selected experiment */}
      {selected && (
        <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
          <div className="flex flex-col gap-3 border-b border-slate-100 p-5 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 items-center gap-3">
              <button
                type="button"
                onClick={() => setSelectedId(null)}
                className="rounded-lg border border-slate-200 p-2 text-slate-500 hover:bg-slate-50"
                title="Close details"
              >
                <ChevronLeft size={17} />
              </button>

              <div className="min-w-0">
                <h2 className="truncate font-semibold">
                  {selected.filename}
                </h2>
                <p className="mt-1 font-mono text-[11px] text-slate-400">
                  Experiment {selected.id}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-slate-500">
                {formatDate(selected.created_at)}
              </span>
              {selected.status === "completed" && (
                <a
                  href={`${API}/api/experiments/${encodeURIComponent(selected.id)}/report.pdf`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-teal-700 px-3 py-2 text-xs font-semibold text-white hover:bg-teal-800"
                  title="Generate a PDF report from this saved experiment"
                >
                  <Download size={14} />
                  Download PDF report
                </a>
              )}
            </div>
          </div>

          <div className="grid gap-5 p-5 xl:grid-cols-[minmax(0,1fr)_320px]">
            <div className="grid gap-4 md:grid-cols-2">
              <ArtifactImage
                title="Original ultrasound"
                url={apiUrl(selected.input_url)}
              />

              <ArtifactImage
                title="Original AI prediction"
                url={apiUrl(selected.prediction_url)}
                subtitle="Unmodified model output"
              />
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-200 p-4">
                <h3 className="mb-3 text-sm font-semibold">
                  Run configuration
                </h3>

                <Detail label="Model" value={selected.model} />
                <Detail label="Device" value={selected.device || "—"} />

                <Detail
                  label="Inference time"
                  value={
                    typeof selected.inference_seconds === "number"
                      ? `${selected.inference_seconds.toFixed(3)} seconds`
                      : "—"
                  }
                />

                <Detail
                  label="MedSAM score"
                  value={metric(selected.medsam_score)}
                />

                <Detail label="AI Dice" value={metric(selected.dice)} />
                <Detail label="AI IoU" value={metric(selected.iou)} />
                <Detail
                  label="AI precision"
                  value={metric(selected.precision)}
                />
                <Detail
                  label="AI recall"
                  value={metric(selected.recall)}
                />
              </div>

              {selected.bbox && (
                <div className="rounded-2xl border border-slate-200 p-4">
                  <h3 className="mb-2 text-sm font-semibold">
                    Predicted bounding box
                  </h3>
                  <p className="break-words font-mono text-xs leading-5 text-slate-600">
                    [
                    {selected.bbox
                      .map((value) => Number(value).toFixed(1))
                      .join(", ")}
                    ]
                  </p>
                </div>
              )}

              {selected.config && (
                <div className="rounded-2xl border border-slate-200 p-4">
                  <h3 className="mb-2 text-sm font-semibold">
                    Configuration details
                  </h3>
                  <pre className="overflow-x-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-slate-600">
                    {JSON.stringify(selected.config, null, 2)}
                  </pre>
                </div>
              )}

              {selected.error && (
                <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
                  {selected.error}
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* Reviewed annotations */}
      <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-col gap-3 border-b border-slate-100 p-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-2 text-teal-700">
              <GitCompareArrows size={18} />
              <span className="text-xs font-semibold uppercase tracking-wider">
                Annotation review
              </span>
            </div>

            <h2 className="text-xl font-semibold">
              Reviewed masks and metric comparison
            </h2>

            <p className="mt-1 text-sm leading-6 text-slate-500">
              Select a saved brush-and-eraser annotation to compare it with
              its original AI prediction.
            </p>
          </div>

          <span className="self-start rounded-full bg-teal-50 px-3 py-1.5 text-xs font-semibold text-teal-800">
            {reviews.length} reviewed mask{reviews.length === 1 ? "" : "s"}
          </span>
        </div>

        {reviewError && (
          <div className="m-5 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            <AlertCircle size={18} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold">
                Reviewed-mask history unavailable
              </p>
              <p className="mt-1">{reviewError}</p>
              <p className="mt-2 text-xs leading-5">
                Check that the backend provides GET /api/reviewed-masks and
                returns the saved annotations.
              </p>
            </div>
          </div>
        )}

        {!reviewError && reviews.length === 0 && (
          <div className="p-10 text-center">
            <UserRoundCheck
              size={26}
              className="mx-auto mb-3 text-slate-400"
            />
            <h3 className="font-semibold text-slate-800">
              No reviewed annotations found
            </h3>
            <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-500">
              Save a reviewed mask from Interactive Mask Correction, then
              refresh this page. The original AI prediction should remain
              unchanged.
            </p>
          </div>
        )}

        {reviews.length > 0 && (
          <div className="grid gap-5 p-5 lg:grid-cols-[280px_minmax(0,1fr)]">
            {/* Review selector */}
            <div className="space-y-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Saved annotations
              </h3>

              {reviews.map((review) => {
                const relatedExperiment = experiments.find(
                  (item) => item.id === review.experiment_id
                );

                const active =
                  review.review_id === selectedReviewId;

                return (
                  <button
                    key={review.review_id}
                    type="button"
                    onClick={() =>
                      setSelectedReviewId(review.review_id)
                    }
                    className={`w-full rounded-2xl border p-3 text-left transition ${
                      active
                        ? "border-teal-300 bg-teal-50/70 ring-1 ring-teal-200"
                        : "border-slate-200 hover:border-teal-200 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex gap-3">
                      <div className="h-20 w-20 shrink-0 overflow-hidden rounded-xl bg-slate-950">
                        {review.mask_url ? (
                          <img
                            src={apiUrl(review.mask_url)}
                            alt="Reviewed mask thumbnail"
                            className="h-full w-full object-contain"
                          />
                        ) : (
                          <div className="flex h-full items-center justify-center text-slate-400">
                            <ImageIcon size={20} />
                          </div>
                        )}
                      </div>

                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-semibold text-slate-800">
                          {review.filename ||
                            relatedExperiment?.filename ||
                            "Reviewed annotation"}
                        </p>

                        <p className="mt-1 line-clamp-2 break-all font-mono text-[10px] text-slate-400">
                          Review {review.review_id}
                        </p>

                        <p className="mt-2 flex items-center gap-1 text-[11px] text-slate-500">
                          <CalendarClock size={12} />
                          {formatDate(
                            review.review_date || review.created_at
                          )}
                        </p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Selected review details */}
            <div className="min-w-0 space-y-5">
              {selectedReview && (
                <>
                  <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                    <div>
                      <h3 className="text-lg font-semibold text-slate-900">
                        {selectedReview.filename ||
                          selectedReviewExperiment?.filename ||
                          "Reviewed annotation"}
                      </h3>

                      <p className="mt-1 text-xs text-slate-500">
                        Reviewed on{" "}
                        {formatDate(
                          selectedReview.review_date ||
                            selectedReview.created_at
                        )}
                      </p>

                      <p className="mt-1 break-all font-mono text-[10px] text-slate-400">
                        Review ID: {selectedReview.review_id}
                      </p>
                    </div>

                    <span className="self-start rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700">
                      Saved annotation
                    </span>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <ArtifactImage
                      title="Original AI prediction"
                      url={
                        selectedReviewExperiment
                          ? apiUrl(
                              selectedReviewExperiment.prediction_url
                            )
                          : ""
                      }
                      subtitle="Preserved model output"
                    />

                    <ArtifactImage
                      title="Reviewed mask"
                      url={apiUrl(selectedReview.mask_url)}
                      subtitle="Your saved brush-and-eraser edits"
                    />
                  </div>

                  {evaluationLoading && (
                    <div className="rounded-2xl border border-slate-200 p-6 text-center text-sm text-slate-500">
                      <RefreshCw
                        size={20}
                        className="mx-auto mb-3 animate-spin text-teal-700"
                      />
                      Calculating comparison metrics…
                    </div>
                  )}

                  {evaluationError && !evaluationLoading && (
                    <div className="flex items-start gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
                      <AlertCircle
                        size={18}
                        className="mt-0.5 shrink-0"
                      />
                      <div>
                        <p className="font-semibold">
                          Comparison metrics unavailable
                        </p>
                        <p className="mt-1 leading-6">
                          {evaluationError}
                        </p>
                        <p className="mt-2 text-xs leading-5">
                          Metrics require a linked experiment, its original
                          prediction, and an exact matching BUSI ground-truth
                          annotation. The review can still be inspected
                          without these metrics.
                        </p>
                      </div>
                    </div>
                  )}

                  {evaluation && !evaluationLoading && (
                    <ComparisonTable evaluation={evaluation} />
                  )}

                  {selectedReviewExperiment && (
                    <div className="rounded-2xl border border-slate-200 p-4">
                      <h4 className="mb-2 text-sm font-semibold">
                        Linked experiment
                      </h4>

                      <Detail
                        label="Experiment ID"
                        value={selectedReview.experiment_id}
                      />

                      <Detail
                        label="Model"
                        value={selectedReviewExperiment.model}
                      />

                      <Detail
                        label="Original inference time"
                        value={
                          typeof selectedReviewExperiment.inference_seconds ===
                          "number"
                            ? `${selectedReviewExperiment.inference_seconds.toFixed(3)} seconds`
                            : "—"
                        }
                      />

                      <Detail
                        label="Device"
                        value={
                          selectedReviewExperiment.device || "—"
                        }
                      />
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </section>

      <p className="text-[11px] leading-5 text-slate-400">
        Research prototype only. Dice, IoU, precision, and recall are valid
        only when calculated against an appropriate reference annotation.
        The MedSAM score is not a substitute for ground-truth evaluation.
      </p>
    </div>
  );
}

export default ExperimentHistoryView;




