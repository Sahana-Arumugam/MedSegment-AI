
import React, { useEffect, useRef, useState } from "react";
import MaskCorrectionEditor from "../components/MaskCorrectionEditor";
import {
  Activity,
  ArrowDownToLine,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Crosshair,
  FileImage,
  Focus,
  Image as ImageIcon,
  Layers3,
  LoaderCircle,
  Maximize2,
  RefreshCw,
  ScanLine,
  Upload,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

const API = "http://127.0.0.1:8000";

type ViewMode = "original" | "ground_truth" | "predicted" | "overlay";

type DatasetImage = {
  id: string;
  filename: string;
  category: string;
  image_url: string;
  mask_urls: string[];
  has_mask: boolean;
};

type DatasetResponse = {
  total_images: number;
  images: DatasetImage[];
};

type Prediction = {
  filename: string;
  width: number;
  height: number;
  device: string;
  bbox_xyxy: number[];
  mask_png_base64: string;
  mask_mime_type: string;
  medsam_score: number;
  foreground_pixels: number;
  model: string;
  note: string;
  experiment_id?: string;
};
const views: { id: ViewMode; label: string; short: string }[] = [
  { id: "original", label: "Original image", short: "Original" },
  { id: "ground_truth", label: "Ground truth", short: "Ground truth" },
  { id: "predicted", label: "AI prediction", short: "Prediction" },
  { id: "overlay", label: "Segmentation overlay", short: "Overlay" },
];

function createMaskDataUrl(prediction: Prediction): string {
  return `data:${prediction.mask_mime_type};base64,${prediction.mask_png_base64}`;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load an image required for visualization."));
    image.src = src;
  });
}

async function createOverlayDataUrl(
  originalUrl: string,
  maskUrl: string,
  bbox: number[]
): Promise<string> {
  const [original, mask] = await Promise.all([
    loadImage(originalUrl),
    loadImage(maskUrl),
  ]);

  const canvas = document.createElement("canvas");
  canvas.width = original.naturalWidth;
  canvas.height = original.naturalHeight;

  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not create the segmentation overlay.");

  context.drawImage(original, 0, 0, canvas.width, canvas.height);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = canvas.width;
  maskCanvas.height = canvas.height;

  const maskContext = maskCanvas.getContext("2d");
  if (!maskContext) throw new Error("Could not process the predicted mask.");

  maskContext.drawImage(mask, 0, 0, canvas.width, canvas.height);

  const maskPixels = maskContext.getImageData(
    0, 0, canvas.width, canvas.height
  );
  const overlayPixels = context.getImageData(
    0, 0, canvas.width, canvas.height
  );

  // Red overlay for foreground pixels in the binary mask.
  for (let i = 0; i < maskPixels.data.length; i += 4) {
    if (maskPixels.data[i] > 127) {
      overlayPixels.data[i] = overlayPixels.data[i] * 0.35 + 255 * 0.65;
      overlayPixels.data[i + 1] = overlayPixels.data[i + 1] * 0.35;
      overlayPixels.data[i + 2] = overlayPixels.data[i + 2] * 0.35;
    }
  }

  context.putImageData(overlayPixels, 0, 0);

  // Draw the predicted bounding box in the original image coordinates.
  if (bbox.length >= 4 && bbox.every(Number.isFinite)) {
    const [x1, y1, x2, y2] = bbox;
    context.save();
    context.strokeStyle = "#f97316";
    context.lineWidth = Math.max(2, canvas.width / 400);
    context.setLineDash([8, 5]);
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.restore();
  }

  return canvas.toDataURL("image/png");
}

function Metric({
  label,
  value,
  suffix = "",
  icon,
}: {
  label: string;
  value: string;
  suffix?: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="group rounded-2xl border border-slate-200/80 bg-white p-4 transition-all duration-200 hover:-translate-y-0.5 hover:border-teal-200 hover:shadow-md hover:shadow-slate-200/40">
      <div className="mb-4 flex items-center justify-between">
        <span className="text-xs font-medium tracking-wide text-slate-500">
          {label}
        </span>
        <span className="rounded-xl bg-gradient-to-br from-teal-50 to-emerald-50 p-2 text-teal-700 transition-colors group-hover:from-teal-100 group-hover:to-emerald-100">
          {icon}
        </span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-semibold tracking-tight text-slate-900">
          {value}
        </span>
        {suffix && <span className="text-xs text-slate-500">{suffix}</span>}
      </div>
    </div>
  );
}

export default function SegmentationWorkspace() {
  const fileInput = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [mode, setMode] = useState<ViewMode>("original");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [groundTruthUrl, setGroundTruthUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [zoom, setZoom] = useState(1);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    return () => {
      if (previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const clearResults = () => {
    setPrediction(null);
    setMaskUrl(null);
    setOverlayUrl(null);
    setGroundTruthUrl(null);
    setElapsed(null);
  };

  const chooseFile = (nextFile?: File) => {
    if (!nextFile) return;

    const allowed = /\.(png|jpe?g|bmp|tiff?)$/i;
    if (!allowed.test(nextFile.name)) {
      setError("Choose a PNG, JPG, JPEG, BMP, or TIFF image.");
      return;
    }

    if (nextFile.size > 15 * 1024 * 1024) {
      setError("The image must be 15 MB or smaller.");
      return;
    }

    setError("");
    clearResults();
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setMode("original");
    setZoom(1);

    // Allow choosing the same file again later.
    if (fileInput.current) fileInput.current.value = "";
  };

  const runSegmentation = async () => {
    if (!file || !previewUrl) {
      setError("Upload an ultrasound image to begin.");
      return;
    }

    setBusy(true);
    setError("");
    clearResults();

    const start = performance.now();

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API}/api/segment`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json().catch(() => null);

      if (!response.ok) {
        const detail =
          result && typeof result.detail === "string"
            ? result.detail
            : `Segmentation failed (${response.status}).`;
        throw new Error(detail);
      }

      if (
        !result ||
        typeof result.mask_png_base64 !== "string" ||
        !result.mask_png_base64 ||
        typeof result.mask_mime_type !== "string" ||
        !Array.isArray(result.bbox_xyxy) ||
        result.bbox_xyxy.length < 4
      ) {
        throw new Error(
          "The backend response is missing the predicted mask or bounding box. Check the /api/segment response."
        );
      }

      const data = result as Prediction;
      const generatedMask = createMaskDataUrl(data);

      // Validate that the returned predicted mask is readable.
      await loadImage(generatedMask);

      const generatedOverlay = await createOverlayDataUrl(
        previewUrl,
        generatedMask,
        data.bbox_xyxy
      );

      // Ground-truth annotations are available only when the uploaded file
      // matches an image in the BUSI dataset by filename.
      let matchedGroundTruthUrl: string | null = null;
      try {
        const datasetResponse = await fetch(`${API}/api/dataset`);
        if (datasetResponse.ok) {
          const dataset = (await datasetResponse.json()) as DatasetResponse;
          const uploadedName = file.name.trim().toLowerCase();
          const matches = (dataset.images || []).filter(
            (item) => item.filename.trim().toLowerCase() === uploadedName &&
              item.has_mask && Array.isArray(item.mask_urls) && item.mask_urls.length > 0
          );

          if (matches.length > 0) {
            // Use the first annotation by default. Some BUSI images have
            // multiple expert/reference masks; those can be exposed separately later.
            const maskPath = matches[0].mask_urls[0];
            const candidateUrl = /^https?:\/\//i.test(maskPath)
              ? maskPath
              : `${API}${maskPath.startsWith("/") ? "" : "/"}${maskPath}`;
            await loadImage(candidateUrl);
            matchedGroundTruthUrl = candidateUrl;
          }
        }
      } catch (groundTruthError) {
        // Prediction should still succeed if a reference mask is unavailable.
        console.warn("Could not load BUSI ground-truth mask:", groundTruthError);
      }

      setPrediction(data);
      setMaskUrl(generatedMask);
      setOverlayUrl(generatedOverlay);
      setGroundTruthUrl(matchedGroundTruthUrl);
      setElapsed((performance.now() - start) / 1000);
      setMode("overlay");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Segmentation failed. Check the backend and model checkpoints."
      );
    } finally {
      setBusy(false);
    }
  };

  const activeImage = (() => {
    switch (mode) {
      case "original":
        return previewUrl;
      case "ground_truth":
        return groundTruthUrl;
      case "predicted":
        return maskUrl;
      case "overlay":
        return overlayUrl;
      default:
        return previewUrl;
    }
  })();

  const downloadImage = () => {
    if (!activeImage) return;

    const link = document.createElement("a");
    link.href = activeImage;
    link.download = `medsegment-${mode}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const bbox = prediction?.bbox_xyxy ?? null;

  return (
    <main className="min-h-full bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-teal-50/70 via-slate-50 to-white px-4 py-7 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1600px] space-y-7">
        <header className="flex flex-col justify-between gap-5 rounded-3xl border border-white/80 bg-white/85 p-6 shadow-[0_18px_60px_-35px_rgba(15,118,110,0.28)] backdrop-blur lg:flex-row lg:items-end">
          <div>
            <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">
              <span className="h-2 w-2 rounded-full bg-teal-500" />
              Medical imaging / Workspace
            </div>
            <h1 className="text-3xl font-bold tracking-[-0.04em] text-slate-950 sm:text-4xl lg:text-[2.7rem]">
              Image segmentation<span className="text-teal-700">.</span>
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
              Generate an automatic bounding-box prompt, segment an ultrasound
              image, and inspect the predicted mask.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-3.5 py-2.5 text-xs font-medium text-slate-600 shadow-sm">
              <span
                className={`h-2 w-2 rounded-full ${
                  busy ? "animate-pulse bg-amber-400" : "bg-emerald-500"
                }`}
              />
              {busy ? "Processing image" : prediction ? "Segmentation complete" : "Workspace ready"}
            </div>
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              className="inline-flex items-center gap-2 rounded-xl border border-teal-200 bg-white px-4 py-2.5 text-sm font-semibold text-teal-800 shadow-sm transition duration-200 hover:-translate-y-0.5 hover:border-teal-400 hover:bg-teal-50 hover:shadow-md"
            >
              <Upload size={16} />
              Upload image
            </button>
          </div>
        </header>

        <input
          ref={fileInput}
          type="file"
          accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,image/*"
          className="hidden"
          onChange={(event) => chooseFile(event.target.files?.[0])}
        />

        <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
          <section className="min-w-0 space-y-4">
            <div className="overflow-hidden rounded-3xl border border-white bg-white shadow-[0_18px_55px_-35px_rgba(15,23,42,0.28)] ring-1 ring-slate-200/80">
              <div className="flex flex-col justify-between gap-4 border-b border-slate-100 bg-gradient-to-r from-white via-white to-teal-50/60 px-5 py-4 sm:flex-row sm:items-center">
                <div className="flex items-center gap-3">
                  <div className="rounded-xl bg-gradient-to-br from-teal-100 to-emerald-50 p-2.5 text-teal-800 ring-1 ring-teal-100">
                    <ScanLine size={19} />
                  </div>
                  <div className="min-w-0">
                    <h2 className="text-sm font-semibold text-slate-900">Image viewer</h2>
                    <p className="mt-0.5 max-w-xs truncate text-xs text-slate-500">
                      {file ? file.name : "No image selected"}
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-1 rounded-xl border border-slate-200/80 bg-slate-100/80 p-1">
                  {views.map((view) => {
                    const disabled =
                      (view.id === "ground_truth" && !groundTruthUrl) ||
                      (view.id === "predicted" && !maskUrl) ||
                      (view.id === "overlay" && !overlayUrl);

                    return (
                      <button
                        key={view.id}
                        type="button"
                        onClick={() => setMode(view.id)}
                        disabled={disabled}
                        title={
                          view.id === "ground_truth" && !groundTruthUrl
                            ? "No matching BUSI annotation was found for this uploaded filename."
                            : view.label
                        }
                        className={`rounded-lg px-3 py-2 text-xs font-semibold transition duration-200 ${
                          mode === view.id
                            ? "bg-white text-teal-800 shadow-sm ring-1 ring-teal-100"
                            : "text-slate-500 hover:text-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                        }`}
                      >
                        {view.short}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget as Node)) {
                    setDragging(false);
                  }
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  chooseFile(event.dataTransfer.files?.[0]);
                }}
                className={`relative m-4 flex min-h-[360px] items-center justify-center overflow-hidden rounded-2xl border transition sm:min-h-[480px] ${
                  dragging
                    ? "border-teal-400 bg-teal-50"
                    : "border-slate-700/80 bg-[#0b1420] shadow-inner shadow-black/20"
                }`}
              >
                <div
                  className="pointer-events-none absolute inset-0 opacity-[0.10]"
                  style={{
                    backgroundImage:
                      "linear-gradient(rgba(148,163,184,.28) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,.28) 1px, transparent 1px)",
                    backgroundSize: "32px 32px",
                  }}
                />

                {activeImage ? (
                  <>
                    <img
                      src={activeImage}
                      alt={`Ultrasound ${mode} view`}
                      className="relative z-10 max-h-[560px] max-w-full rounded-lg object-contain transition-transform duration-200 drop-shadow-2xl"
                      style={{ transform: `scale(${zoom})` }}
                    />
                    {mode === "predicted" && (
                      <div className="absolute left-4 top-4 z-20 rounded-lg border border-teal-300/30 bg-teal-950/90 px-3 py-2 text-xs text-teal-100">
                        Predicted binary mask
                      </div>
                    )}
                    {mode === "overlay" && (
                      <div className="absolute left-4 top-4 z-20 flex flex-wrap gap-3 rounded-xl border border-white/15 bg-slate-950/85 px-3.5 py-2.5 text-xs text-white shadow-lg backdrop-blur-md">
                        <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm bg-red-500" />Predicted mask</span>
                        <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm bg-orange-400" />Predicted box</span>
                      </div>
                    )}
                    {mode === "ground_truth" && groundTruthUrl && (
                      <div className="absolute left-4 top-4 z-20 rounded-lg bg-slate-900/85 px-3 py-2 text-xs text-white">
                        BUSI ground-truth annotation
                      </div>
                    )}
                  </>
                ) : (
                  <div className="relative z-10 mx-auto max-w-sm px-6 py-12 text-center">
                    <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.06] text-teal-300">
                      <FileImage size={28} strokeWidth={1.5} />
                    </div>
                    <h3 className="text-lg font-medium text-white">Your imaging canvas</h3>
                    <p className="mx-auto mt-2 max-w-xs text-sm leading-6 text-slate-400">
                      Drop an ultrasound image here, or browse your files to
                      start segmentation.
                    </p>
                    <button
                      type="button"
                      onClick={() => fileInput.current?.click()}
                      className="mt-6 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-teal-400 to-emerald-400 px-4 py-2.5 text-sm font-bold text-slate-950 shadow-lg shadow-teal-950/20 transition hover:-translate-y-0.5 hover:from-teal-300 hover:to-emerald-300"
                    >
                      <Upload size={16} />
                      Select an image
                    </button>
                    <p className="mt-4 text-[11px] text-slate-500">
                      PNG · JPG · JPEG · BMP · TIFF
                    </p>
                  </div>
                )}

                <div className="pointer-events-none absolute left-4 bottom-4 z-20 hidden items-center gap-2 text-[10px] uppercase tracking-[0.16em] text-slate-400 sm:flex">
                  <Focus size={13} />
                  Imaging viewport
                </div>

                {activeImage && (
                  <div className="absolute bottom-3 right-3 z-20 flex items-center gap-1 rounded-xl border border-white/10 bg-slate-950/85 p-1.5 shadow-lg backdrop-blur-md">
                    <button
                      type="button"
                      onClick={() => setZoom((value) => Math.max(0.5, value - 0.1))}
                      className="rounded-lg p-2 text-slate-300 transition hover:bg-white/10 hover:text-white"
                      title="Zoom out"
                    >
                      <ZoomOut size={15} />
                    </button>
                    <span className="min-w-12 text-center text-[11px] tabular-nums text-slate-300">
                      {Math.round(zoom * 100)}%
                    </span>
                    <button
                      type="button"
                      onClick={() => setZoom((value) => Math.min(2, value + 0.1))}
                      className="rounded-lg p-2 text-slate-300 transition hover:bg-white/10 hover:text-white"
                      title="Zoom in"
                    >
                      <ZoomIn size={15} />
                    </button>
                    <button
                      type="button"
                      onClick={() => setZoom(1)}
                      className="rounded-lg p-2 text-slate-300 transition hover:bg-white/10 hover:text-white"
                      title="Reset zoom"
                    >
                      <Maximize2 size={15} />
                    </button>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 px-5 pb-4">
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Layers3 size={15} className="text-teal-700" />
                  <span>
                    Viewing:{" "}
                    <span className="font-medium text-slate-700">
                      {views.find((view) => view.id === mode)?.label}
                    </span>
                  </span>
                </div>
                <button
                  type="button"
                  onClick={downloadImage}
                  disabled={!activeImage}
                  className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium text-slate-600 transition hover:bg-slate-100 hover:text-teal-800 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <ArrowDownToLine size={15} />
                  Export current view
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <Metric
                label="MedSAM score"
                value={prediction ? Number(prediction.medsam_score).toFixed(3) : "—"}
                icon={<Activity size={17} />}
              />
              <Metric
                label="Foreground pixels"
                value={prediction ? prediction.foreground_pixels.toLocaleString() : "—"}
                icon={<Layers3 size={17} />}
              />
              <Metric
                label="Image dimensions"
                value={prediction ? `${prediction.width} × ${prediction.height}` : "—"}
                icon={<ImageIcon size={17} />}
              />
              <Metric
                label="Inference time"
                value={elapsed !== null ? elapsed.toFixed(2) : "—"}
                suffix="sec"
                icon={<Clock3 size={17} />}
              />
            </div>

            {prediction && (
              <section className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-[0_12px_35px_-25px_rgba(15,23,42,0.3)] transition hover:border-teal-100">
                <div className="mb-3 flex items-center gap-2">
                  <Crosshair size={17} className="text-teal-700" />
                  <h2 className="text-sm font-semibold text-slate-900">Predicted bounding box</h2>
                </div>
                <p className="text-xs text-slate-500">
                  Coordinates in the original image:{" "}
                  <span className="font-mono text-slate-700">
                    [{prediction.bbox_xyxy.map((value) => Number(value).toFixed(1)).join(", ")}]
                  </span>
                </p>
                <p className="mt-2 break-words text-xs leading-5 text-slate-500">
                  {prediction.model}
                </p>
              </section>
            )}
          </section>

          <aside className="space-y-4">
            <section className="overflow-hidden rounded-3xl border border-white bg-white shadow-[0_18px_55px_-35px_rgba(15,23,42,0.28)] ring-1 ring-slate-200/80">
              <div className="border-b border-slate-100 px-5 py-4">
                <div className="flex items-center gap-2">
                  <div className="rounded-lg bg-teal-50 p-2 text-teal-800">
                    <Activity size={17} />
                  </div>
                  <h2 className="text-sm font-semibold text-slate-900">Segmentation controls</h2>
                </div>
                <p className="mt-2 text-xs leading-5 text-slate-500">
                  Upload an ultrasound image and launch the model pipeline.
                </p>
              </div>

              <div className="space-y-5 p-5">
                <div>
                  <div className="mb-2 flex items-center justify-between">
                    <label className="text-xs font-semibold text-slate-700">Input image</label>
                    <span className="text-[10px] uppercase tracking-wider text-slate-400">Required</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => fileInput.current?.click()}
                    className="group flex w-full items-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-gradient-to-br from-slate-50 to-white p-3.5 text-left transition duration-200 hover:border-teal-400 hover:bg-teal-50/50 hover:shadow-sm"
                  >
                    <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-slate-500 transition group-hover:text-teal-700">
                      <ImageIcon size={18} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium text-slate-800">
                        {file ? file.name : "Choose an image"}
                      </p>
                      <p className="mt-1 text-[11px] text-slate-500">
                        {file ? `${(file.size / 1024).toFixed(0)} KB` : "Browse or drag and drop"}
                      </p>
                    </div>
                    {file ? <Check size={16} className="text-teal-700" /> : <ChevronDown size={16} className="text-slate-400 group-hover:text-teal-700" />}
                  </button>
                </div>

                <div className="rounded-2xl border border-teal-100/80 bg-gradient-to-br from-teal-50/70 to-slate-50 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-700">Analysis pipeline</span>
                    <span className="rounded-full bg-teal-50 px-2 py-1 text-[10px] font-medium text-teal-800">
                      {busy ? "Running" : prediction ? "Completed" : "Ready"}
                    </span>
                  </div>
                  <div className="space-y-3">
                    {[
                      "Image preparation",
                      "Automatic prompt generation",
                      "MedSAM segmentation",
                      "Mask visualization",
                    ].map((label, index) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-[10px] font-semibold ${
                          prediction
                            ? "bg-teal-100 text-teal-800"
                            : busy && index < 3
                              ? "bg-teal-100 text-teal-800"
                              : "border border-slate-200 bg-white text-slate-500"
                        }`}>
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span className="text-xs text-slate-600">{label}</span>
                        <span className="ml-auto">
                          {prediction && <Check size={14} className="text-teal-700" />}
                          {busy && index === 2 && <LoaderCircle size={14} className="animate-spin text-teal-700" />}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={runSegmentation}
                  disabled={!file || busy}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-teal-700 via-teal-700 to-emerald-700 px-4 py-3.5 text-sm font-semibold text-white shadow-lg shadow-teal-900/15 transition duration-200 hover:-translate-y-0.5 hover:from-teal-800 hover:to-emerald-800 hover:shadow-xl disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {busy ? (
                    <>
                      <LoaderCircle size={17} className="animate-spin" />
                      Processing image...
                    </>
                  ) : prediction ? (
                    <>
                      <RefreshCw size={16} />
                      Run again
                    </>
                  ) : (
                    <>
                      <ScanLine size={17} />
                      Run segmentation
                    </>
                  )}
                </button>

                {error && (
                  <div role="alert" className="break-words rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs leading-5 text-rose-700">
                    {error}
                  </div>
                )}

                <div className="flex items-start gap-2 text-[11px] leading-5 text-slate-500">
                  <CircleHelp size={15} className="mt-0.5 shrink-0 text-teal-700" />
                  Experimental research output only. Review masks and validate results against ground-truth annotations.
                </div>
              </div>
            </section>

            <section className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-[0_12px_35px_-25px_rgba(15,23,42,0.3)] transition hover:border-teal-100">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-900">Result summary</h3>
                <span className="text-[10px] uppercase tracking-wider text-slate-400">Current run</span>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3 border-b border-slate-100 pb-3">
                  <span className="text-xs text-slate-500">Image status</span>
                  <span className="text-xs font-medium text-slate-700">{file ? "Loaded" : "Awaiting image"}</span>
                </div>
                <div className="flex items-center justify-between gap-3 border-b border-slate-100 pb-3">
                  <span className="text-xs text-slate-500">Prediction</span>
                  <span className={`text-xs font-medium ${prediction ? "text-teal-700" : "text-slate-400"}`}>
                    {prediction ? "Available" : "Not generated"}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3 border-b border-slate-100 pb-3">
                  <span className="text-xs text-slate-500">Compute device</span>
                  <span className="text-xs font-semibold text-slate-800">{prediction?.device ?? "—"}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-xs text-slate-500">Mask pixels</span>
                  <span className="text-xs font-semibold tabular-nums text-slate-800">
                    {prediction ? prediction.foreground_pixels.toLocaleString() : "—"}
                  </span>
                </div>
              </div>
            </section>
          </aside>
        </div>
                {prediction && maskUrl && previewUrl && file && (
          <MaskCorrectionEditor
            imageUrl={previewUrl}
            maskUrl={maskUrl}
            filename={file.name}
            experimentId={prediction.experiment_id}
          />
        )}
      </div>
    </main>
  );
}
