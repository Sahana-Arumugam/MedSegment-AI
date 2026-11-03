
import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  Brush,
  Eraser,
  Undo2,
  Redo2,
  RotateCcw,
  Save,
  LoaderCircle,
  Download,
  ZoomIn,
  ZoomOut,
  Move,
  Eye,
  Image as ImageIcon,
} from "lucide-react";

const API = "http://127.0.0.1:8000";

type Props = {
  imageUrl: string;
  maskUrl: string;
  filename: string;
  experimentId?: string;
};

type Tool = "brush" | "eraser" | "pan";

export default function MaskCorrectionEditor({
  imageUrl,
  maskUrl,
  filename,
  experimentId,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const drawingRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const historyRef = useRef<string[]>([]);
  const historyIndexRef = useRef(-1);
  const initializedRef = useRef(false);

  const [tool, setTool] = useState<Tool>("brush");
  const [brushSize, setBrushSize] = useState(12);
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [historyVersion, setHistoryVersion] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [showOverlay, setShowOverlay] = useState(true);

  // ----------------------------------------------------------
  // Draw ultrasound first, then place a translucent red mask
  // over the foreground pixels.
  // ----------------------------------------------------------
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const image = imageRef.current;

    if (!canvas || !maskCanvas || !image) return;

    const ctx = canvas.getContext("2d");
    const maskCtx = maskCanvas.getContext("2d");

    if (!ctx || !maskCtx) return;

    const width = image.naturalWidth;
    const height = image.naturalHeight;

    if (!width || !height) return;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    // Always draw the actual ultrasound image as the background.
    ctx.clearRect(0, 0, width, height);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
    ctx.drawImage(image, 0, 0, width, height);

    if (!showOverlay) return;

    const maskPixels = maskCtx.getImageData(
      0,
      0,
      width,
      height
    );

    const overlay = ctx.createImageData(width, height);

    for (let i = 0; i < maskPixels.data.length; i += 4) {
      if (maskPixels.data[i] > 127) {
        overlay.data[i] = 255;
        overlay.data[i + 1] = 45;
        overlay.data[i + 2] = 75;
        overlay.data[i + 3] = 115;
      }
    }

    const overlayCanvas = document.createElement("canvas");
    overlayCanvas.width = width;
    overlayCanvas.height = height;

    const overlayCtx = overlayCanvas.getContext("2d");
    if (!overlayCtx) return;

    overlayCtx.putImageData(overlay, 0, 0);
    ctx.drawImage(overlayCanvas, 0, 0);
  }, [showOverlay]);

  // ----------------------------------------------------------
  // Capture snapshots for undo and redo.
  // ----------------------------------------------------------
  const captureMask = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;

    const snapshot = maskCanvas.toDataURL("image/png");

    const history = historyRef.current.slice(
      0,
      historyIndexRef.current + 1
    );

    history.push(snapshot);

    // Keep a bounded history to limit memory usage.
    historyRef.current = history.slice(-40);
    historyIndexRef.current = historyRef.current.length - 1;

    setHistoryVersion((v) => v + 1);
  }, []);

  // ----------------------------------------------------------
  // Load the original image and predicted mask.
  // ----------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    initializedRef.current = false;

    async function initialise() {
      setReady(false);
      setError("");
      setMessage("");

      try {
        const image = new Image();

        if (/^https?:\/\//i.test(imageUrl)) {
          image.crossOrigin = "anonymous";
        }

        image.src = imageUrl;

        await new Promise<void>((resolve, reject) => {
          image.onload = () => resolve();
          image.onerror = () =>
            reject(
              new Error(
                "Could not load the ultrasound image. Check the image URL."
              )
            );
        });

        const mask = new Image();
        mask.src = maskUrl;

        await new Promise<void>((resolve, reject) => {
          mask.onload = () => resolve();
          mask.onerror = () =>
            reject(
              new Error(
                "Could not load the AI-predicted mask."
              )
            );
        });

        if (cancelled) return;

        const maskCanvas = maskCanvasRef.current;
        if (!maskCanvas) return;

        const width = image.naturalWidth;
        const height = image.naturalHeight;

        maskCanvas.width = width;
        maskCanvas.height = height;

        const maskCtx = maskCanvas.getContext("2d");
        if (!maskCtx) {
          throw new Error("Could not initialise the mask canvas.");
        }

        maskCtx.clearRect(0, 0, width, height);
        maskCtx.drawImage(mask, 0, 0, width, height);

        // Convert the prediction to a strict binary mask.
        const pixels = maskCtx.getImageData(
          0,
          0,
          width,
          height
        );

        for (let i = 0; i < pixels.data.length; i += 4) {
          const value = pixels.data[i] > 127 ? 255 : 0;

          pixels.data[i] = value;
          pixels.data[i + 1] = value;
          pixels.data[i + 2] = value;
          pixels.data[i + 3] = 255;
        }

        maskCtx.putImageData(pixels, 0, 0);

        imageRef.current = image;

        const initialSnapshot = maskCanvas.toDataURL("image/png");

        historyRef.current = [initialSnapshot];
        historyIndexRef.current = 0;

        initializedRef.current = true;

        renderCanvas();
        setReady(true);
        setHistoryVersion((v) => v + 1);
      } catch (e) {
        if (!cancelled) {
          setError(
            e instanceof Error
              ? e.message
              : "Could not initialise the editor."
          );
        }
      }
    }

    void initialise();

    return () => {
      cancelled = true;
    };
  }, [imageUrl, maskUrl, renderCanvas]);

  // Redraw the overlay when its visibility changes.
  useEffect(() => {
    if (ready) renderCanvas();
  }, [ready, renderCanvas]);

  // ----------------------------------------------------------
  // Restore an earlier mask snapshot.
  // ----------------------------------------------------------
  const applyHistory = useCallback(
    async (index: number) => {
      const snapshot = historyRef.current[index];
      const maskCanvas = maskCanvasRef.current;

      if (!snapshot || !maskCanvas) return;

      const restoredImage = new Image();
      restoredImage.src = snapshot;

      await new Promise<void>((resolve, reject) => {
        restoredImage.onload = () => resolve();
        restoredImage.onerror = () =>
          reject(new Error("Could not restore mask history."));
      });

      const ctx = maskCanvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(
        0,
        0,
        maskCanvas.width,
        maskCanvas.height
      );

      ctx.drawImage(
        restoredImage,
        0,
        0,
        maskCanvas.width,
        maskCanvas.height
      );

      historyIndexRef.current = index;

      renderCanvas();
      setHistoryVersion((v) => v + 1);
      setMessage("");
    },
    [renderCanvas]
  );

  const undo = () => {
    if (historyIndexRef.current > 0) {
      void applyHistory(historyIndexRef.current - 1);
    }
  };

  const redo = () => {
    if (
      historyIndexRef.current <
      historyRef.current.length - 1
    ) {
      void applyHistory(historyIndexRef.current + 1);
    }
  };

  // ----------------------------------------------------------
  // Convert pointer coordinates to original image coordinates.
  // ----------------------------------------------------------
  const getImagePoint = (
    event: React.PointerEvent<HTMLCanvasElement>
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();

    if (!rect.width || !rect.height) return null;

    return {
      x: Math.max(
        0,
        Math.min(
          canvas.width - 1,
          ((event.clientX - rect.left) / rect.width) *
            canvas.width
        )
      ),
      y: Math.max(
        0,
        Math.min(
          canvas.height - 1,
          ((event.clientY - rect.top) / rect.height) *
            canvas.height
        )
      ),
    };
  };

  // ----------------------------------------------------------
  // Draw a continuous stroke in original image coordinates.
  // ----------------------------------------------------------
  const paint = (
    event: React.PointerEvent<HTMLCanvasElement>
  ) => {
    const maskCanvas = maskCanvasRef.current;
    const point = getImagePoint(event);

    if (!maskCanvas || !point) return;

    const ctx = maskCanvas.getContext("2d");
    if (!ctx) return;

    const previous = lastPointRef.current;

    ctx.save();
    ctx.fillStyle =
      tool === "brush" ? "#ffffff" : "#000000";
    ctx.strokeStyle =
      tool === "brush" ? "#ffffff" : "#000000";
    ctx.lineWidth = brushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (previous) {
      ctx.beginPath();
      ctx.moveTo(previous.x, previous.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(
        point.x,
        point.y,
        brushSize / 2,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }

    ctx.restore();

    lastPointRef.current = point;
    renderCanvas();
  };

  const startDrawing = (
    event: React.PointerEvent<HTMLCanvasElement>
  ) => {
    if (!ready || tool === "pan") return;

    drawingRef.current = true;
    lastPointRef.current = null;

    event.currentTarget.setPointerCapture(event.pointerId);
    paint(event);
  };

  const moveDrawing = (
    event: React.PointerEvent<HTMLCanvasElement>
  ) => {
    if (drawingRef.current) paint(event);
  };

  const stopDrawing = () => {
    if (!drawingRef.current) return;

    drawingRef.current = false;
    lastPointRef.current = null;

    captureMask();
  };

  // ----------------------------------------------------------
  // Reset to the initial prediction, without changing it.
  // ----------------------------------------------------------
  const resetMask = async () => {
    if (!historyRef.current.length) return;

    await applyHistory(0);

    historyRef.current = [historyRef.current[0]];
    historyIndexRef.current = 0;

    setHistoryVersion((v) => v + 1);
    setMessage(
      "Restored the original AI prediction. No saved annotation was changed."
    );
  };

  // ----------------------------------------------------------
  // Download current binary mask.
  // ----------------------------------------------------------
  const downloadMask = () => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;

    const link = document.createElement("a");

    link.href = maskCanvas.toDataURL("image/png");
    link.download = `reviewed-${filename.replace(
      /\.[^.]+$/,
      ""
    )}.png`;

    link.click();
  };

  // ----------------------------------------------------------
  // Save the reviewed mask separately through FastAPI.
  // ----------------------------------------------------------
  const saveMask = async () => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;

    setSaving(true);
    setError("");
    setMessage("");

    try {
      const response = await fetch(
        `${API}/api/reviewed-masks`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            filename,
            experiment_id: experimentId ?? null,
            width: maskCanvas.width,
            height: maskCanvas.height,
            mask_png_base64: maskCanvas
              .toDataURL("image/png")
              .split(",")[1],
          }),
        }
      );

      const result = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          result?.detail ||
            `Could not save annotation (${response.status}).`
        );
      }

      setMessage(
        `Reviewed annotation saved. Review ID: ${result.review_id}`
      );
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not save annotation. Check the backend."
      );
    } finally {
      setSaving(false);
    }
  };

  const canUndo =
    historyIndexRef.current > 0;

  const canRedo =
    historyIndexRef.current <
    historyRef.current.length - 1;

  // historyVersion triggers updates when the undo/redo stack changes.
  void historyVersion;

  return (
    <section className="mt-6 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            Interactive Mask Correction
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Edit the tumor mask directly over the ultrasound image.
          </p>
        </div>

        <span className="rounded-full bg-teal-50 px-3 py-1 text-xs font-medium text-teal-800">
          {ready ? "Editor ready" : "Loading editor"}
        </span>
      </div>

      {/* Toolbar */}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setTool("brush")}
          disabled={!ready}
          className={`inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium ${
            tool === "brush"
              ? "bg-teal-700 text-white"
              : "border border-slate-200 text-slate-700 hover:bg-slate-50"
          }`}
        >
          <Brush size={16} />
          Brush
        </button>

        <button
          type="button"
          onClick={() => setTool("eraser")}
          disabled={!ready}
          className={`inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium ${
            tool === "eraser"
              ? "bg-rose-600 text-white"
              : "border border-slate-200 text-slate-700 hover:bg-slate-50"
          }`}
        >
          <Eraser size={16} />
          Eraser
        </button>

        <button
          type="button"
          onClick={() => setTool("pan")}
          disabled={!ready}
          className={`inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium ${
            tool === "pan"
              ? "bg-slate-800 text-white"
              : "border border-slate-200 text-slate-700 hover:bg-slate-50"
          }`}
          title="Disable painting"
        >
          <Move size={16} />
          Inspect
        </button>

        <label className="flex min-w-48 flex-1 items-center gap-3 rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-600">
          Brush size

          <input
            type="range"
            min={2}
            max={60}
            value={brushSize}
            onChange={(event) =>
              setBrushSize(Number(event.target.value))
            }
            className="min-w-12 flex-1 accent-teal-700"
          />

          <span className="w-6 text-right tabular-nums">
            {brushSize}
          </span>
        </label>

        <button
          type="button"
          onClick={undo}
          disabled={!canUndo}
          className="inline-flex items-center gap-1 rounded-xl border border-slate-200 px-3 py-2 text-sm hover:bg-slate-50 disabled:opacity-40"
        >
          <Undo2 size={16} />
          Undo
        </button>

        <button
          type="button"
          onClick={redo}
          disabled={!canRedo}
          className="inline-flex items-center gap-1 rounded-xl border border-slate-200 px-3 py-2 text-sm hover:bg-slate-50 disabled:opacity-40"
        >
          <Redo2 size={16} />
          Redo
        </button>

        <button
          type="button"
          onClick={resetMask}
          disabled={!ready}
          className="inline-flex items-center gap-1 rounded-xl border border-slate-200 px-3 py-2 text-sm hover:bg-slate-50 disabled:opacity-40"
        >
          <RotateCcw size={16} />
          Reset
        </button>
      </div>

      {/* View controls */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => setShowOverlay((v) => !v)}
          disabled={!ready}
          className="inline-flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-slate-600 hover:bg-slate-100"
        >
          {showOverlay ? (
            <Eye size={16} />
          ) : (
            <ImageIcon size={16} />
          )}
          {showOverlay ? "Hide mask overlay" : "Show mask overlay"}
        </button>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setZoom((v) => Math.max(1, v - 0.25))}
            disabled={zoom <= 1}
            className="rounded-lg border border-slate-200 p-2 disabled:opacity-40"
            title="Zoom out"
          >
            <ZoomOut size={16} />
          </button>

          <span className="w-12 text-center text-xs tabular-nums text-slate-600">
            {Math.round(zoom * 100)}%
          </span>

          <button
            type="button"
            onClick={() => setZoom((v) => Math.min(3, v + 0.25))}
            disabled={zoom >= 3}
            className="rounded-lg border border-slate-200 p-2 disabled:opacity-40"
            title="Zoom in"
          >
            <ZoomIn size={16} />
          </button>
        </div>
      </div>

      {/* Main canvas: ultrasound background + red mask */}
      <div className="mt-4 overflow-auto rounded-2xl border border-slate-700 bg-slate-950 p-3">
        {ready ? (
          <canvas
            ref={canvasRef}
            onPointerDown={startDrawing}
            onPointerMove={moveDrawing}
            onPointerUp={stopDrawing}
            onPointerCancel={stopDrawing}
            onLostPointerCapture={stopDrawing}
            className={`mx-auto block max-w-full touch-none ${
              tool === "brush"
                ? "cursor-crosshair"
                : tool === "eraser"
                  ? "cursor-cell"
                  : "cursor-default"
            }`}
            style={{
              width: `${zoom * 100}%`,
              height: "auto",
              maxWidth: zoom === 1 ? "100%" : "none",
            }}
            aria-label="Ultrasound image with editable tumor segmentation mask"
          />
        ) : (
          <div className="flex min-h-72 items-center justify-center px-5 text-center text-sm text-slate-300">
            {error || "Loading ultrasound and predicted mask…"}
          </div>
        )}
      </div>

      {/* Hidden binary mask canvas */}
      <canvas
        ref={maskCanvasRef}
        className="hidden"
      />

      {/* Legend */}
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-slate-500">
        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-sm bg-rose-500/80" />
          Predicted / edited tumor region
        </span>

        <span className="inline-flex items-center gap-2">
          <span className="h-3 w-3 rounded-sm border border-slate-300 bg-white" />
          Visible ultrasound background
        </span>
      </div>

      {/* Save and download */}
      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={saveMask}
          disabled={!ready || saving}
          className="inline-flex items-center gap-2 rounded-xl bg-teal-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-teal-800 disabled:opacity-50"
        >
          {saving ? (
            <LoaderCircle size={16} className="animate-spin" />
          ) : (
            <Save size={16} />
          )}
          {saving ? "Saving…" : "Save reviewed annotation"}
        </button>

        <button
          type="button"
          onClick={downloadMask}
          disabled={!ready}
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          <Download size={16} />
          Download mask PNG
        </button>
      </div>

      {message && (
        <p
          role="status"
          className="mt-3 break-words text-sm text-emerald-700"
        >
          {message}
        </p>
      )}

      {error && (
        <p
          role="alert"
          className="mt-3 break-words text-sm text-rose-700"
        >
          {error}
        </p>
      )}

      <p className="mt-3 text-xs leading-5 text-slate-500">
        The reviewed mask is stored separately from the original AI
        prediction. Brush adds foreground pixels; eraser removes
        them. Validate corrections against expert annotations before
        using them for research conclusions.
      </p>
    </section>
  );
}
