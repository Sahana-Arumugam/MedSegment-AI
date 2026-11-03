
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Crosshair,
  Layers,
  Image as ImageIcon,
  AlertCircle,
} from 'lucide-react';
import { BusiImage, PromptMode } from '../types';

const API_BASE = 'http://127.0.0.1:8000';

interface SonogramCanvasProps {
  image: BusiImage | null;
  uploadedImageUrl: string | null;
  activeTab: 'original' | 'ground_truth' | 'predicted' | 'overlay';
  showBoundingBox: boolean;
  predictedBbox?: [number, number, number, number];
  promptConfidence?: number;
  promptMode: PromptMode;
  contrast: number;
  brightness: number;
  onExportMask?: () => void;
  onExportComposite?: () => void;
  predictedMaskUrl?: string | null;
}

function resolveAssetUrl(url?: string | null): string | null {
  if (!url) return null;

  if (/^(https?:|data:|blob:)/i.test(url)) {
    return url;
  }

  return `${API_BASE}${url.startsWith('/') ? url : `/${url}`}`;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Unable to load the image.'));

    img.src = src;
  });
}

async function drawMask(
  ctx: CanvasRenderingContext2D,
  maskUrl: string,
  width: number,
  height: number,
  color: [number, number, number],
  opacity: number,
): Promise<void> {
  const maskImage = await loadImage(maskUrl);

  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = width;
  maskCanvas.height = height;

  const maskContext = maskCanvas.getContext('2d', {
    willReadFrequently: true,
  });

  if (!maskContext) {
    throw new Error('Unable to process the mask image.');
  }

  maskContext.drawImage(maskImage, 0, 0, width, height);

  const pixels = maskContext.getImageData(0, 0, width, height);
  const output = ctx.createImageData(width, height);

  for (let i = 0; i < pixels.data.length; i += 4) {
    const luminance =
      pixels.data[i] * 0.299 +
      pixels.data[i + 1] * 0.587 +
      pixels.data[i + 2] * 0.114;

    const isForeground =
      pixels.data[i + 3] > 0 && luminance > 127;

    if (isForeground) {
      output.data[i] = color[0];
      output.data[i + 1] = color[1];
      output.data[i + 2] = color[2];
      output.data[i + 3] = Math.round(opacity * 255);
    }
  }

  const overlayCanvas = document.createElement('canvas');
  overlayCanvas.width = width;
  overlayCanvas.height = height;

  const overlayContext = overlayCanvas.getContext('2d');

  if (!overlayContext) {
    throw new Error('Unable to render the mask overlay.');
  }

  overlayContext.putImageData(output, 0, 0);
  ctx.drawImage(overlayCanvas, 0, 0);
}

export const SonogramCanvas: React.FC<SonogramCanvasProps> = ({
  image,
  uploadedImageUrl,
  activeTab,
  showBoundingBox,
  predictedBbox,
  promptConfidence,
  contrast,
  brightness,
  predictedMaskUrl,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const dragStartRef = useRef({ x: 0, y: 0 });

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [crosshairPos, setCrosshairPos] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [showCrosshairs, setShowCrosshairs] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.55);
  const [renderError, setRenderError] = useState<string | null>(null);

  const imageUrl = resolveAssetUrl(
    uploadedImageUrl || image?.imageUrl,
  );

  const groundTruthMaskUrl = resolveAssetUrl(
    image?.maskUrls?.[0],
  );

  const resolvedPredictionUrl = resolveAssetUrl(predictedMaskUrl);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setCrosshairPos(null);
  }, []);

  const handleZoomIn = () => {
    setZoom((previous) => Math.min(previous * 1.25, 5));
  };

  const handleZoomOut = () => {
    setZoom((previous) => Math.max(previous / 1.25, 0.4));
  };

  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    event.preventDefault();

    const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;

    setZoom((previous) =>
      Math.min(Math.max(previous * factor, 0.4), 5),
    );
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;

    setIsDragging(true);
    dragStartRef.current = {
      x: event.clientX - pan.x,
      y: event.clientY - pan.y,
    };
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    const canvas = canvasRef.current;

    if (canvas && showCrosshairs) {
      const rect = canvas.getBoundingClientRect();

      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      setCrosshairPos({
        x: Math.round((event.clientX - rect.left) * scaleX),
        y: Math.round((event.clientY - rect.top) * scaleY),
      });
    }

    if (isDragging) {
      setPan({
        x: event.clientX - dragStartRef.current.x,
        y: event.clientY - dragStartRef.current.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    let cancelled = false;

    const renderCanvas = async () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      setRenderError(null);

      if (!imageUrl) {
        canvas.width = 640;
        canvas.height = 440;

        ctx.fillStyle = '#080d14';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = '#253746';
        ctx.setLineDash([6, 6]);
        ctx.strokeRect(24, 24, canvas.width - 48, canvas.height - 48);
        ctx.setLineDash([]);

        ctx.fillStyle = '#94a3b8';
        ctx.textAlign = 'center';
        ctx.font = '14px sans-serif';
        ctx.fillText(
          'Select a dataset image or upload an image to begin',
          canvas.width / 2,
          canvas.height / 2,
        );

        ctx.fillStyle = '#64748b';
        ctx.font = '11px sans-serif';
        ctx.fillText(
          'No image is currently available',
          canvas.width / 2,
          canvas.height / 2 + 24,
        );

        return;
      }

      try {
        const sourceImage = await loadImage(imageUrl);

        if (cancelled) return;

        const width = sourceImage.naturalWidth;
        const height = sourceImage.naturalHeight;

        canvas.width = width;
        canvas.height = height;

        ctx.clearRect(0, 0, width, height);
        ctx.filter = `contrast(${contrast}%) brightness(${100 + brightness}%)`;
        ctx.drawImage(sourceImage, 0, 0, width, height);
        ctx.filter = 'none';

        if (
          (activeTab === 'ground_truth' || activeTab === 'overlay') &&
          groundTruthMaskUrl
        ) {
          await drawMask(
            ctx,
            groundTruthMaskUrl,
            width,
            height,
            [16, 185, 129],
            maskOpacity,
          );

          if (cancelled) return;
        }

        if (
          (activeTab === 'predicted' || activeTab === 'overlay') &&
          resolvedPredictionUrl
        ) {
          await drawMask(
            ctx,
            resolvedPredictionUrl,
            width,
            height,
            [6, 182, 212],
            maskOpacity,
          );

          if (cancelled) return;
        }

        if (
          showBoundingBox &&
          predictedBbox &&
          predictedBbox.every(Number.isFinite)
        ) {
          const [ymin, xmin, ymax, xmax] = predictedBbox;

          // Bounding-box coordinates are normalized to the image dimensions.
          const x1 = xmin * width;
          const y1 = ymin * height;
          const x2 = xmax * width;
          const y2 = ymax * height;

          if (x2 > x1 && y2 > y1) {
            ctx.save();

            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = Math.max(2, width / 400);
            ctx.setLineDash([8, 5]);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.setLineDash([]);

            const corners = [
              [x1, y1, 1, 1],
              [x2, y1, -1, 1],
              [x1, y2, 1, -1],
              [x2, y2, -1, -1],
            ];

            const cornerLength = Math.min(
              18,
              Math.max(8, Math.min(x2 - x1, y2 - y1) * 0.12),
            );

            for (const [x, y, dx, dy] of corners) {
              ctx.beginPath();
              ctx.moveTo(x, y + dy * cornerLength);
              ctx.lineTo(x, y);
              ctx.lineTo(x + dx * cornerLength, y);
              ctx.stroke();
            }

            if (typeof promptConfidence === 'number') {
              const label = `Prompt confidence: ${(promptConfidence * 100).toFixed(1)}%`;

              ctx.font = '12px monospace';

              const labelWidth = ctx.measureText(label).width + 16;
              const labelY = Math.max(0, y1 - 28);

              ctx.fillStyle = '#f59e0b';
              ctx.fillRect(x1, labelY, labelWidth, 24);

              ctx.fillStyle = '#111827';
              ctx.fillText(label, x1 + 8, labelY + 16);
            }

            ctx.restore();
          }
        }
      } catch (error) {
        if (cancelled) return;

        const message =
          error instanceof Error
            ? error.message
            : 'Unable to render the image.';

        setRenderError(message);

        canvas.width = 640;
        canvas.height = 440;

        ctx.fillStyle = '#080d14';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#fca5a5';
        ctx.textAlign = 'center';
        ctx.font = '14px sans-serif';
        ctx.fillText(
          'Unable to load the selected image',
          canvas.width / 2,
          canvas.height / 2,
        );

        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px sans-serif';
        ctx.fillText(
          'Check the image URL and backend connection.',
          canvas.width / 2,
          canvas.height / 2 + 24,
        );
      }
    };

    void renderCanvas();

    return () => {
      cancelled = true;
    };
  }, [
    imageUrl,
    groundTruthMaskUrl,
    resolvedPredictionUrl,
    activeTab,
    showBoundingBox,
    predictedBbox,
    promptConfidence,
    contrast,
    brightness,
    maskOpacity,
  ]);

  const currentLayerLabel = {
    original: 'Original image',
    ground_truth: 'Ground-truth mask',
    predicted: 'Predicted mask',
    overlay: 'Mask comparison',
  }[activeTab];

  const groundTruthAvailable = Boolean(groundTruthMaskUrl);
  const predictionAvailable = Boolean(resolvedPredictionUrl);

  return (
    <div className="flex flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-950 shadow-lg">
      {/* Canvas toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 bg-[#101923] px-3.5 py-3 sm:px-4">
        <div className="flex min-w-0 items-center gap-2.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-500/10 text-teal-400">
            <ImageIcon className="h-4 w-4" />
          </div>

          <div className="min-w-0">
            <p className="truncate text-xs font-semibold text-slate-100">
              {currentLayerLabel}
            </p>
            <p className="mt-0.5 text-[10px] text-slate-500">
              {image
                ? image.filename
                : uploadedImageUrl
                  ? 'Uploaded image'
                  : 'No image selected'}
            </p>
          </div>

          <span className="ml-1 hidden rounded-md border border-slate-700 px-2 py-1 font-mono text-[10px] text-teal-300 sm:inline">
            {Math.round(zoom * 100)}%
          </span>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleZoomIn}
            className="rounded-lg p-2 text-slate-400 transition-colors hover:bg-slate-800 hover:text-white"
            title="Zoom in"
            aria-label="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={handleZoomOut}
            className="rounded-lg p-2 text-slate-400 transition-colors hover:bg-slate-800 hover:text-white"
            title="Zoom out"
            aria-label="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={resetView}
            className="rounded-lg p-2 text-slate-400 transition-colors hover:bg-slate-800 hover:text-white"
            title="Reset view"
            aria-label="Reset view"
          >
            <RotateCcw className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={() => setShowCrosshairs((previous) => !previous)}
            className={`rounded-lg p-2 transition-colors ${
              showCrosshairs
                ? 'bg-teal-950 text-teal-300'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
            }`}
            title="Toggle crosshair coordinates"
            aria-label="Toggle crosshair coordinates"
            aria-pressed={showCrosshairs}
          >
            <Crosshair className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Image stage */}
      <div
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={`relative flex min-h-[360px] max-h-[620px] items-center justify-center overflow-hidden bg-[#080d14] sm:min-h-[460px] ${
          isDragging
            ? 'cursor-grabbing'
            : showCrosshairs
              ? 'cursor-crosshair'
              : 'cursor-grab'
        }`}
      >
        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isDragging
              ? 'none'
              : 'transform 120ms cubic-bezier(0.16, 1, 0.3, 1)',
          }}
          className="relative flex max-h-full max-w-full items-center justify-center"
        >
          <canvas
            ref={canvasRef}
            className="block max-h-[580px] max-w-full object-contain"
          />
        </div>

        {showCrosshairs && crosshairPos && imageUrl && (
          <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg border border-slate-700 bg-slate-900/90 px-3 py-2 font-mono text-[10px] text-slate-300 shadow-lg">
            <span className="text-teal-300">X: {crosshairPos.x}px</span>
            <span className="mx-2 text-slate-600">|</span>
            <span className="text-teal-300">Y: {crosshairPos.y}px</span>
          </div>
        )}

        {/* Layer legend */}
        {imageUrl && (
          <div className="absolute right-3 top-3 max-w-[210px] rounded-xl border border-slate-700/80 bg-slate-900/90 p-3 shadow-lg backdrop-blur">
            <p className="mb-2 text-[9px] font-bold uppercase tracking-[0.15em] text-slate-400">
              Annotation layers
            </p>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" />
                <span className="text-[10px] text-slate-300">
                  Ground truth
                </span>
                <span
                  className={`ml-auto text-[9px] ${
                    groundTruthAvailable ? 'text-emerald-400' : 'text-slate-500'
                  }`}
                >
                  {groundTruthAvailable ? 'Available' : 'Unavailable'}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-cyan-500" />
                <span className="text-[10px] text-slate-300">
                  Prediction
                </span>
                <span
                  className={`ml-auto text-[9px] ${
                    predictionAvailable ? 'text-cyan-400' : 'text-slate-500'
                  }`}
                >
                  {predictionAvailable ? 'Available' : 'Unavailable'}
                </span>
              </div>

              {showBoundingBox && predictedBbox && (
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-sm border border-dashed border-amber-400" />
                  <span className="text-[10px] text-slate-300">
                    Predicted box
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {renderError && (
          <div className="absolute bottom-3 right-3 flex max-w-[280px] items-start gap-2 rounded-lg border border-red-900/60 bg-red-950/90 px-3 py-2 text-[10px] leading-relaxed text-red-200">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{renderError}</span>
          </div>
        )}
      </div>

      {/* Canvas footer */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 bg-[#101923] px-4 py-3">
        <div className="flex items-center gap-2 text-[10px] text-slate-400">
          <Layers className="h-3.5 w-3.5" />
          <span>Mask opacity</span>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={maskOpacity}
            onChange={(event) =>
              setMaskOpacity(Number(event.target.value))
            }
            className="h-1 w-20 cursor-pointer accent-teal-500"
            aria-label="Mask opacity"
          />
          <span className="w-8 text-right font-mono text-slate-300">
            {Math.round(maskOpacity * 100)}%
          </span>
        </div>

        <p className="text-[10px] text-slate-500">
          {image
            ? `${image.width} × ${image.height} px`
            : 'Image dimensions unavailable'}
        </p>
      </div>

      {activeTab === 'ground_truth' && !groundTruthAvailable && (
        <div className="flex items-start gap-2 border-t border-amber-900/40 bg-amber-950/30 px-4 py-3 text-[11px] leading-relaxed text-amber-200">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          No ground-truth mask is available for this image.
        </div>
      )}

      {activeTab === 'predicted' && !predictionAvailable && (
        <div className="flex items-start gap-2 border-t border-slate-800 bg-slate-900 px-4 py-3 text-[11px] leading-relaxed text-slate-400">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          No predicted mask has been provided. Run segmentation to obtain an
          actual model output.
        </div>
      )}

      {activeTab === 'overlay' &&
        (!groundTruthAvailable || !predictionAvailable) && (
          <div className="flex items-start gap-2 border-t border-slate-800 bg-slate-900 px-4 py-3 text-[11px] leading-relaxed text-slate-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            The comparison overlay requires both a ground-truth mask and a
            predicted mask. Only available layers are displayed.
          </div>
        )}
    </div>
  );
};

export default SonogramCanvas;
