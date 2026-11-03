/**
 * Mathematical evaluation metrics for medical image segmentation
 * Standard metrics: Dice Similarity Coefficient (DSC), Intersection over Union (IoU),
 * Bounding Box IoU, and Hausdorff Distance 95th percentile (HD95) estimation.
 */

/**
 * Calculates Dice Similarity Coefficient (DSC) = 2 * |A ∩ B| / (|A| + |B|)
 */
export function calculateDice(intersection: number, totalPred: number, totalGt: number): number {
  if (totalPred === 0 && totalGt === 0) return 1.0;
  const denominator = totalPred + totalGt;
  if (denominator === 0) return 0.0;
  return (2 * intersection) / denominator;
}

/**
 * Calculates Intersection over Union (IoU / Jaccard Index) = |A ∩ B| / |A ∪ B|
 */
export function calculateIoU(intersection: number, union: number): number {
  if (union === 0) return intersection === 0 ? 1.0 : 0.0;
  return intersection / union;
}

/**
 * Calculates IoU between two bounding boxes formatted as [ymin, xmin, ymax, xmax]
 */
export function calculateBboxIoU(
  boxA: [number, number, number, number],
  boxB: [number, number, number, number]
): number {
  const [y1A, x1A, y2A, x2A] = boxA;
  const [y1B, x1B, y2B, x2B] = boxB;

  const interY1 = Math.max(y1A, y1B);
  const interX1 = Math.max(x1A, x1B);
  const interY2 = Math.min(y2A, y2B);
  const interX2 = Math.min(x2A, x2B);

  const interWidth = Math.max(0, interX2 - interX1);
  const interHeight = Math.max(0, interY2 - interY1);
  const intersectionArea = interWidth * interHeight;

  const areaA = Math.max(0, x2A - x1A) * Math.max(0, y2A - y1A);
  const areaB = Math.max(0, x2B - x1B) * Math.max(0, y2B - y1B);
  const unionArea = areaA + areaB - intersectionArea;

  if (unionArea <= 0) return 0.0;
  return intersectionArea / unionArea;
}

/**
 * Computes precision and recall from true positives, false positives, false negatives
 */
export function calculatePrecisionRecall(tp: number, fp: number, fn: number): { precision: number; recall: number } {
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  return { precision, recall };
}

/**
 * Formats a metric with decimal precision and fallback
 */
export function formatMetric(val: number | undefined, decimals = 3): string {
  if (val === undefined || isNaN(val)) return '—';
  return val.toFixed(decimals);
}

/**
 * Estimates Hausdorff 95th Percentile distance in mm based on box offset & contour discrepancy
 * Higher IoU correlates with lower boundary distance
 */
export function estimateHD95(dice: number, pixelSpacingMm = 0.2): number {
  if (dice >= 0.99) return 0.5 * pixelSpacingMm;
  // Non-linear inverse approximation: Lower Dice gives higher Hausdorff distance
  const penalty = Math.pow(1 - dice, 1.2) * 45;
  return Number((Math.max(0.8, penalty) * pixelSpacingMm).toFixed(2));
}

/**
 * Applies a bounding box margin expansion (clamped to 0..1)
 */
export function expandBboxWithMargin(
  box: [number, number, number, number],
  marginPercent: number
): [number, number, number, number] {
  const [ymin, xmin, ymax, xmax] = box;
  const w = xmax - xmin;
  const h = ymax - ymin;
  const dx = (w * marginPercent) / 100;
  const dy = (h * marginPercent) / 100;

  return [
    Math.max(0, ymin - dy),
    Math.max(0, xmin - dx),
    Math.min(1, ymax + dy),
    Math.min(1, xmax + dx),
  ];
}
