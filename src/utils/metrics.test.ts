import test from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateDice,
  calculateIoU,
  calculateBboxIoU,
  calculatePrecisionRecall,
  estimateHD95,
  expandBboxWithMargin,
  formatMetric,
} from './metrics';

test('calculateDice returns 1.0 on identical empty and perfect overlap', () => {
  assert.equal(calculateDice(0, 0, 0), 1.0);
  assert.equal(calculateDice(50, 50, 50), 1.0);
});

test('calculateDice computes standard overlap ratio accurately', () => {
  // Intersection = 40, Pred = 50, GT = 50 -> 2*40 / 100 = 0.8
  assert.equal(calculateDice(40, 50, 50), 0.8);
  // Zero intersection
  assert.equal(calculateDice(0, 50, 50), 0.0);
});

test('calculateIoU returns correct Jaccard index', () => {
  assert.equal(calculateIoU(40, 60), 40 / 60);
  assert.equal(calculateIoU(0, 0), 1.0);
  assert.equal(calculateIoU(0, 10), 0.0);
});

test('calculateBboxIoU computes overlap between bounding boxes accurately', () => {
  // Identical boxes
  const box1: [number, number, number, number] = [0.1, 0.1, 0.5, 0.5];
  assert.equal(calculateBboxIoU(box1, box1), 1.0);

  // Non-overlapping boxes
  const box2: [number, number, number, number] = [0.6, 0.6, 0.9, 0.9];
  assert.equal(calculateBboxIoU(box1, box2), 0.0);

  // Partial overlap
  const boxA: [number, number, number, number] = [0, 0, 2, 2];
  const boxB: [number, number, number, number] = [1, 1, 3, 3];
  // BoxA area = 4, BoxB area = 4. Intersection = [1,1] to [2,2] -> area = 1
  // Union = 4 + 4 - 1 = 7. IoU = 1/7
  const iou = calculateBboxIoU(boxA, boxB);
  assert.ok(Math.abs(iou - 1 / 7) < 0.0001);
});

test('calculatePrecisionRecall handles zero divisions and expected rates', () => {
  const { precision, recall } = calculatePrecisionRecall(80, 20, 10);
  assert.equal(precision, 80 / 100);
  assert.equal(recall, 80 / 90);

  const zeros = calculatePrecisionRecall(0, 0, 0);
  assert.equal(zeros.precision, 0);
  assert.equal(zeros.recall, 0);
});

test('estimateHD95 produces lower distance for higher Dice scores', () => {
  const hdHighDice = estimateHD95(0.92);
  const hdLowDice = estimateHD95(0.60);
  assert.ok(hdHighDice < hdLowDice, 'Higher dice should have smaller HD95 error');
});

test('expandBboxWithMargin expands box without exceeding 0..1 bounds', () => {
  const baseBox: [number, number, number, number] = [0.2, 0.2, 0.6, 0.6];
  // Width = 0.4, Height = 0.4. Margin = 10% -> dx = 0.04, dy = 0.04
  const expanded = expandBboxWithMargin(baseBox, 10);
  assert.ok(Math.abs(expanded[0] - 0.16) < 0.001);
  assert.ok(Math.abs(expanded[1] - 0.16) < 0.001);
  assert.ok(Math.abs(expanded[2] - 0.64) < 0.001);
  assert.ok(Math.abs(expanded[3] - 0.64) < 0.001);

  // Clamped at 0..1
  const edgeBox: [number, number, number, number] = [0.05, 0.05, 0.95, 0.95];
  const clamped = expandBboxWithMargin(edgeBox, 20);
  assert.equal(clamped[0], 0);
  assert.equal(clamped[1], 0);
  assert.equal(clamped[2], 1);
  assert.equal(clamped[3], 1);
});

test('formatMetric formats numbers cleanly', () => {
  assert.equal(formatMetric(0.87654, 3), '0.877');
  assert.equal(formatMetric(undefined), '—');
});
