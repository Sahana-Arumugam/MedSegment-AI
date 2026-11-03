export type DatasetCategory = 'benign' | 'malignant' | 'normal';
export type DatasetSplit = 'train' | 'val' | 'test';


export interface BusiImage {
  id: string;
  filename: string;
  category: DatasetCategory;
  width: number;
  height: number;
  split: DatasetSplit;
  hasMask: boolean;
  maskCount: number;

  // Real BUSI files served by the Python backend
  imageUrl?: string;
  maskUrls?: string[];

  isDuplicateCandidate?: boolean;
  notes?: string;

  lesionDetails?: {
    shape: 'oval' | 'round' | 'irregular';
    margin: 'circumscribed' | 'microlobulated' | 'spiculated' | 'ill-defined';
    echogenicity: 'hypoechoic' | 'isoechoic' | 'complex';
    posteriorFeatures: 'enhancement' | 'shadowing' | 'none';
  };

  groundTruthBbox?: [number, number, number, number];
  sonogramSeed: number;
}


export type ModelArchitecture = 
  | 'medsam_baseline'
  | 'medsam_autoprompt'
  | 'medsam_lora'
  | 'full_finetune';

export type PromptMode = 'auto_bbox' | 'reference_bbox' | 'manual_bbox' | 'point_prompts';

export interface InferenceConfig {
  model: ModelArchitecture;
  promptMode: PromptMode;
  loraRank: number; // 4, 8, 16, 32
  boxMarginPercent: number; // 0% to 20%
  confidenceThreshold: number; // 0.0 to 1.0
  nmsThreshold: number;
}

export interface SegmentationResult {
  id: string;
  imageId: string;
  model: ModelArchitecture;
  promptMode: PromptMode;
  inferenceDurationMs: number;
  metrics: {
    dice: number;
    iou: number;
    hausdorffDistance95: number;
    precision: number;
    recall: number;
  };
  predictedBbox: [number, number, number, number]; // [ymin, xmin, ymax, xmax] 0..1
  promptConfidence: number;
  completedAt: string;
  isDemoSimulation: boolean;
}

export interface Experiment {
  id: string;
  name: string;
  modelArchitecture: ModelArchitecture;
  promptMode: PromptMode;
  datasetSplit: 'busi_70_30' | 'busi_80_20' | 'busi_5_fold';
  adaptationMethod: 'zero_shot' | 'lora' | 'full_finetune' | 'active_learning';
  labeledImagesCount: number;
  trainableParamsMillion: number;
  totalParamsMillion: number;
  status: 'Completed' | 'Running' | 'Planned' | 'Failed';
  diceScore: number;
  iouScore: number;
  hd95: number;
  trainingTimeMin: number;
  inferenceLatencyMs: number;
  date: string;
  description: string;
}

export interface MetricDataPoint {
  labeledCount: number;
  medsamZeroShot: number;
  medsamAutoPrompt: number;
  medsamLoraAutoPrompt: number;
  fullFineTuning: number;
}
