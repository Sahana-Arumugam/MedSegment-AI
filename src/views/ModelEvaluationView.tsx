
import React, { useState } from 'react';
import {
  BarChart3,
  CheckCircle2,
  Info,
  TrendingUp,
  FileSpreadsheet,
  Image as ImageIcon,
} from 'lucide-react';

type MetricCardProps = {
  title: string;
  value: string;
  description: string;
};

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  description,
}) => (
  <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
    <p className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
      {title}
    </p>
    <p className="mt-2 text-3xl font-bold font-mono text-slate-900">
      {value}
    </p>
    <p className="mt-2 text-xs text-slate-500">{description}</p>
  </div>
);

export const ModelEvaluationView: React.FC = () => {
  const [selectedSection, setSelectedSection] = useState<
    'segmentation' | 'prompt'
  >('segmentation');

  const segmentationMetrics = [
    {
      title: 'Mean Dice',
      value: '0.5906',
      description: 'Overlap between predicted and ground-truth masks',
    },
    {
      title: 'Mean Segmentation IoU',
      value: '0.4618',
      description: 'Intersection over Union of the masks',
    },
    {
      title: 'Mean Precision',
      value: '0.6526',
      description: 'Fraction of predicted foreground that is correct',
    },
    {
      title: 'Mean Recall',
      value: '0.6135',
      description: 'Fraction of ground-truth foreground recovered',
    },
    {
      title: 'Mean Box IoU',
      value: '0.4744',
      description: 'Predicted boxes compared with ground-truth mask bounds',
    },
    {
      title: 'Mean MedSAM Score',
      value: '0.5614',
      description: 'Mean score returned during MedSAM inference',
    },
  ];

  const promptComparison = [
    {
      model: 'AutoPromptCNN',
      iou: 0.2678,
      description: 'Original bounding-box prediction model',
    },
    {
      model: 'AutoPromptCNNV2',
      iou: 0.4721,
      description: 'Improved spatial-feature architecture',
    },
  ];

  const improvement = ((0.4721 - 0.2678) / 0.2678) * 100;

  return (
    <div className="max-w-7xl mx-auto pb-12 space-y-6">
      <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono font-semibold text-teal-700 uppercase tracking-wider">
              <BarChart3 className="w-4 h-4" />
              Experimental Results
            </div>

            <h1 className="text-2xl font-bold text-slate-900 mt-2">
              Model Evaluation on BUSI
            </h1>

            <p className="text-sm text-slate-500 mt-2 max-w-2xl">
              Evaluation of automatic bounding-box prediction using
              AutoPromptCNNV2 followed by MedSAM ViT-B segmentation.
            </p>
          </div>

          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-teal-50 text-teal-800 border border-teal-100 text-xs">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            Evaluation completed: 99 test images
          </div>
        </div>

        <div className="mt-5 p-3 rounded-lg bg-amber-50 border border-amber-200 flex gap-2">
          <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-900">
            These are experimental research results, not clinical validation.
            Segmentation quality varies across images.
          </p>
        </div>
      </section>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {segmentationMetrics.map((metric) => (
          <MetricCard key={metric.title} {...metric} />
        ))}
      </div>

      <section className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-5 border-b border-slate-200">
          <h2 className="font-bold text-slate-900">
            Automatic Bounding-Box Model Comparison
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Comparison of bounding-box prediction performance on the test set.
          </p>
        </div>

        <div className="p-5 space-y-6">
          {promptComparison.map((item) => (
            <div key={item.model}>
              <div className="flex justify-between items-center gap-3 mb-2">
                <div>
                  <p className="font-semibold text-sm text-slate-800">
                    {item.model}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    {item.description}
                  </p>
                </div>

                <span className="font-mono font-bold text-slate-900">
                  {item.iou.toFixed(4)}
                </span>
              </div>

              <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    item.model === 'AutoPromptCNNV2'
                      ? 'bg-teal-600'
                      : 'bg-slate-400'
                  }`}
                  style={{ width: `${item.iou * 100}%` }}
                />
              </div>
            </div>
          ))}

          <div className="p-4 rounded-lg bg-teal-50 border border-teal-100">
            <div className="flex items-center gap-2 text-teal-800 font-semibold text-sm">
              <TrendingUp className="w-4 h-4" />
              Relative improvement in mean box IoU
            </div>

            <p className="text-2xl font-bold text-teal-800 mt-2">
              {improvement.toFixed(1)}%
            </p>

            <p className="text-xs text-teal-900 mt-1">
              AutoPromptCNNV2 compared with AutoPromptCNN. This improvement
              refers to bounding-box IoU, not segmentation Dice.
            </p>
          </div>
        </div>
      </section>

      <section className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-5 border-b border-slate-200">
          <h2 className="font-bold text-slate-900">Evaluation Details</h2>
          <p className="text-xs text-slate-500 mt-1">
            Test configuration and saved evaluation artifacts.
          </p>
        </div>

        <div className="p-5 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-lg border border-slate-200 p-4">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className="w-5 h-5 text-teal-600" />
              <h3 className="font-semibold text-sm text-slate-800">
                Test Configuration
              </h3>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between gap-3">
                <span className="text-slate-500">Dataset</span>
                <span className="font-medium text-slate-800">BUSI</span>
              </div>

              <div className="flex justify-between gap-3">
                <span className="text-slate-500">Annotated test images</span>
                <span className="font-medium text-slate-800">99</span>
              </div>

              <div className="flex justify-between gap-3">
                <span className="text-slate-500">Execution device</span>
                <span className="font-medium text-slate-800">CPU</span>
              </div>

              <div className="flex justify-between gap-3">
                <span className="text-slate-500">Evaluation time</span>
                <span className="font-medium text-slate-800">18.9 minutes</span>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 p-4">
            <div className="flex items-center gap-2 mb-3">
              <FileSpreadsheet className="w-5 h-5 text-indigo-600" />
              <h3 className="font-semibold text-sm text-slate-800">
                Saved Results
              </h3>
            </div>

            <p className="text-xs text-slate-500 mb-3">
              Generated by the segmentation evaluation script.
            </p>

            <p className="text-xs font-mono text-slate-700 break-all">
              backend/segmentation_evaluation/test_segmentation_metrics.csv
            </p>

            <div className="flex items-center gap-2 mt-4">
              <ImageIcon className="w-4 h-4 text-indigo-600 shrink-0" />
              <p className="text-xs font-mono text-slate-700 break-all">
                backend/segmentation_evaluation/visual_examples/
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
        <h2 className="font-bold text-slate-900">Metric Interpretation</h2>

        <div className="flex flex-wrap gap-2 mt-4 border-b border-slate-200 pb-3">
          <button
            onClick={() => setSelectedSection('segmentation')}
            className={`px-3 py-2 rounded-lg text-sm font-medium ${
              selectedSection === 'segmentation'
                ? 'bg-teal-50 text-teal-800'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            Segmentation Metrics
          </button>

          <button
            onClick={() => setSelectedSection('prompt')}
            className={`px-3 py-2 rounded-lg text-sm font-medium ${
              selectedSection === 'prompt'
                ? 'bg-teal-50 text-teal-800'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            Bounding-Box Metrics
          </button>
        </div>

        {selectedSection === 'segmentation' ? (
          <div className="space-y-3 mt-4 text-sm text-slate-600">
            <p>
              <strong>Dice coefficient:</strong> Measures overlap between
              predicted and ground-truth masks. Higher is better.
            </p>
            <p>
              <strong>Segmentation IoU:</strong> Measures intersection divided
              by union of the predicted and ground-truth regions. Higher is better.
            </p>
            <p>
              <strong>Precision:</strong> Measures how much of the predicted
              foreground is correct.
            </p>
            <p>
              <strong>Recall:</strong> Measures how much of the ground-truth
              foreground is recovered.
            </p>
          </div>
        ) : (
          <div className="space-y-3 mt-4 text-sm text-slate-600">
            <p>
              <strong>Bounding-box IoU:</strong> Measures overlap between the
              predicted box and the reference box derived from the ground-truth mask.
            </p>
            <p>
              <strong>Automatic prompting:</strong> AutoPromptCNNV2 predicts a
              box that is supplied to MedSAM, avoiding manual box drawing in this workflow.
            </p>
            <p>
              <strong>Limitation:</strong> An inaccurate predicted box can
              affect the final segmentation.
            </p>
          </div>
        )}
      </section>

      <p className="text-xs text-slate-400 text-center">
        MedSegment AI · Experimental research prototype · Not intended for clinical diagnosis
      </p>
    </div>
  );
};

export default ModelEvaluationView;
