
import React from 'react';
import {
  BookOpen,
  Layers3,
  Cpu,
  Target,
  ShieldAlert,
  CheckCircle2,
  ArrowDown,
  ArrowRight,
  Microscope,
  FlaskConical,
  GitBranch,
  Activity,
  ScanLine,
  Info,
  AlertTriangle,
} from 'lucide-react';

const SectionHeading = ({
  number,
  title,
  subtitle,
  icon: Icon,
}: {
  number: string;
  title: string;
  subtitle: string;
  icon: React.ElementType;
}) => (
  <div className="flex items-start gap-3">
    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-teal-50 text-teal-700">
      <Icon size={19} />
    </div>
    <div>
      <div className="mb-1 text-[10px] font-bold uppercase tracking-[0.16em] text-teal-700">
        SECTION {number}
      </div>
      <h2 className="text-lg font-bold tracking-tight text-slate-900">
        {title}
      </h2>
      <p className="mt-1 text-sm leading-relaxed text-slate-500">
        {subtitle}
      </p>
    </div>
  </div>
);

const PipelineStep = ({
  number,
  title,
  description,
  tag,
  color,
}: {
  number: string;
  title: string;
  description: string;
  tag: string;
  color: string;
}) => (
  <div className="relative rounded-xl border border-slate-200 bg-white p-4 transition-all duration-200 hover:-translate-y-0.5 hover:border-teal-200 hover:shadow-md">
    <div className="mb-3 flex items-center justify-between">
      <span className="text-[11px] font-bold tracking-widest text-slate-400">
        STEP {number}
      </span>
      <span className={`h-2 w-2 rounded-full ${color}`} />
    </div>
    <h3 className="text-sm font-bold text-slate-900">{title}</h3>
    <p className="mt-2 text-xs leading-relaxed text-slate-500">
      {description}
    </p>
    <div className="mt-4 inline-flex rounded-md bg-slate-50 px-2 py-1 text-[10px] font-semibold text-slate-600">
      {tag}
    </div>
  </div>
);

const MetricCard = ({
  title,
  formula,
  description,
  direction,
}: {
  title: string;
  formula: string;
  description: string;
  direction: string;
}) => (
  <div className="rounded-xl border border-slate-200 bg-white p-5 transition-shadow hover:shadow-sm">
    <div className="mb-3 flex items-center justify-between gap-2">
      <h3 className="text-sm font-bold text-slate-900">{title}</h3>
      <span
        className={`rounded-md px-2 py-1 text-[10px] font-bold ${
          direction === 'Higher is better'
            ? 'bg-emerald-50 text-emerald-700'
            : 'bg-amber-50 text-amber-700'
        }`}
      >
        {direction}
      </span>
    </div>
    <div className="rounded-lg bg-slate-50 px-3 py-3 font-mono text-sm font-semibold text-teal-800">
      {formula}
    </div>
    <p className="mt-3 text-xs leading-relaxed text-slate-500">
      {description}
    </p>
  </div>
);

export const DocumentationView: React.FC = () => {
  return (
    <div className="mx-auto max-w-6xl space-y-7 pb-12">
      {/* Page header */}
      <header className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white">
        <div className="absolute right-0 top-0 h-48 w-48 translate-x-12 -translate-y-16 rounded-full bg-teal-50 opacity-80 blur-2xl" />

        <div className="relative p-6 sm:p-9">
          <div className="mb-5 flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-teal-100 bg-teal-50 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-teal-800">
              <Microscope size={13} />
              Research documentation
            </span>
            <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-[10px] font-semibold text-slate-600">
              Medical computer vision
            </span>
          </div>

          <h1 className="max-w-3xl text-2xl font-bold leading-tight tracking-tight text-slate-950 sm:text-3xl">
            Prompt-Efficient Domain Adaptation of Foundation Segmentation
            Models for Low-Resource Medical Imaging
          </h1>

          <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-600">
            A research framework for automating lesion localization and
            investigating parameter-efficient adaptation of foundation
            segmentation models for breast ultrasound imaging.
          </p>

          <div className="mt-7 grid grid-cols-1 gap-3 border-t border-slate-100 pt-5 sm:grid-cols-3">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                <Layers3 size={17} />
              </div>
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  Dataset
                </p>
                <p className="mt-0.5 text-sm font-semibold text-slate-800">
                  BUSI
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                <Cpu size={17} />
              </div>
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  Foundation model
                </p>
                <p className="mt-0.5 text-sm font-semibold text-slate-800">
                  MedSAM · ViT-B
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                <ScanLine size={17} />
              </div>
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  Research focus
                </p>
                <p className="mt-0.5 text-sm font-semibold text-slate-800">
                  Automatic prompting
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Research-use disclaimer */}
      <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50/70 p-4 sm:p-5">
        <ShieldAlert
          size={20}
          className="mt-0.5 shrink-0 text-amber-700"
        />
        <div>
          <h2 className="text-sm font-bold text-amber-950">
            Research use only
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-amber-900">
            MedSegment AI is an experimental research application. Its
            segmentations and model outputs are not validated for clinical
            diagnosis, treatment decisions, or patient care.
          </p>
        </div>
      </div>

      {/* 01: Problem statement */}
      <section className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5 sm:p-7">
        <SectionHeading
          number="01"
          title="Research problem"
          subtitle="Why automatic prompting and domain adaptation matter for breast ultrasound segmentation."
          icon={Target}
        />

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-slate-200 p-5">
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-orange-50 text-orange-700">
              <Target size={19} />
            </div>
            <h3 className="text-sm font-bold text-slate-900">
              Manual prompt dependency
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-600">
              MedSAM ordinarily uses prompts such as bounding boxes or points.
              Manual prompt selection introduces extra work and can lead to
              variability between users.
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 p-5">
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-sky-50 text-sky-700">
              <Activity size={19} />
            </div>
            <h3 className="text-sm font-bold text-slate-900">
              Ultrasound domain gap
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-600">
              Speckle noise, weak boundaries, acoustic artifacts, and varying
              lesion appearance make breast ultrasound segmentation
              challenging.
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 p-5">
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-violet-50 text-violet-700">
              <FlaskConical size={19} />
            </div>
            <h3 className="text-sm font-bold text-slate-900">
              Limited labeled data
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-600">
              Medical image annotation is expensive. Adaptation strategies
              must be evaluated carefully when only a limited number of
              labeled images are available.
            </p>
          </div>
        </div>

        <div className="rounded-xl bg-slate-50 p-4">
          <p className="text-xs leading-relaxed text-slate-700">
            <strong className="text-slate-900">Research question:</strong>{' '}
            Can a lightweight localization network reduce manual prompting,
            while parameter-efficient adaptation improves the suitability of
            MedSAM for breast ultrasound images?
          </p>
        </div>
      </section>

      {/* 02: Objectives */}
      <section className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5 sm:p-7">
        <SectionHeading
          number="02"
          title="Research objectives"
          subtitle="The main components being investigated in the proposed framework."
          icon={CheckCircle2}
        />

        <div className="space-y-3">
          <div className="flex gap-4 rounded-xl border border-slate-200 p-4">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-teal-50 text-sm font-bold text-teal-800">
              01
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Automatic region-of-interest localization
              </h3>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">
                Develop a lightweight CNN-based prompt generator that
                predicts lesion bounding boxes without requiring a manually
                drawn box for every inference.
              </p>
            </div>
          </div>

          <div className="flex gap-4 rounded-xl border border-slate-200 p-4">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-indigo-50 text-sm font-bold text-indigo-800">
              02
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Parameter-efficient model adaptation
              </h3>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">
                Investigate low-rank adaptation of MedSAM attention
                projections as a research direction, comparing it with
                suitable baseline configurations. Actual LoRA implementation
                and performance must be verified experimentally.
              </p>
            </div>
          </div>

          <div className="flex gap-4 rounded-xl border border-slate-200 p-4">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-50 text-sm font-bold text-emerald-800">
              03
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Quantitative evaluation
              </h3>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">
                Measure localization and segmentation performance using
                appropriate overlap metrics, and investigate how results
                change with different amounts of labeled training data.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 03: Architecture */}
      <section className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5 sm:p-7">
        <SectionHeading
          number="03"
          title="Model architecture"
          subtitle="The proposed end-to-end workflow, from ultrasound image to predicted lesion mask."
          icon={GitBranch}
        />

        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 sm:p-5">
          <div className="mb-5 flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Segmentation pipeline
              </h3>
              <p className="mt-1 text-xs text-slate-500">
                Conceptual architecture
              </p>
            </div>
            <span className="rounded-md border border-slate-200 bg-white px-2.5 py-1 text-[10px] font-semibold text-slate-600">
              5 stages
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
            <PipelineStep
              number="01"
              title="Input image"
              description="Load a breast ultrasound image and prepare it for inference."
              tag="Preprocessing"
              color="bg-slate-400"
            />
            <PipelineStep
              number="02"
              title="Image encoder"
              description="Extract image features using the MedSAM ViT-B encoder."
              tag="MedSAM"
              color="bg-sky-500"
            />
            <PipelineStep
              number="03"
              title="Auto-prompt"
              description="Predict a lesion bounding box with the localization network."
              tag="CNN / RoI"
              color="bg-amber-500"
            />
            <PipelineStep
              number="04"
              title="Mask decoder"
              description="Combine image features and box prompts to predict a segmentation mask."
              tag="Segmentation"
              color="bg-violet-500"
            />
            <PipelineStep
              number="05"
              title="Output mask"
              description="Display the predicted lesion region for research evaluation."
              tag="Prediction"
              color="bg-emerald-500"
            />
          </div>

          <div className="mt-4 flex items-start gap-2 rounded-lg border border-sky-100 bg-sky-50 p-3">
            <Info size={15} className="mt-0.5 shrink-0 text-sky-700" />
            <p className="text-xs leading-relaxed text-sky-900">
              <strong>Adaptation note:</strong> LoRA is an additional
              parameter-efficient adaptation strategy under investigation;
              it is not assumed to be active in every pipeline run. Confirm
              the selected checkpoint and implementation before attributing
              any output to LoRA.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-xl border border-slate-200 p-4">
            <h3 className="text-sm font-bold text-slate-900">
              Automatic prompt generator
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-600">
              The localization network estimates a region of interest and
              supplies the predicted box to the segmentation model. Its
              accuracy can be evaluated independently using ground-truth
              annotations.
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 p-4">
            <h3 className="text-sm font-bold text-slate-900">
              MedSAM segmentation
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-600">
              MedSAM uses image embeddings and prompt information to generate
              a mask. The quality of the resulting segmentation depends on
              the image, prompt, checkpoint, and model configuration.
            </p>
          </div>
        </div>
      </section>

      {/* 04: Dataset */}
      <section className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5 sm:p-7">
        <SectionHeading
          number="04"
          title="Dataset and experimental design"
          subtitle="The BUSI dataset provides the breast ultrasound images used in this project."
          icon={Layers3}
        />

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs font-medium text-slate-500">
              Dataset
            </p>
            <p className="mt-2 text-xl font-bold text-slate-900">BUSI</p>
            <p className="mt-1 text-xs text-slate-500">
              Breast ultrasound
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs font-medium text-slate-500">
              Image categories
            </p>
            <p className="mt-2 text-xl font-bold text-slate-900">3</p>
            <p className="mt-1 text-xs text-slate-500">
              Benign, malignant, normal
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs font-medium text-slate-500">
              Dataset size
            </p>
            <p className="mt-2 text-xl font-bold text-slate-900">
              780
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Images in the project dataset
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-bold text-slate-900">
            Evaluation protocol
          </h3>
          <ul className="mt-3 space-y-2.5 text-xs leading-relaxed text-slate-600">
            <li className="flex items-start gap-2">
              <CheckCircle2
                size={15}
                className="mt-0.5 shrink-0 text-teal-700"
              />
              Evaluate localization using predicted bounding boxes and
              appropriate ground-truth regions.
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2
                size={15}
                className="mt-0.5 shrink-0 text-teal-700"
              />
              Evaluate segmentation using predicted masks and corresponding
              ground-truth masks.
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2
                size={15}
                className="mt-0.5 shrink-0 text-teal-700"
              />
              Keep evaluation data separate from training and model-selection
              data to reduce data leakage.
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2
                size={15}
                className="mt-0.5 shrink-0 text-teal-700"
              />
              Compare baseline and adapted models using the same evaluation
              protocol.
            </li>
          </ul>
        </div>

        <div className="flex items-start gap-2 rounded-lg bg-amber-50 p-3">
          <AlertTriangle
            size={15}
            className="mt-0.5 shrink-0 text-amber-700"
          />
          <p className="text-xs leading-relaxed text-amber-900">
            Dataset size alone does not establish the train, validation, or
            test split. Report the actual split used in each experiment and
            verify that related images do not leak across splits.
          </p>
        </div>
      </section>

      {/* 05: Metrics */}
      <section className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5 sm:p-7">
        <SectionHeading
          number="05"
          title="Evaluation metrics"
          subtitle="Use overlap metrics to assess segmentation quality and distinguish them from localization performance."
          icon={BookOpen}
        />

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <MetricCard
            title="Dice similarity"
            formula="2|A ∩ B| / (|A| + |B|)"
            description="Measures overlap between the predicted mask A and ground-truth mask B. A value closer to 1 indicates greater overlap."
            direction="Higher is better"
          />

          <MetricCard
            title="Intersection over Union"
            formula="|A ∩ B| / |A ∪ B|"
            description="Measures the intersection relative to the union of the predicted and ground-truth regions."
            direction="Higher is better"
          />

          <MetricCard
            title="Bounding-box IoU"
            formula="|P ∩ G| / |P ∪ G|"
            description="Measures overlap between a predicted box P and ground-truth box G. This evaluates localization, not the final segmentation mask."
            direction="Higher is better"
          />
        </div>

        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <h3 className="text-sm font-bold text-slate-900">
            Metric interpretation
          </h3>
          <p className="mt-2 text-xs leading-relaxed text-slate-600">
            Dice and mask IoU assess segmentation overlap, while bounding-box
            IoU assesses localization. These scores should be reported
            separately. Boundary-distance metrics such as HD95 may also be
            considered when the mask geometry and pixel-spacing assumptions
            are defined.
          </p>
        </div>
      </section>

      {/* 06: Limitations */}
      <section className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5 sm:p-7">
        <SectionHeading
          number="06"
          title="Limitations and future work"
          subtitle="Areas requiring further experimentation before drawing broader conclusions."
          icon={FlaskConical}
        />

        <div className="space-y-3">
          <div className="flex gap-3 rounded-xl border border-slate-200 p-4">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-amber-50 text-amber-700">
              <Target size={17} />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Small or low-contrast lesions
              </h3>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">
                Small lesions and indistinct boundaries may make automatic
                localization and segmentation more difficult. Their error
                patterns should be measured during evaluation.
              </p>
            </div>
          </div>

          <div className="flex gap-3 rounded-xl border border-slate-200 p-4">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-sky-50 text-sky-700">
              <Layers3 size={17} />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Generalization across datasets
              </h3>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">
                Evaluation on additional datasets and acquisition settings
                would help assess whether the approach generalizes beyond
                BUSI.
              </p>
            </div>
          </div>

          <div className="flex gap-3 rounded-xl border border-slate-200 p-4">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-violet-50 text-violet-700">
              <Cpu size={17} />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Adaptation and efficiency
              </h3>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">
                Compare parameter-efficient adaptation with a baseline,
                measuring segmentation quality, trainable parameters, and
                inference cost under a reproducible protocol.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-slate-50 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-bold text-slate-800">
            MedSegment AI
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Research methodology and evaluation reference
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
          <span className="h-2 w-2 rounded-full bg-teal-600" />
          Experimental research
          <ArrowRight size={14} />
        </div>
      </footer>
    </div>
  );
};
