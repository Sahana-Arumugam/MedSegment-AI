
import React, { useEffect, useMemo, useState } from 'react';
import {
  Grid2X2,
  List,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Database,
  ExternalLink,
  LoaderCircle,
  ImageOff,
  ScanSearch,
  CheckCircle2,
  Layers3,
  ArrowUpRight,
  Image as ImageIcon,
} from 'lucide-react';

import { fetchBUSIDataset } from '../services/busiApi';
import { BusiImage, DatasetCategory } from '../types';

interface DatasetExplorerViewProps {
  onSelectForSegmentation: (image: BusiImage) => void;
}

type CategoryFilter = 'all' | DatasetCategory;
type ViewMode = 'grid' | 'list';

const ITEMS_PER_PAGE = 8;

const categoryConfig: Record<
  DatasetCategory,
  {
    label: string;
    color: string;
    dot: string;
    badge: string;
    description: string;
  }
> = {
  benign: {
    label: 'Benign',
    color: 'text-teal-700',
    dot: 'bg-teal-500',
    badge: 'border-teal-200 bg-teal-50 text-teal-800',
    description: 'Non-malignant cases',
  },
  malignant: {
    label: 'Malignant',
    color: 'text-rose-700',
    dot: 'bg-rose-500',
    badge: 'border-rose-200 bg-rose-50 text-rose-800',
    description: 'Malignant cases',
  },
  normal: {
    label: 'Normal',
    color: 'text-slate-600',
    dot: 'bg-slate-400',
    badge: 'border-slate-200 bg-slate-100 text-slate-700',
    description: 'No lesion identified',
  },
};

export const DatasetExplorerView: React.FC<DatasetExplorerViewProps> = ({
  onSelectForSegmentation,
}) => {
  const [dataset, setDataset] = useState<BusiImage[]>([]);
  const [selectedCategory, setSelectedCategory] =
    useState<CategoryFilter>('all');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [inspectImage, setInspectImage] = useState<BusiImage | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showMask, setShowMask] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    let cancelled = false;

    async function loadDataset() {
      setIsLoading(true);
      setLoadError(null);

      try {
        const response = await fetchBUSIDataset();

        const images: BusiImage[] = response.images.map((item) => ({
          id: item.id,
          filename: item.filename,
          category: item.category,
          width: 0,
          height: 0,
          // Compatibility field only; this view does not assign real splits.
          split: 'train',
          hasMask: item.has_mask,
          maskCount: item.mask_urls.length,
          imageUrl: item.image_url,
          maskUrls: item.mask_urls,
          sonogramSeed: 0,
        }));

        if (!cancelled) {
          setDataset(images);
          setInspectImage(images[0] ?? null);
          setCurrentPage(1);
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(
            error instanceof Error
              ? error.message
              : 'Unable to load the BUSI dataset.',
          );
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    void loadDataset();

    return () => {
      cancelled = true;
    };
  }, [retryKey]);

  const counts = useMemo(
    () => ({
      all: dataset.length,
      benign: dataset.filter((item) => item.category === 'benign').length,
      malignant: dataset.filter((item) => item.category === 'malignant').length,
      normal: dataset.filter((item) => item.category === 'normal').length,
      annotated: dataset.filter((item) => item.hasMask).length,
    }),
    [dataset],
  );

  const filteredImages = useMemo(
    () =>
      selectedCategory === 'all'
        ? dataset
        : dataset.filter((item) => item.category === selectedCategory),
    [dataset, selectedCategory],
  );

  const totalPages = Math.max(
    1,
    Math.ceil(filteredImages.length / ITEMS_PER_PAGE),
  );

  const paginatedImages = useMemo(
    () =>
      filteredImages.slice(
        (currentPage - 1) * ITEMS_PER_PAGE,
        currentPage * ITEMS_PER_PAGE,
      ),
    [filteredImages, currentPage],
  );

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [currentPage, totalPages]);

  useEffect(() => {
    if (
      inspectImage &&
      !filteredImages.some((item) => item.id === inspectImage.id)
    ) {
      setInspectImage(filteredImages[0] ?? null);
      setShowMask(false);
    } else if (!inspectImage && filteredImages.length > 0) {
      setInspectImage(filteredImages[0]);
    }
  }, [filteredImages, inspectImage]);

  function selectCategory(category: CategoryFilter) {
    setSelectedCategory(category);
    setCurrentPage(1);
  }

  function selectImage(item: BusiImage) {
    setInspectImage(item);
    setShowMask(false);
  }

  function updateDimensions(
    imageId: string,
    width: number,
    height: number,
  ) {
    setDataset((previous) =>
      previous.map((item) =>
        item.id === imageId ? { ...item, width, height } : item,
      ),
    );

    setInspectImage((previous) =>
      previous?.id === imageId
        ? { ...previous, width, height }
        : previous,
    );
  }

  function categoryBadge(category: DatasetCategory) {
    return categoryConfig[category]?.badge ?? categoryConfig.normal.badge;
  }

  function getImageUrl(item: BusiImage) {
    const url = item.imageUrl;
    if (!url) return undefined;
    if (/^(https?:|data:|blob:)/i.test(url)) return url;
    return `http://127.0.0.1:8000${url.startsWith('/') ? url : `/${url}`}`;
  }

  function getMaskUrl(item: BusiImage) {
    const url = item.maskUrls?.[0];
    if (!url) return undefined;
    if (/^(https?:|data:|blob:)/i.test(url)) return url;
    return `http://127.0.0.1:8000${url.startsWith('/') ? url : `/${url}`}`;
  }

  return (
    <main className="mx-auto max-w-[1600px] space-y-6 pb-12">
      {/* Page introduction */}
      <section className="relative overflow-hidden rounded-3xl bg-[#102d35] p-6 text-white shadow-sm sm:p-8">
        <div className="pointer-events-none absolute -right-12 -top-20 h-72 w-72 rounded-full border border-white/10" />
        <div className="pointer-events-none absolute -right-1 -top-9 h-52 w-52 rounded-full border border-white/10" />
        <div className="pointer-events-none absolute bottom-[-100px] right-44 h-56 w-56 rounded-full bg-teal-400/10 blur-3xl" />

        <div className="relative flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
          <div className="max-w-2xl">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-teal-300/20 bg-teal-300/10 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-teal-200">
              <Database className="h-3.5 w-3.5" />
              Research data workspace
            </div>

            <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
              Dataset Explorer
            </h1>

            <p className="mt-3 max-w-xl text-sm leading-6 text-slate-300">
              Explore breast ultrasound images, review available annotations,
              and send a selected image directly to the segmentation workspace.
            </p>

            <div className="mt-5 flex flex-wrap items-center gap-3 text-xs text-slate-300">
              <span className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                <span className="h-2 w-2 rounded-full bg-teal-400" />
                BUSI dataset
              </span>
              <span className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                <Layers3 className="h-3.5 w-3.5 text-teal-300" />
                Image and mask records
              </span>
            </div>
          </div>

          <div className="relative grid min-w-[220px] grid-cols-2 gap-3">
            <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-4 backdrop-blur-sm">
              <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
                Images loaded
              </p>
              <p className="mt-2 text-3xl font-semibold tabular-nums">
                {isLoading ? '—' : counts.all.toLocaleString()}
              </p>
              <p className="mt-1 text-[10px] text-slate-400">
                Dataset records
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-4 backdrop-blur-sm">
              <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
                With masks
              </p>
              <p className="mt-2 text-3xl font-semibold tabular-nums">
                {isLoading ? '—' : counts.annotated.toLocaleString()}
              </p>
              <p className="mt-1 text-[10px] text-slate-400">
                Annotated images
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Dataset statistics */}
      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <button
          type="button"
          onClick={() => selectCategory('all')}
          className={`rounded-2xl border p-4 text-left transition-all hover:-translate-y-0.5 hover:shadow-md ${
            selectedCategory === 'all'
              ? 'border-slate-800 bg-white shadow-sm ring-1 ring-slate-800'
              : 'border-slate-200 bg-white hover:border-slate-300'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">
              All images
            </span>
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
              <Database className="h-4 w-4" />
            </span>
          </div>
          <p className="mt-4 text-2xl font-semibold tracking-tight text-slate-900 tabular-nums">
            {isLoading ? '—' : counts.all.toLocaleString()}
          </p>
          <p className="mt-1 text-[11px] text-slate-500">
            Entire loaded collection
          </p>
        </button>

        {(['benign', 'malignant', 'normal'] as const).map((category) => {
          const config = categoryConfig[category];
          const isActive = selectedCategory === category;

          return (
            <button
              key={category}
              type="button"
              onClick={() => selectCategory(category)}
              className={`rounded-2xl border bg-white p-4 text-left transition-all hover:-translate-y-0.5 hover:shadow-md ${
                isActive
                  ? 'border-slate-800 shadow-sm ring-1 ring-slate-800'
                  : 'border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-500">
                  {config.label}
                </span>
                <span
                  className={`flex h-9 w-9 items-center justify-center rounded-xl ${
                    category === 'benign'
                      ? 'bg-teal-50 text-teal-700'
                      : category === 'malignant'
                        ? 'bg-rose-50 text-rose-700'
                        : 'bg-slate-100 text-slate-600'
                  }`}
                >
                  <span className={`h-2.5 w-2.5 rounded-full ${config.dot}`} />
                </span>
              </div>
              <p className={`mt-4 text-2xl font-semibold tracking-tight tabular-nums ${config.color}`}>
                {isLoading ? '—' : counts[category].toLocaleString()}
              </p>
              <p className="mt-1 text-[11px] text-slate-500">
                {config.description}
              </p>
            </button>
          );
        })}
      </section>

      {/* Distribution */}
      {!isLoading && !loadError && counts.all > 0 && (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6">
          <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">
                Dataset composition
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Distribution of the loaded records by category
              </p>
            </div>
            <span className="text-xs font-medium text-slate-500">
              {counts.all.toLocaleString()} total records
            </span>
          </div>

          <div className="mt-5 flex h-3 w-full overflow-hidden rounded-full bg-slate-100">
            {counts.benign > 0 && (
              <div
                className="bg-teal-500 transition-all"
                style={{ width: `${(counts.benign / counts.all) * 100}%` }}
                title={`Benign: ${counts.benign}`}
              />
            )}
            {counts.malignant > 0 && (
              <div
                className="bg-rose-500 transition-all"
                style={{ width: `${(counts.malignant / counts.all) * 100}%` }}
                title={`Malignant: ${counts.malignant}`}
              />
            )}
            {counts.normal > 0 && (
              <div
                className="bg-slate-400 transition-all"
                style={{ width: `${(counts.normal / counts.all) * 100}%` }}
                title={`Normal: ${counts.normal}`}
              />
            )}
          </div>

          <div className="mt-4 flex flex-wrap gap-x-6 gap-y-3">
            {(['benign', 'malignant', 'normal'] as const).map((category) => (
              <div key={category} className="flex items-center gap-2">
                <span className={`h-2.5 w-2.5 rounded-full ${categoryConfig[category].dot}`} />
                <span className="text-xs text-slate-600">
                  {categoryConfig[category].label}
                </span>
                <span className="text-xs font-semibold text-slate-900 tabular-nums">
                  {counts[category].toLocaleString()}
                </span>
                <span className="text-[10px] text-slate-400">
                  {((counts[category] / counts.all) * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Gallery and inspection workspace */}
      <section className="grid grid-cols-1 items-start gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-4">
          {/* Gallery toolbar */}
          <div className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-200 bg-white p-4 sm:flex-row sm:items-center">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold text-slate-900">
                  Image collection
                </h2>
                <span className="rounded-md bg-slate-100 px-2 py-1 text-[10px] font-medium text-slate-600">
                  {filteredImages.length.toLocaleString()} images
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                {selectedCategory === 'all'
                  ? 'Viewing all categories'
                  : `Viewing ${categoryConfig[selectedCategory].label.toLowerCase()} cases`}
              </p>
            </div>

            <div className="flex items-center justify-between gap-3">
              <div className="flex flex-wrap gap-1.5">
                {(['all', 'benign', 'malignant', 'normal'] as const).map(
                  (category) => (
                    <button
                      key={category}
                      type="button"
                      onClick={() => selectCategory(category)}
                      className={`rounded-lg px-3 py-2 text-xs font-medium capitalize transition-colors ${
                        selectedCategory === category
                          ? 'bg-[#12343b] text-white'
                          : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                      }`}
                    >
                      {category === 'all' ? 'All' : category}
                    </button>
                  ),
                )}
              </div>

              <div className="flex shrink-0 items-center gap-1 rounded-lg border border-slate-200 p-1">
                <button
                  type="button"
                  onClick={() => setViewMode('grid')}
                  className={`rounded-md p-2 ${
                    viewMode === 'grid'
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-500 hover:bg-slate-100'
                  }`}
                  title="Grid view"
                  aria-label="Grid view"
                  aria-pressed={viewMode === 'grid'}
                >
                  <Grid2X2 className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('list')}
                  className={`rounded-md p-2 ${
                    viewMode === 'list'
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-500 hover:bg-slate-100'
                  }`}
                  title="List view"
                  aria-label="List view"
                  aria-pressed={viewMode === 'list'}
                >
                  <List className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>

          {isLoading && (
            <div className="flex min-h-72 flex-col items-center justify-center rounded-2xl border border-slate-200 bg-white p-10 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-50 text-teal-700">
                <LoaderCircle className="h-5 w-5 animate-spin" />
              </div>
              <p className="mt-4 text-sm font-semibold text-slate-800">
                Loading dataset
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Fetching image records from the backend
              </p>
            </div>
          )}

          {loadError && (
            <div className="rounded-2xl border border-rose-200 bg-white p-6">
              <div className="flex items-start gap-3">
                <div className="rounded-xl bg-rose-50 p-2 text-rose-700">
                  <AlertTriangle className="h-5 w-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-slate-900">
                    Dataset could not be loaded
                  </h3>
                  <p className="mt-1 break-words text-xs leading-5 text-slate-600">
                    {loadError}
                  </p>
                  <p className="mt-2 text-xs text-slate-500">
                    Check that the FastAPI server is running and that the
                    dataset endpoint is available.
                  </p>
                  <button
                    type="button"
                    onClick={() => setRetryKey((value) => value + 1)}
                    className="mt-4 inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-xs font-semibold text-white hover:bg-slate-800"
                  >
                    <LoaderCircle className="h-3.5 w-3.5" />
                    Retry connection
                  </button>
                </div>
              </div>
            </div>
          )}

          {!isLoading && !loadError && paginatedImages.length === 0 && (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-12 text-center">
              <ImageOff className="mx-auto h-8 w-8 text-slate-300" />
              <h3 className="mt-3 text-sm font-semibold text-slate-800">
                No images in this category
              </h3>
              <p className="mt-1 text-xs text-slate-500">
                Choose another category to view more dataset records.
              </p>
              <button
                type="button"
                onClick={() => selectCategory('all')}
                className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-xs font-semibold text-white hover:bg-slate-800"
              >
                View all images
              </button>
            </div>
          )}

          {!isLoading && !loadError && paginatedImages.length > 0 && (
            <>
              {viewMode === 'grid' ? (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 2xl:grid-cols-4">
                  {paginatedImages.map((item) => {
                    const isSelected = inspectImage?.id === item.id;
                    const itemImageUrl = getImageUrl(item);

                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => selectImage(item)}
                        className={`group overflow-hidden rounded-2xl border bg-white text-left transition-all hover:-translate-y-0.5 hover:shadow-lg ${
                          isSelected
                            ? 'border-teal-600 ring-2 ring-teal-600/20'
                            : 'border-slate-200 hover:border-slate-300'
                        }`}
                      >
                        <div className="relative flex aspect-[4/3] items-center justify-center overflow-hidden bg-[#0a1018]">
                          {itemImageUrl ? (
                            <img
                              src={itemImageUrl}
                              alt={item.filename}
                              loading="lazy"
                              className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-[1.03]"
                              onLoad={(event) => {
                                const element = event.currentTarget;
                                updateDimensions(
                                  item.id,
                                  element.naturalWidth,
                                  element.naturalHeight,
                                );
                              }}
                            />
                          ) : (
                            <ImageOff className="h-7 w-7 text-slate-600" />
                          )}

                          <span
                            className={`absolute left-2.5 top-2.5 rounded-md border px-2 py-1 text-[9px] font-semibold uppercase tracking-wide ${categoryBadge(item.category)}`}
                          >
                            {item.category}
                          </span>

                          {item.hasMask && (
                            <span className="absolute right-2.5 top-2.5 flex items-center gap-1 rounded-md border border-emerald-300/20 bg-emerald-950/80 px-2 py-1 text-[9px] font-medium text-emerald-200">
                              <CheckCircle2 className="h-3 w-3" />
                              Mask
                            </span>
                          )}

                          {isSelected && (
                            <span className="absolute bottom-2.5 right-2.5 rounded-full bg-teal-500 p-1.5 text-white shadow-md">
                              <CheckCircle2 className="h-3.5 w-3.5" />
                            </span>
                          )}
                        </div>

                        <div className="p-3">
                          <p
                            title={item.filename}
                            className="truncate text-xs font-semibold text-slate-900"
                          >
                            {item.filename}
                          </p>

                          <div className="mt-2 flex items-center justify-between gap-2">
                            <span className="truncate text-[10px] text-slate-500">
                              {item.width > 0 && item.height > 0
                                ? `${item.width} × ${item.height} px`
                                : 'Image record'}
                            </span>
                            <ArrowUpRight
                              className={`h-3.5 w-3.5 shrink-0 transition-colors ${
                                isSelected
                                  ? 'text-teal-700'
                                  : 'text-slate-300 group-hover:text-slate-700'
                              }`}
                            />
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse text-left">
                      <thead>
                        <tr className="border-b border-slate-200 bg-slate-50 text-[10px] uppercase tracking-wider text-slate-500">
                          <th className="px-4 py-3 font-semibold">Image</th>
                          <th className="px-3 py-3 font-semibold">Category</th>
                          <th className="px-3 py-3 font-semibold">Resolution</th>
                          <th className="px-4 py-3 font-semibold">Annotation</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {paginatedImages.map((item) => {
                          const isSelected = inspectImage?.id === item.id;
                          const itemImageUrl = getImageUrl(item);

                          return (
                            <tr
                              key={item.id}
                              onClick={() => selectImage(item)}
                              className={`cursor-pointer transition-colors ${
                                isSelected ? 'bg-teal-50' : 'hover:bg-slate-50'
                              }`}
                            >
                              <td className="px-4 py-3">
                                <div className="flex min-w-[190px] items-center gap-3">
                                  {itemImageUrl ? (
                                    <img
                                      src={itemImageUrl}
                                      alt=""
                                      loading="lazy"
                                      className="h-12 w-16 rounded-lg bg-slate-950 object-contain"
                                      onLoad={(event) => {
                                        const element = event.currentTarget;
                                        updateDimensions(
                                          item.id,
                                          element.naturalWidth,
                                          element.naturalHeight,
                                        );
                                      }}
                                    />
                                  ) : (
                                    <div className="flex h-12 w-16 items-center justify-center rounded-lg bg-slate-100">
                                      <ImageOff className="h-4 w-4 text-slate-400" />
                                    </div>
                                  )}
                                  <span className="max-w-[200px] truncate text-xs font-semibold text-slate-800">
                                    {item.filename}
                                  </span>
                                </div>
                              </td>
                              <td className="px-3 py-3">
                                <span
                                  className={`rounded-md border px-2 py-1 text-[10px] font-semibold capitalize ${categoryBadge(item.category)}`}
                                >
                                  {item.category}
                                </span>
                              </td>
                              <td className="px-3 py-3 font-mono text-[11px] text-slate-500">
                                {item.width > 0 && item.height > 0
                                  ? `${item.width} × ${item.height}`
                                  : '—'}
                              </td>
                              <td className="px-4 py-3">
                                <span
                                  className={`text-[11px] font-medium ${
                                    item.hasMask
                                      ? 'text-emerald-700'
                                      : 'text-slate-400'
                                  }`}
                                >
                                  {item.hasMask
                                    ? `${item.maskCount} available`
                                    : 'Not available'}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Pagination */}
              <div className="flex flex-col justify-between gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 sm:flex-row sm:items-center">
                <p className="text-xs text-slate-500">
                  Showing{' '}
                  <span className="font-semibold text-slate-800">
                    {(currentPage - 1) * ITEMS_PER_PAGE + 1}–
                    {Math.min(
                      currentPage * ITEMS_PER_PAGE,
                      filteredImages.length,
                    )}
                  </span>{' '}
                  of {filteredImages.length.toLocaleString()} images
                </p>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setCurrentPage((page) => Math.max(1, page - 1))
                    }
                    disabled={currentPage === 1}
                    className="rounded-lg border border-slate-200 p-2 text-slate-600 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
                    aria-label="Previous page"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>

                  <span className="min-w-16 text-center text-xs font-semibold tabular-nums text-slate-700">
                    {currentPage} / {totalPages}
                  </span>

                  <button
                    type="button"
                    onClick={() =>
                      setCurrentPage((page) =>
                        Math.min(totalPages, page + 1),
                      )
                    }
                    disabled={currentPage === totalPages}
                    className="rounded-lg border border-slate-200 p-2 text-slate-600 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
                    aria-label="Next page"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Inspection panel */}
        <aside className="overflow-hidden rounded-2xl border border-slate-200 bg-white xl:sticky xl:top-5">
          <div className="border-b border-slate-200 bg-slate-50/80 px-5 py-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#12343b] text-teal-200">
                <ScanSearch className="h-4 w-4" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-slate-900">
                  Image details
                </h2>
                <p className="mt-0.5 text-[10px] text-slate-500">
                  Inspect the selected record
                </p>
              </div>
            </div>
          </div>

          {inspectImage ? (
            <>
              <div className="relative flex aspect-[4/3] items-center justify-center overflow-hidden bg-[#080d14]">
                {showMask && getMaskUrl(inspectImage) ? (
                  <img
                    src={getMaskUrl(inspectImage)}
                    alt={`Ground-truth mask for ${inspectImage.filename}`}
                    className="max-h-full max-w-full object-contain"
                  />
                ) : getImageUrl(inspectImage) ? (
                  <img
                    src={getImageUrl(inspectImage)}
                    alt={inspectImage.filename}
                    className="max-h-full max-w-full object-contain"
                    onLoad={(event) => {
                      const element = event.currentTarget;
                      updateDimensions(
                        inspectImage.id,
                        element.naturalWidth,
                        element.naturalHeight,
                      );
                    }}
                  />
                ) : (
                  <div className="flex flex-col items-center gap-2 text-slate-500">
                    <ImageOff className="h-8 w-8" />
                    <span className="text-xs">Image unavailable</span>
                  </div>
                )}

                <span
                  className={`absolute left-3 top-3 rounded-md border px-2 py-1 text-[9px] font-semibold uppercase tracking-wide ${categoryBadge(inspectImage.category)}`}
                >
                  {inspectImage.category}
                </span>
              </div>

              <div className="space-y-5 p-5">
                <div>
                  <p className="break-all text-sm font-semibold leading-5 text-slate-900">
                    {inspectImage.filename}
                  </p>
                  <p className="mt-1 text-[11px] text-slate-500">
                    BUSI ultrasound record
                  </p>
                </div>

                {inspectImage.hasMask && getMaskUrl(inspectImage) && (
                  <button
                    type="button"
                    onClick={() => setShowMask((previous) => !previous)}
                    className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 px-3 py-2.5 text-xs font-semibold text-slate-700 transition-colors hover:border-teal-300 hover:bg-teal-50 hover:text-teal-800"
                  >
                    <Layers3 className="h-4 w-4" />
                    {showMask ? 'View ultrasound image' : 'View ground-truth mask'}
                  </button>
                )}

                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded-xl bg-slate-50 p-3">
                    <p className="text-[10px] text-slate-500">Resolution</p>
                    <p className="mt-1 text-xs font-semibold text-slate-900">
                      {inspectImage.width > 0 && inspectImage.height > 0
                        ? `${inspectImage.width} × ${inspectImage.height}`
                        : 'Not loaded'}
                    </p>
                  </div>
                  <div className="rounded-xl bg-slate-50 p-3">
                    <p className="text-[10px] text-slate-500">Masks</p>
                    <p className="mt-1 text-xs font-semibold text-slate-900">
                      {inspectImage.maskCount}
                    </p>
                  </div>
                </div>

                <div className="space-y-3 border-t border-slate-100 pt-4">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-xs text-slate-500">Category</span>
                    <span className="text-xs font-semibold capitalize text-slate-800">
                      {inspectImage.category}
                    </span>
                  </div>

                  <div className="flex items-center justify-between gap-3">
                    <span className="text-xs text-slate-500">Ground truth</span>
                    <span
                      className={`inline-flex items-center gap-1.5 text-xs font-medium ${
                        inspectImage.hasMask
                          ? 'text-emerald-700'
                          : 'text-slate-500'
                      }`}
                    >
                      {inspectImage.hasMask ? (
                        <CheckCircle2 className="h-3.5 w-3.5" />
                      ) : (
                        <ImageOff className="h-3.5 w-3.5" />
                      )}
                      {inspectImage.hasMask ? 'Available' : 'Unavailable'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between gap-3">
                    <span className="text-xs text-slate-500">Record ID</span>
                    <span className="max-w-40 truncate font-mono text-[10px] text-slate-600">
                      {inspectImage.id}
                    </span>
                  </div>
                </div>

                {!inspectImage.hasMask &&
                  inspectImage.category !== 'normal' && (
                    <div className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3">
                      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-700" />
                      <p className="text-[11px] leading-5 text-amber-800">
                        No ground-truth mask was returned for this record.
                      </p>
                    </div>
                  )}

                <button
                  type="button"
                  onClick={() => onSelectForSegmentation(inspectImage)}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#12343b] px-4 py-3 text-xs font-semibold text-white shadow-sm transition-all hover:bg-[#1a4850] hover:shadow-md"
                >
                  Open in Segmentation Studio
                  <ExternalLink className="h-3.5 w-3.5" />
                </button>
              </div>
            </>
          ) : (
            <div className="flex min-h-64 flex-col items-center justify-center p-8 text-center">
              <ImageIcon className="h-8 w-8 text-slate-300" />
              <p className="mt-3 text-sm font-semibold text-slate-700">
                Select an image
              </p>
              <p className="mt-1 text-xs leading-5 text-slate-500">
                Choose an image from the collection to inspect its details.
              </p>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
};

export default DatasetExplorerView;
