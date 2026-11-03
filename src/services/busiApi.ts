
const API_BASE_URL = "http://127.0.0.1:8000";

export interface BUSIImage {
  id: string;
  filename: string;
  category: "benign" | "malignant" | "normal";
  image_url: string;
  mask_urls: string[];
  has_mask: boolean;
}

export interface BUSIDatasetResponse {
  total_images: number;
  categories: Record<string, number>;
  images: BUSIImage[];
}

export async function fetchBUSIDataset(): Promise<BUSIDatasetResponse> {
  const response = await fetch(`${API_BASE_URL}/api/dataset`);

  if (!response.ok) {
    throw new Error(
      `Failed to load BUSI dataset: HTTP ${response.status}`
    );
  }

  const dataset: BUSIDatasetResponse = await response.json();

  return {
    ...dataset,
    images: dataset.images.map((image) => ({
      ...image,
      image_url: `${API_BASE_URL}${image.image_url}`,
      mask_urls: image.mask_urls.map(
        (url) => `${API_BASE_URL}${url}`
      ),
    })),
  };
}
