import { Directory, File, Paths } from "expo-file-system";
import { create } from "zustand";

export type Scan = {
  ScanID: number;
  HiveNo: number;
  ImageID: number;
  Confidence: number;
};

export type ImageRecord = {
  ImageID: number;
  // Only the filename is stored (e.g. "scan_001.jpg"), NOT the full URI.
  // The full path can change between app installs, so we reconstruct it at
  // runtime using getImageUri(). This prevents stale URIs after reinstalls.
  ImageFileName: string;
  ImageName: string;
  DateTaken: string;
  xco1: number;
  yco1: number;
  xco2: number;
  yco2: number;
};

// Reconstruct the full file:// URI from just the filename at runtime
export const getImageUri = (fileName: string): string =>
  new File(new Directory(Paths.document, "scans"), fileName).uri;

type StoreState = {
  scans: Scan[];
  images: ImageRecord[];

  // Actions
  initializeData: () => Promise<void>;
  addScan: (scan: Omit<Scan, "ScanID">) => void;
  addImage: (image: Omit<ImageRecord, "ImageID">) => void;
  updateImageName: (imageID: number, newName: string) => void;
  deleteImage: (imageID: number) => void;

  // Derived helpers
  getScansWithImageNames: () => (Scan & { ImageName: string })[];
  getImagesWithHive: () => (ImageRecord & { HiveNo: number })[];
  getDistinctHiveNumbers: () => number[];
  getScansByHive: (hiveNo: number) => (Scan & { ImageName: string; DateTaken: string })[];
};

export const useStore = create<StoreState>((set, get) => ({
  scans: [],
  images: [],

  initializeData: async () => {
    // Ensure the scans directory exists on first launch
    const scansDir = new Directory(Paths.document, "scans");
    if (!scansDir.exists) {
      scansDir.create();
    }
  },

  addScan: (scan) =>
    set((state) => ({
      scans: [
        ...state.scans,
        { ...scan, ScanID: (state.scans.at(-1)?.ScanID ?? 0) + 1 },
      ],
    })),

  addImage: (image) =>
    set((state) => ({
      images: [
        ...state.images,
        { ...image, ImageID: (state.images.at(-1)?.ImageID ?? 0) + 1 },
      ],
    })),

  updateImageName: (imageID, newName) =>
    set((state) => ({
      images: state.images.map((img) =>
        img.ImageID === imageID ? { ...img, ImageName: newName } : img
      ),
    })),

  // Note: this only removes the record from the store.
  // The caller is responsible for deleting the file from disk via expo-file-system.
  deleteImage: (imageID) =>
    set((state) => ({
      images: state.images.filter((img) => img.ImageID !== imageID),
      scans: state.scans.filter((scan) => scan.ImageID !== imageID),
    })),

  getScansWithImageNames: () => {
    const { scans, images } = get();
    return scans.map((scan) => ({
      ...scan,
      ImageName:
        images.find((img) => img.ImageID === scan.ImageID)?.ImageName ??
        `Image ${scan.ImageID}`,
    }));
  },

  getImagesWithHive: () => {
    const { scans, images } = get();
    return [...images]
      .sort((a, b) => new Date(b.DateTaken).getTime() - new Date(a.DateTaken).getTime())
      .map((img) => ({
        ...img,
        HiveNo: scans.find((s) => s.ImageID === img.ImageID)?.HiveNo ?? 0,
      }));
  },

  getDistinctHiveNumbers: () => {
    const hives = get().scans.map((s) => s.HiveNo);
    return [...new Set(hives)].sort((a, b) => a - b);
  },

  getScansByHive: (hiveNo) => {
    const { scans, images } = get();
    return scans
      .filter((s) => s.HiveNo === hiveNo)
      .map((scan) => {
        const img = images.find((i) => i.ImageID === scan.ImageID);
        return {
          ...scan,
          ImageName: img?.ImageName ?? `Image ${scan.ImageID}`,
          DateTaken: img?.DateTaken ?? "",
        };
      })
      .sort((a, b) => new Date(a.DateTaken).getTime() - new Date(b.DateTaken).getTime());
  },
}));