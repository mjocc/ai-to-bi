import AsyncStorage from "@react-native-async-storage/async-storage";
import { Directory, File, Paths } from "expo-file-system";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export const getImageUri = (fileName: string): string =>
  new File(new Directory(Paths.document, "scans"), fileName).uri;

export type Scan = {
  ScanID: number;
  HiveNo: number;
  ImageID: number;
  Confidence: number;
};

export type ImageRecord = {
  ImageID: number;
  ImageFileName: string;
  ImageName: string;
  DateTaken: string;
  xco1: number;
  yco1: number;
  xco2: number;
  yco2: number;
};

interface BeeState {
  scans: Scan[];
  images: ImageRecord[];
  bkaEmail: string;
  initializeData: () => Promise<void>;
  addScan: (scan: Omit<Scan, "ScanID">) => void;
  addImage: (image: Omit<ImageRecord, "ImageID">) => number;
  updateImageName: (imageID: number, newName: string) => void;
  deleteImage: (imageID: number) => void;
  getScansWithImageNames: () => (Scan & { ImageName: string })[];
  getImagesWithHive: () => (ImageRecord & { HiveNo: number })[];
  getDistinctHiveNumbers: () => number[];
  getScansByHive: (
    hiveNo: number
  ) => (Scan & { ImageName: string; DateTaken: string })[];
  setBkaEmail: (email: string) => void;
  clearHistory: () => void;
}

export const useBeeStore = create<BeeState>()(
  persist(
    (set, get) => ({
      scans: [],
      images: [],
      bkaEmail: "",

      initializeData: async () => {
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

      addImage: (image) => {
        const newId = (get().images.at(-1)?.ImageID ?? 0) + 1;
        set((state) => ({
          images: [...state.images, { ...image, ImageID: newId }],
        }));
        return newId;
      },

      updateImageName: (imageID, newName) =>
        set((state) => ({
          images: state.images.map((img) =>
            img.ImageID === imageID ? { ...img, ImageName: newName } : img
          ),
        })),

      deleteImage: (imageID) => {
        const { images } = get();
        const img = images.find(i => i.ImageID === imageID);
        if (img) {
          try {
            const file = new File(new Directory(Paths.document, "scans"), img.ImageFileName);
            if (file.exists) {
              file.delete();
            }
          } catch (e) {
            console.error("Failed to delete image file:", e);
          }
        }
        set((state) => ({
          images: state.images.filter((img) => img.ImageID !== imageID),
          scans: state.scans.filter((scan) => scan.ImageID !== imageID),
        }));
      },

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
          .sort(
            (a, b) =>
              new Date(b.DateTaken).getTime() - new Date(a.DateTaken).getTime()
          )
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
          .sort(
            (a, b) =>
              new Date(a.DateTaken).getTime() - new Date(b.DateTaken).getTime()
          );
      },

      setBkaEmail: (email) => set({ bkaEmail: email }),

      clearHistory: () => {
        try {
          const scansDir = new Directory(Paths.document, "scans");
          if (scansDir.exists) {
            scansDir.delete();
          }
          scansDir.create();
        } catch (e) {
          console.error("Failed to clear scans directory:", e);
          return;
        }
        set({ scans: [], images: [] });
      },
    }),
    {
      name: "bee-storage",
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);
