import AsyncStorage from "@react-native-async-storage/async-storage";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

interface Scan {
  id: string;
  date: string;
  hiveName: string;
  frameNumber: number;
} // TODO: add more here?

export interface StitchedScan {
  id: string;
  panoramaUri: string;
  name: string;
  date: string;
  hiveNo?: number;
  // TODO: add confidence here once ML model is wired up
  // confidence?: number;
}

// TODO: probably change something here, but it works fine for now
interface BeeState {
  scans: Scan[];
  stitchedScans: StitchedScan[];
  bkaEmail: string;
  addScan: (scan: Scan) => void;
  clearHistory: () => void;
  setBkaEmail: (email: string) => void;
  addStitchedScan: (payload: { panoramaUri: string; hiveNo?: number }) => void;
  getLatestStitchedScan: () => StitchedScan | null;
  updateScanName: (id: string, name: string) => void;
  deleteStitchedScan: (id: string) => void;
}

export const useBeeStore = create<BeeState>()(
  persist(
    (set, get) => ({
      scans: [],
      stitchedScans: [],
      bkaEmail: "",

      addScan: (newScan) =>
        set((state) => ({ scans: [newScan, ...state.scans] })),

      clearHistory: () => set({ scans: [] }),

      setBkaEmail: (email) => set({ bkaEmail: email }),

      addStitchedScan: ({ panoramaUri, hiveNo }) => {
        const scan: StitchedScan = {
          id: Date.now().toString(),
          panoramaUri,
          name: `Scan ${new Date().toLocaleDateString()}`,
          date: new Date().toLocaleDateString(),
          hiveNo,
        };
        set((state) => ({
          stitchedScans: [...state.stitchedScans, scan],
        }));
      },

      getLatestStitchedScan: () => {
        const { stitchedScans } = get();
        return stitchedScans.length > 0
          ? stitchedScans[stitchedScans.length - 1]
          : null;
      },

      updateScanName: (id, name) =>
        set((state) => ({
          stitchedScans: state.stitchedScans.map((s) =>
            s.id === id ? { ...s, name } : s
          ),
        })),

      deleteStitchedScan: (id) =>
        set((state) => ({
          stitchedScans: state.stitchedScans.filter((s) => s.id !== id),
        })),
    }),
    {
      name: "bee-storage",
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);