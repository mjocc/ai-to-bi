import { create } from "zustand";

interface Scan {
  id: string;
  date: string;
  hiveName: string;
  frameNumber: number;
} // TODO: add more here once know what they want

// TODO: work out what actually want here as this is bad
interface BeeState {
  scans: Scan[];
  addScan: (scan: Scan) => void;
  clearHistory: () => void;
}

export const useBeeStore = create<BeeState>((set) => ({
  scans: [],

  addScan: (newScan) => set((state) => ({ scans: [newScan, ...state.scans] })),

  clearHistory: () => set({ scans: [] }),
}));
