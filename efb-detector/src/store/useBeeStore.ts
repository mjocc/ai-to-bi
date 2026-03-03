import AsyncStorage from "@react-native-async-storage/async-storage";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

interface Scan {
  id: string;
  date: string;
  hiveName: string;
  frameNumber: number;
} // TODO: add more here once know what they want

// TODO: work out what actually want here as this is bad
interface BeeState {
  scans: Scan[];
  bkaEmail: string;
  addScan: (scan: Scan) => void;
  clearHistory: () => void;
  setBkaEmail: (email: string) => void;
}

export const useBeeStore = create<BeeState>()(
  persist(
    (set) => ({
      scans: [],
      bkaEmail: "",

      addScan: (newScan) =>
        set((state) => ({ scans: [newScan, ...state.scans] })),

      clearHistory: () => set({ scans: [] }),
      setBkaEmail: (email) => set({ bkaEmail: email }),
    }),
    {
      name: "bee-storage",
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);
