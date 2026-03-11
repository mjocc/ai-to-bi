import RNFS from 'react-native-fs';
import { getBBoxes, preloadLocatorModel } from './locate_larvae';
import { processImageWithBBoxes, preloadClassifierModel } from './classify_larvae';

export async function preloadModels(): Promise<void> {
  await Promise.all([preloadLocatorModel(), preloadClassifierModel()]);
}

export async function runMLPipeline(
  fileUri: string,
  width: number,
  height: number
): Promise<number> {
  const filePath = fileUri.startsWith('file://') ? fileUri.slice(7) : fileUri;
  const base64 = await RNFS.readFile(filePath, 'base64');
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  const shape: [number, number, number] = [height, width, 3];
  const bboxes = await getBBoxes(bytes, shape);
  if (bboxes.length === 0) return 0;

  const probs = await processImageWithBBoxes(bytes, shape, bboxes);
  if (probs.length === 0) return 0;

  return Math.round(Math.max(...probs) * 100);
}
