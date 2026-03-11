import RNFS from "react-native-fs";
import * as jpeg from "jpeg-js";
import { toByteArray } from "base64-js";
import { getBBoxes, preloadLocatorModel } from "./locate_larvae";
import { processImageWithBBoxes, preloadClassifierModel } from "./classify_larvae";

export async function preloadModels(): Promise<void> {
  await Promise.all([preloadLocatorModel(), preloadClassifierModel()]);
}

export async function runMLPipeline(
  fileUri: string,
  _width: number,
  _height: number
): Promise<number> {
  console.log("started running ML pipeline");
  const filePath = fileUri.startsWith("file://") ? fileUri.slice(7) : fileUri;
  const base64 = await RNFS.readFile(filePath, "base64");
  const bytes = toByteArray(base64);

  // Decode JPEG once — both functions reuse the same decoded pixel data
  const rawImageData = jpeg.decode(bytes, { useTArray: true });
  console.log("Called getBBoxes with shape: " + [rawImageData.height, rawImageData.width, 3]);

  const bboxes = await getBBoxes(rawImageData);
  if (bboxes.length === 0) return 0;

  console.log("Called processImageWithBBoxes with " + bboxes.length + " bboxes");
  const probs = await processImageWithBBoxes(rawImageData, bboxes);
  if (probs.length === 0) return 0;

  return Math.round(Math.max(...probs) * 100);
}
