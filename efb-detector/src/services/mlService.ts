import RNFS from "react-native-fs";
import * as jpeg from "jpeg-js";
import { toByteArray } from "base64-js";
import * as tf from "@tensorflow/tfjs";
import { getBBoxes, preloadLocatorModel } from "./locate_larvae";
import { processImageWithBBoxes, preloadClassifierModel } from "./classify_larvae";

const MAX_DIM = 1280;

export async function preloadModels(): Promise<void> {
  await Promise.all([preloadLocatorModel(), preloadClassifierModel()]);
}

function preprocessImage(rawImageData: {
  data: Uint8Array;
  width: number;
  height: number;
}): tf.Tensor3D {
  const { data: rgba, width, height } = rawImageData;
  const rgb = new Float32Array(width * height * 3);
  let j = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    rgb[j++] = rgba[i] / 255;
    rgb[j++] = rgba[i + 1] / 255;
    rgb[j++] = rgba[i + 2] / 255;
  }

  let tensor = tf.tensor3d(rgb, [height, width, 3], "float32");
  const h = tensor.shape[0] as number;
  const w = tensor.shape[1] as number;
  const scale = Math.min(1, MAX_DIM / Math.max(h, w));
  const newH = Math.max(Math.round(h * scale), 640);
  const newW = Math.max(Math.round(w * scale), 640);
  const resized = tensor.resizeBilinear([newH, newW]) as tf.Tensor3D;
  tensor.dispose();
  return resized;
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

  const rawImageData = jpeg.decode(bytes, { useTArray: true });
  console.log("Called getBBoxes with shape: " + [rawImageData.height, rawImageData.width, 3]);

  const tensor = preprocessImage(rawImageData);
  try {
    const bboxes = await getBBoxes(tensor);
    if (bboxes.length === 0) return 0;

    console.log("Called processImageWithBBoxes with " + bboxes.length + " bboxes");
    const probs = await processImageWithBBoxes(tensor, bboxes);
    if (probs.length === 0) return 0;

    return Math.round(Math.max(...probs) * 100);
  } finally {
    tensor.dispose();
  }
}
