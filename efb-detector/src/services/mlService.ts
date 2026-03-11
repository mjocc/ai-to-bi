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

  return tf.tidy(() => {
    const rgbaTensor = tf.tensor3d(rgba, [height, width, 4], "int32");
    const rgbTensor = rgbaTensor.slice([0, 0, 0], [height, width, 3]);
    const normalized = rgbTensor.toFloat().div(tf.scalar(255.0));

    const scale = Math.min(1, MAX_DIM / Math.max(height, width));
    const newH = Math.max(Math.round(height * scale), 640);
    const newW = Math.max(Math.round(width * scale), 640);

    return normalized.resizeBilinear([newH, newW]) as tf.Tensor3D;
  });
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

    const sorted = [...probs].sort((a, b) => b - a);
    const topK = sorted.slice(0, 5);
    const mean = topK.reduce((sum, p) => sum + p, 0) / topK.length;
    return Math.round(mean * 100);
  } finally {
    tensor.dispose();
  }
}
