import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";
import * as tf from "@tensorflow/tfjs";

let classifierModel: TensorflowModel | null = null;

export async function preloadClassifierModel(): Promise<void> {
  classifierModel = await loadTensorflowModel(
    require("../../assets/models/disease_classifier_float16.tflite")
  );
}

// Must match MAX_DIM in locate_larvae.ts so bbox coordinates remain consistent
const MAX_DIM = 1280;

export async function processImageWithBBoxes(
  rawImageData: { data: Uint8Array; width: number; height: number },
  bboxes: number[][][]
) {
  const model =
    classifierModel ??
    (await loadTensorflowModel(
      require("../../assets/models/disease_classifier_float16.tflite")
    ));

  // RGBA → normalised RGB Float32Array (pre-allocated, stride-4 iteration)
  const { data: rgba, width, height } = rawImageData;
  const rgb = new Float32Array(width * height * 3);
  let j = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    rgb[j++] = rgba[i] / 255;
    rgb[j++] = rgba[i + 1] / 255;
    rgb[j++] = rgba[i + 2] / 255;
  }

  // Apply identical MAX_DIM cap so bbox coordinates from locate_larvae are valid
  let tensored_frame = tf.tensor3d(rgb, [height, width, 3], "float32");
  const h = tensored_frame.shape[0] as number;
  const w = tensored_frame.shape[1] as number;
  const scale = Math.min(1, MAX_DIM / Math.max(h, w));
  const newH = Math.max(Math.round(h * scale), 640);
  const newW = Math.max(Math.round(w * scale), 640);
  const resized = tensored_frame.resizeBilinear([newH, newW]) as tf.Tensor3D;
  tensored_frame.dispose();
  tensored_frame = resized;

  const results: number[] = [];
  for (const bbox of bboxes) {
    const sliceH = bbox[1][1] - bbox[0][1];
    const sliceW = bbox[1][0] - bbox[0][0];
    if (sliceH <= 0 || sliceW <= 0) continue;

    const cropped = tf.slice3d(tensored_frame, [bbox[0][1], bbox[0][0], 0], [sliceH, sliceW, 3]);
    const resizedCrop = cropped.resizeBilinear([48, 48]);
    cropped.dispose();
    const data = await resizedCrop.data();
    resizedCrop.dispose();

    console.log("Starting classify_larvae run");
    const output = await model.run([data]);
    console.log("Finished classify_larvae run");

    const prob = 1 / (1 + 2 ** (((output[0][1] as number) - (output[0][0] as number)) / 3));
    results.push(prob);
  }

  tensored_frame.dispose();
  return results;
}
