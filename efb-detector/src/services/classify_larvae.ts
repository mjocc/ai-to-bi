import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";
import * as tf from "@tensorflow/tfjs";

let classifierModel: TensorflowModel | null = null;

export async function preloadClassifierModel(): Promise<void> {
  classifierModel = await loadTensorflowModel(
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require("../../assets/models/disease_classifier_float16.tflite")
  );
}

export async function processImageWithBBoxes(
  tensored_frame: tf.Tensor3D,
  bboxes: number[][][]
) {
  const model =
    (classifierModel ??= await loadTensorflowModel(
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      require("../../assets/models/disease_classifier_float16.tflite")
    ));

  const frameH = tensored_frame.shape[0] as number;
  const frameW = tensored_frame.shape[1] as number;

  const results: number[] = [];
  for (const bbox of bboxes) {
    const y1 = Math.max(0, bbox[0][0]);
    const x1 = Math.max(0, bbox[0][1]);
    const y2 = Math.min(frameH, bbox[1][0]);
    const x2 = Math.min(frameW, bbox[1][1]);
    const sliceH = y2 - y1;
    const sliceW = x2 - x1;
    if (sliceH <= 0 || sliceW <= 0) continue;

    const cropped = tf.slice3d(tensored_frame, [y1, x1, 0], [sliceH, sliceW, 3]);
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

  return results;
}
