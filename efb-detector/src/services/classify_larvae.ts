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

  if (bboxes.length === 0) return [];

  // Convert bounding boxes to normalized [y1, x1, y2, x2] for tf.image.cropAndResize
  const boxesData = bboxes.map((bbox) => [
    Math.max(0, bbox[0][0]) / (frameH - 1),
    Math.max(0, bbox[0][1]) / (frameW - 1),
    Math.min(frameH - 1, bbox[1][0]) / (frameH - 1),
    Math.min(frameW - 1, bbox[1][1]) / (frameW - 1),
  ]);

  const boxesTensor = tf.tensor2d(boxesData, [bboxes.length, 4]);
  const boxIndices = tf.zeros([bboxes.length], "int32") as tf.Tensor1D;

  // Run a single cropAndResize operation instead of manual slicing in a loop
  const crops = tf.tidy(() => {
    return tf.image.cropAndResize(
      tensored_frame.expandDims(0) as tf.Tensor4D,
      boxesTensor,
      boxIndices,
      [48, 48]
    );
  });

  boxesTensor.dispose();
  boxIndices.dispose();

  // Extract all tensor data asynchronously at once for parallel efficiency
  const unstackedCrops = tf.unstack(crops, 0);
  const cropsData = await Promise.all(
    unstackedCrops.map(async (crop) => {
      const data = await crop.data();
      crop.dispose();
      return data;
    })
  );
  crops.dispose();

  const results: number[] = [];
  for (const data of cropsData) {
    console.log("Starting classify_larvae run");
    const output = await model.run([data]);
    console.log("Finished classify_larvae run");

    const prob = 1 / (1 + 2 ** (((output[0][1] as number) - (output[0][0] as number)) / 3));
    results.push(prob);
  }

  return results;
}
