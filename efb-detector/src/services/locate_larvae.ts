import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";
import * as tf from "@tensorflow/tfjs";

let locatorModel: TensorflowModel | null = null;

export async function preloadLocatorModel(): Promise<void> {
  locatorModel = await loadTensorflowModel(
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require("../../assets/models/larvae_locator_float32.tflite")
  );
}

export async function getBBoxes(tensored_frame: tf.Tensor3D) {
  const model =
    (locatorModel ??= await loadTensorflowModel(
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      require("../../assets/models/larvae_locator_float32.tflite")
    ));

  const out_bboxes: number[][][] = [];

  for (let start_y = 0; start_y < (tensored_frame.shape[0] as number); start_y += 540) {
    let done1 = false;
    if (start_y > (tensored_frame.shape[0] as number) - 640) {
      start_y = (tensored_frame.shape[0] as number) - 640;
      done1 = true;
    }
    for (let start_x = 0; start_x < (tensored_frame.shape[1] as number); start_x += 540) {
      let done2 = false;
      if (start_x > (tensored_frame.shape[1] as number) - 640) {
        start_x = (tensored_frame.shape[1] as number) - 640;
        done2 = true;
      }

      const cropped = tf.slice3d(tensored_frame, [start_y, start_x, 0], [640, 640, 3]);
      const data = (await cropped.data()).slice();
      cropped.dispose();

      console.log("Started locate_larvae run");
      const output = await model.run([data]);
      console.log("Finished locate_larvae run");

      const stride = Math.round(output[0].length / 5);
      for (let i = 0; i < stride; i++) {
        const prob = output[0][i + 4 * stride];
        if (prob > 0.7) {
          const bbox = [
            [Math.round(output[0][i] as number * 640), Math.round(output[0][i + stride] as number * 640)],
            [Math.round(output[0][i + 2 * stride] as number * 640), Math.round(output[0][i + 3 * stride] as number * 640)],
          ];
          out_bboxes.push([
            [bbox[0][1] - Math.round(bbox[1][1] / 2) + start_y, bbox[0][0] - Math.round(bbox[1][0] / 2) + start_x],
            [bbox[0][1] + Math.round(bbox[1][1] / 2) + start_y, bbox[0][0] + Math.round(bbox[1][0] / 2) + start_x],
          ]);
        }
      }

      if (done2) break;
    }
    if (done1) break;
  }

  // Merge overlapping boxes (iterative NMS)
  const removed = new Set<number>();
  const new_bboxes: number[][][] = [];

  for (let i = 0; i < out_bboxes.length; i++) {
    if (removed.has(i)) continue;
    let current = out_bboxes[i];
    for (let j = i + 1; j < out_bboxes.length; j++) {
      if (removed.has(j)) continue;
      const area_1 = (current[1][0] - current[0][0]) * (current[1][1] - current[0][1]);
      const area_2 =
        (out_bboxes[j][1][0] - out_bboxes[j][0][0]) *
        (out_bboxes[j][1][1] - out_bboxes[j][0][1]);
      const intersection_bbox = [
        [Math.max(current[0][0], out_bboxes[j][0][0]), Math.max(current[0][1], out_bboxes[j][0][1])],
        [Math.min(current[1][0], out_bboxes[j][1][0]), Math.min(current[1][1], out_bboxes[j][1][1])],
      ];
      let area_3 =
        (intersection_bbox[1][0] - intersection_bbox[0][0]) *
        (intersection_bbox[1][1] - intersection_bbox[0][1]);
      if (
        intersection_bbox[1][0] < intersection_bbox[0][0] ||
        intersection_bbox[1][1] < intersection_bbox[0][1]
      ) {
        area_3 = 0;
      }
      if (area_3 > Math.min(area_1, area_2) * 0.5) {
        current = [
          [Math.min(current[0][0], out_bboxes[j][0][0]), Math.min(current[0][1], out_bboxes[j][0][1])],
          [Math.max(current[1][0], out_bboxes[j][1][0]), Math.max(current[1][1], out_bboxes[j][1][1])],
        ];
        removed.add(j);
      }
    }
    new_bboxes.push(current);
  }

  return new_bboxes;
}
