import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";
import * as tf from "@tensorflow/tfjs";

let locatorModel: TensorflowModel | null = null;

export async function preloadLocatorModel(): Promise<void> {
  locatorModel = await loadTensorflowModel(
    require("../../assets/models/larvae_locator_float16.tflite")
  );
}

const MAX_DIM = 1280;

export async function getBBoxes(
  rawImageData: { data: Uint8Array; width: number; height: number }
) {
  const model =
    locatorModel ??
    (await loadTensorflowModel(
      require("../../assets/models/larvae_locator_float16.tflite")
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

  // Cap to MAX_DIM on the longest side to reduce crop count
  let tensored_frame = tf.tensor3d(rgb, [height, width, 3], "float32");
  const h = tensored_frame.shape[0] as number;
  const w = tensored_frame.shape[1] as number;
  const scale = Math.min(1, MAX_DIM / Math.max(h, w));
  const newH = Math.max(Math.round(h * scale), 640);
  const newW = Math.max(Math.round(w * scale), 640);
  const resized = tensored_frame.resizeBilinear([newH, newW]) as tf.Tensor3D;
  tensored_frame.dispose();
  tensored_frame = resized;

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
      const data = await cropped.data();
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

  tensored_frame.dispose();

  // Merge overlapping boxes
  const new_bboxes: number[][][] = [];
  const removed: number[] = [];
  for (let i = 0; i < out_bboxes.length; i++) {
    for (let j = i + 1; j < out_bboxes.length; j++) {
      const bbox1 = out_bboxes[i];
      const bbox2 = out_bboxes[j];
      const area_1 = (bbox1[1][0] - bbox1[0][0]) * (bbox1[1][1] - bbox1[0][1]);
      const area_2 = (bbox2[1][0] - bbox2[0][0]) * (bbox2[1][1] - bbox2[0][1]);
      const intersection_bbox = [
        [Math.max(bbox1[0][0], bbox2[0][0]), Math.max(bbox1[0][1], bbox2[0][1])],
        [Math.min(bbox1[1][0], bbox2[1][0]), Math.min(bbox1[1][1], bbox2[1][1])],
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
        new_bboxes.push([
          [Math.min(bbox1[0][0], bbox2[0][0]), Math.min(bbox1[0][1], bbox2[0][1])],
          [Math.max(bbox1[0][0], bbox2[0][0]), Math.max(bbox1[0][1], bbox2[0][1])],
        ]);
        removed.push(i);
        removed.push(j);
      }
    }
    if (!removed.includes(i)) {
      new_bboxes.push(out_bboxes[i]);
    }
  }

  return new_bboxes;
}
