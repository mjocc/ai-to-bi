import { loadTensorflowModel, useTensorflowModel } from "react-native-fast-tflite";
import type { Tensor } from "react-native-fast-tflite";
import * as tf from '@tensorflow/tfjs';

const classification = useTensorflowModel(require('../../assets/models/disease_classifier_float16.tflite'));
const model = classification.state === 'loaded' ? classification.model : undefined

export async function processImageWithBBoxes(frame: tf.TypedArray, bboxes: number[][]) {
    if (model == null) return;
    console.log(frame);
    const rescaled = await tf.div(frame, 255).data()
    model.runSync([rescaled]);
}