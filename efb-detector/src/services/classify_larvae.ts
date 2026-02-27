import { loadTensorflowModel } from "react-native-fast-tflite";
import * as tf from '@tensorflow/tfjs';
import * as jpeg from 'jpeg-js';

export async function processImageWithBBoxes(frame: tf.TypedArray, image_shape: [number, number, number], bboxes: number[][][]) {
    const rawImageData = jpeg.decode(frame, { useTArray: true });
    const model = await loadTensorflowModel(require('../../assets/models/disease_classifier_float16.tflite'));
    let newframe = rawImageData.data
    var newNewFrame = []
    for (var i = 0; i < newframe.length; i++) {
        if (i%4 != 3) {
            newNewFrame.push(newframe[i])
        }
    }
    if (model == null) {
        return [];  
    }
    var tensored_frame = tf.tensor3d(newNewFrame, image_shape, 'int32')
    tensored_frame = tf.div(tensored_frame, 255);
    var results = []
    for (var bbox of bboxes) {
        let cropped_frame = tf.slice3d(tensored_frame, [bbox[0][1], bbox[0][0], 0], [bbox[1][1]-bbox[0][1], bbox[1][0]-bbox[0][0], 3])
        cropped_frame = cropped_frame.resizeBilinear([48, 48])
        let output = await model.run([await cropped_frame.data()])
        let prob = 1/(1+2**((<number>output[0][1]-<number>output[0][0])/3))
        results.push(prob)
    }
    return results
}