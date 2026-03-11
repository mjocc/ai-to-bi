import { loadTensorflowModel } from "react-native-fast-tflite";
import * as tf from '@tensorflow/tfjs';
import * as jpeg from 'jpeg-js';

export async function getBBoxes(frame: tf.TypedArray, image_shape: [number, number, number]) {
    const rawImageData = jpeg.decode(frame, { useTArray: true });
    const model = await loadTensorflowModel(require('../../assets/models/larvae_locator_float16.tflite'));
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
    // resize
    tensored_frame = tensored_frame.resizeBilinear([Math.max(tensored_frame.shape[0], 640), Math.max(tensored_frame.shape[1], 640)]);
    //console.log(model)
    //console.log(tensored_frame.shape)
    //console.log(await tensored_frame.data())
    var out_bboxes = []
    console.log("ho")
    for (var start_y = 0; start_y < tensored_frame.shape[0]; start_y += 540) {
        var done1 = false
        if (start_y > tensored_frame.shape[0] - 640) {
            start_y = tensored_frame.shape[0] - 640
            done1 = true;
        }
        for (var start_x = 0; start_x < tensored_frame.shape[1]; start_x += 540) {
            var done2 = false
            if (start_x > tensored_frame.shape[1] - 640) {
                start_x = tensored_frame.shape[1] - 640
                done2 = true;
            }
            let cropped_frame = tf.slice3d(tensored_frame, [start_y, start_x, 0], [640, 640, 3])

            //let sample_im = cropped_frame.resizeBilinear([32, 32]);
    
            let output = await model.run([await cropped_frame.data()])
            //console.log(output[0].slice(0, 100))
            for (var i = 0; i < Math.round(output[0].length/5); i += 1) {
                let bbox = [[Math.round(<number>output[0][i]*640), Math.round(<number>output[0][i+1*Math.round(output[0].length/5)]*640)], [Math.round(<number>output[0][i+2*Math.round(output[0].length/5)]*640), Math.round(<number>output[0][i+3*Math.round(output[0].length/5)]*640)]]
                let prob = output[0][i+4*Math.round(output[0].length/5)]
                if (prob > 0.7) {
                    //console.log(i)
                    //console.log("Bounding box: " + bbox + ", Probability of containing larvae: " + prob)
                    let bboxs = [[bbox[0][1]-Math.round(bbox[1][1]/2) + start_y, bbox[0][0]-Math.round(bbox[1][0]/2) + start_x], [bbox[0][1]+Math.round(bbox[1][1]/2) + start_y, bbox[0][0]+Math.round(bbox[1][0]/2) + start_x]]
                    //let cropped_frame1 = tf.slice3d(tensored_frame, [bboxs[0][0], bboxs[0][1], 0], [bboxs[1][0]-bboxs[0][0], bboxs[1][1]-bboxs[0][1], 3])
                    //cropped_frame1 = cropped_frame1.resizeBilinear([48, 48])
                    //console.log(await cropped_frame1.data())
                    out_bboxes.push(bboxs)
                }
            }
            if (done2) {
                break
            }
        }
        if (done1) {
            break
        }
    }
    var new_bboxes = []
    var removed = []
    for (let i = 0; i < out_bboxes.length; i++) {
        for (let j = i+1; j < out_bboxes.length; j++) {
            var bbox1 = out_bboxes[i]
            var bbox2 = out_bboxes[j]
            var area_1 = ((bbox1[1][0]-bbox1[0][0]) * (bbox1[1][1]-bbox1[0][1]))
            var area_2 = ((bbox2[1][0]-bbox2[0][0]) * (bbox2[1][1]-bbox2[0][1]))
            var intersection_bbox = [[Math.max(bbox1[0][0], bbox2[0][0]), Math.max(bbox1[0][1], bbox2[0][1])], [Math.min(bbox1[0][0], bbox2[0][0]), Math.min(bbox1[0][1], bbox2[0][1])]]
            var area_3 = ((intersection_bbox[1][0]-intersection_bbox[0][0]) * (intersection_bbox[1][1]-intersection_bbox[0][1]))
            if ((intersection_bbox[1][0] < intersection_bbox[0][0]) || (intersection_bbox[1][1] < intersection_bbox[0][1])) {
                area_3 = 0
            }
            else {
                //console.log(area_3)
            }
            if (area_3 > Math.min(area_1, area_2) * 0.5) {
                var new_box = [[Math.min(bbox1[0][0], bbox2[0][0]), Math.min(bbox1[0][1], bbox2[0][1])], [Math.max(bbox1[0][0], bbox2[0][0]), Math.max(bbox1[0][1], bbox2[0][1])]]
                new_bboxes.push(new_box)
                removed.push(i)
                removed.push(j)
            }
        }
        if (!(i in removed)) {
            new_bboxes.push(out_bboxes[i])
        }

    }
    console.log(new_bboxes)
    return 0;
}