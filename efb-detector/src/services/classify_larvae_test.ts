import { loadTensorflowModel, useTensorflowModel } from "react-native-fast-tflite";
import { processImageWithBBoxes  } from "@/services/classify_larvae";
import { decodeJpeg } from '@tensorflow/tfjs-react-native';
import { Image } from 'react-native';
import { readFile } from 'react-native-fs';
import { toByteArray } from 'base64-js';

export async function test() {
    const imageId = require('../../assets/test_image/EFB-infected-larvae.jpg');
    const assetUri = Image.resolveAssetSource(imageId).uri;
    const base64String = await readFile(assetUri, 'base64');
    const rawFileBytes = toByteArray(base64String);
    const tensor = decodeJpeg(rawFileBytes);
    const typed_arr = await tensor.data()
    processImageWithBBoxes(typed_arr, [[2, 2], [2, 2]]);
}