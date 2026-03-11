import { getBBoxes  } from "@/services/locate_larvae";
import { toByteArray } from 'base64-js';
import * as jpeg from 'jpeg-js';
import * as tf from "@tensorflow/tfjs";

import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';

const myImage = require('../../assets/test_image/istockphoto-1675490449-1024x1024.jpg');
const MAX_DIM = 1280;

export async function test() { // require returns a number (module ID)
    const [asset] = await Asset.loadAsync(myImage);
    const localUri = asset.localUri || asset.uri;
    const base64String = await FileSystem.readAsStringAsync(localUri, { encoding: FileSystem.EncodingType.Base64 });
    const rawFileBytes = toByteArray(base64String);
    const rawImageData = jpeg.decode(rawFileBytes, { useTArray: true });

    const { data: rgba, width, height } = rawImageData;
    const rgb = new Float32Array(width * height * 3);
    let j = 0;
    for (let i = 0; i < rgba.length; i += 4) {
      rgb[j++] = rgba[i] / 255;
      rgb[j++] = rgba[i + 1] / 255;
      rgb[j++] = rgba[i + 2] / 255;
    }

    let tensor = tf.tensor3d(rgb, [height, width, 3], "float32");
    const h = tensor.shape[0] as number;
    const w = tensor.shape[1] as number;
    const scale = Math.min(1, MAX_DIM / Math.max(h, w));
    const newH = Math.max(Math.round(h * scale), 640);
    const newW = Math.max(Math.round(w * scale), 640);
    const resized = tensor.resizeBilinear([newH, newW]) as tf.Tensor3D;
    tensor.dispose();

    const result = await getBBoxes(resized);
    resized.dispose();
    console.log(result);
}