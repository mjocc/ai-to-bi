import { getBBoxes  } from "@/services/locate_larvae";
import { toByteArray } from 'base64-js';
import * as jpeg from 'jpeg-js';

import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';

const myImage = require('../../assets/test_image/istockphoto-1675490449-1024x1024.jpg');

export async function test() { // require returns a number (module ID)
    const [asset] = await Asset.loadAsync(myImage);
    const localUri = asset.localUri || asset.uri;
    const base64String = await FileSystem.readAsStringAsync(localUri, { encoding: FileSystem.EncodingType.Base64 });
    const rawFileBytes = toByteArray(base64String);
    const rawImageData = jpeg.decode(rawFileBytes, { useTArray: true });
    const result = await getBBoxes(rawImageData);
    console.log(result);
}