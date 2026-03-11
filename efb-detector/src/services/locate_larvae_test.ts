import { getBBoxes  } from "@/services/locate_larvae";
import { toByteArray } from 'base64-js';

import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';

const myImage = require('../../assets/test_image/istockphoto-1675490449-1024x1024.jpg');

export async function test() { // require returns a number (module ID)
    const [asset] = await Asset.loadAsync(myImage);
    const localUri = asset.localUri || asset.uri;
    const base64String = await FileSystem.readAsStringAsync(localUri, { encoding: FileSystem.EncodingType.Base64 });
    const rawFileBytes = toByteArray(base64String);
    console.log("hi")
    const result = await getBBoxes(rawFileBytes, [683, 1024, 3]);
    //console.log(result);
}