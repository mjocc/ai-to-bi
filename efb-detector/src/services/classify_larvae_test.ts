import { processImageWithBBoxes  } from "@/services/classify_larvae";
import { toByteArray } from 'base64-js';

import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';

const myImage = require('../../assets/test_image/EFB-infected-larvae.jpg');

export async function test() { // require returns a number (module ID)
    const [asset] = await Asset.loadAsync(myImage);
    const localUri = asset.localUri || asset.uri;
    const base64String = await FileSystem.readAsStringAsync(localUri, { encoding: FileSystem.EncodingType.Base64 });
    const rawFileBytes = toByteArray(base64String);
    const result = await processImageWithBBoxes(rawFileBytes, [383, 680, 3], [[[76, 18], [120, 59]], [[120, 1], [175, 61]], [[172, 5], [228, 60]], [[276, 8], [333, 60]], [[376, 1], [441, 61]], [[542, 5], [607, 57]], [[44, 55], [98, 106]], [[95, 54], [149, 107]], [[18, 102], [67, 153]], [[283, 103], [331, 152]], [[337, 101], [386, 151]], [[438, 96], [496, 147]], [[522, 139], [575, 197]], [[15, 187], [66, 243]], [[331, 191], [387, 243]], [[389, 192], [441, 246]], [[440, 191], [492, 244]], [[513, 230], [577, 289]], [[253, 237], [306, 293]], [[283, 286], [336, 335]], [[90, 323], [147, 376]], [[149, 323], [205, 378]], [[624, 325], [678, 377]]]);
    const rough_probs_from_python = [0.35958045721054077, 0.18697254359722137, 0.8099425435066223, 0.3031874895095825, 0.8306248188018799, 0.2674219608306885, 0.30785250663757324, 0.17915509641170502, 0.750471830368042, 0.8079157471656799, 0.21936485171318054, 0.14936678111553192, 0.713472843170166, 0.8975349068641663, 0.9410672783851624, 0.48290789127349854, 0.8102576732635498, 0.3792813718318939, 0.24091945588588715, 0.2801132798194885, 0.2094847410917282, 0.8670417666435242, 0.12366727739572525];
    console.log("Resulting probabilities for image:")
    for (var i = 0; i < result.length; i++) {
        console.log("Produced probability: " + result[i] + ", Expected probability from Python: " + rough_probs_from_python[i])
    }
}