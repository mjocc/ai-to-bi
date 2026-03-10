/**
 * capture/index.tsx
 *
 * Step 1 of the capture flow: take a single wide photo of the entire hive
 * frame before the close-up video scan.  The photo URI is stored in the bee
 * store and forwarded to the stitcher API as an alignment reference.
 *
 * Navigation:
 *   → capture/record  (the video recording screen)
 */

import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useRouter } from "expo-router";
import * as React from "react";
import { useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useBeeStore } from "../../store/useBeeStore";

export default function ReferenceScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const router = useRouter();
  const { setReferencePhoto } = useBeeStore();

  if (!permission) return <View style={styles.center} />;

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.permissionText}>
          We need camera access to capture the reference photo.
        </Text>
        <Pressable style={styles.grantButton} onPress={requestPermission}>
          <Text style={styles.grantButtonText}>Grant Permission</Text>
        </Pressable>
      </View>
    );
  }

  const takePhoto = async () => {
    if (!cameraRef.current || capturing) return;
    setCapturing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.85,
        exif: true,
      });

      if (!photo?.uri) return;

      // expo-camera always returns sensor-native (portrait) pixels regardless
      // of device orientation, but writes the correct EXIF orientation tag.
      // We rotate the image to match what was shown on screen.
      const orientation: number = photo.exif?.Orientation ?? 1;
      let rotation = 0;
      if (orientation === 3) rotation = 180;
      else if (orientation === 6) rotation = 90;
      else if (orientation === 8) rotation = 270;

      if (rotation !== 0) {
        const rotated = await manipulateAsync(
          photo.uri,
          [{ rotate: rotation }],
          { compress: 0.85, format: SaveFormat.JPEG },
        );
        setPhotoUri(rotated.uri);
      } else {
        setPhotoUri(photo.uri);
      }
    } catch (e) {
      console.error("Reference photo failed:", e);
    } finally {
      setCapturing(false);
    }
  };

  const retake = () => setPhotoUri(null);

  const proceed = () => {
    setReferencePhoto(photoUri ?? null);
    router.replace("/capture/record");
  };

  // ── preview state ──────────────────────────────────────────────────────────
  if (photoUri) {
    return (
      <View style={styles.container}>
        <Image
          source={{ uri: photoUri }}
          style={StyleSheet.absoluteFillObject}
          resizeMode="contain"
        />

        <SafeAreaView style={styles.overlay} pointerEvents="box-none">
          <View style={styles.topBar}>
            <Text style={styles.previewLabel}>Reference Photo</Text>
          </View>

          <View style={styles.bottomBar}>
            <Pressable style={styles.secondaryButton} onPress={retake}>
              <Text style={styles.secondaryButtonText}>Retake</Text>
            </Pressable>
            <Pressable style={styles.primaryButton} onPress={proceed}>
              <Text style={styles.primaryButtonText}>Use & Start Recording →</Text>
            </Pressable>
          </View>
        </SafeAreaView>
      </View>
    );
  }

  // ── viewfinder state ───────────────────────────────────────────────────────
  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFillObject}
        facing="back"
      />

      <SafeAreaView style={styles.overlay} pointerEvents="box-none">
        <View style={styles.topBar}>
          <Pressable style={styles.backButton} onPress={() => router.back()}>
            <Text style={styles.backButtonText}>Cancel</Text>
          </Pressable>
        </View>

        <View style={styles.instructionWrap} pointerEvents="none">
          <Text style={styles.instructionText}>
            Step back so the entire hive frame fits in the shot, then take a photo.
          </Text>
        </View>

        <View style={styles.bottomBar}>
          <Pressable
            style={styles.captureOuter}
            onPress={takePhoto}
            disabled={capturing}
          >
            {capturing ? (
              <ActivityIndicator color="#E9B44C" />
            ) : (
              <View style={styles.captureInner} />
            )}
          </Pressable>
          <Text style={styles.buttonLabel}>Photo</Text>

          <Pressable style={styles.skipButton} onPress={proceed}>
            <Text style={styles.skipButtonText}>Skip →</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#fff",
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "space-between",
  },
  topBar: {
    padding: 20,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  backButton: {
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  backButtonText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  previewLabel: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "700",
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 20,
  },
  instructionWrap: {
    alignItems: "center",
    paddingHorizontal: 40,
  },
  instructionText: {
    color: "rgba(255,255,255,0.85)",
    fontSize: 14,
    textAlign: "center",
    backgroundColor: "rgba(0,0,0,0.4)",
    borderRadius: 10,
    paddingHorizontal: 16,
    paddingVertical: 8,
    overflow: "hidden",
  },
  bottomBar: {
    paddingBottom: 40,
    alignItems: "center",
    gap: 8,
  },
  buttonLabel: {
    color: "rgba(255,255,255,0.8)",
    fontSize: 13,
    fontWeight: "600",
  },
  captureOuter: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: "#E9B44C",
    justifyContent: "center",
    alignItems: "center",
  },
  captureInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#E9B44C",
  },
  skipButton: {
    marginTop: 4,
    backgroundColor: "rgba(0,0,0,0.45)",
    paddingVertical: 8,
    paddingHorizontal: 20,
    borderRadius: 20,
  },
  skipButtonText: { color: "rgba(255,255,255,0.7)", fontSize: 14, fontWeight: "600" },
  secondaryButton: {
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
    width: "80%",
    alignItems: "center",
  },
  secondaryButtonText: { color: "#fff", fontWeight: "600", fontSize: 15 },
  primaryButton: {
    backgroundColor: "#E9B44C",
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
    width: "80%",
    alignItems: "center",
  },
  primaryButtonText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  permissionText: { fontSize: 16, marginBottom: 20, textAlign: "center", paddingHorizontal: 32 },
  grantButton: { backgroundColor: "#E9B44C", padding: 15, borderRadius: 8 },
  grantButtonText: { color: "#fff", fontWeight: "bold" },
});