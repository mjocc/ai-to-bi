import {
  CameraView,
  CameraCapturedPicture,
  useCameraPermissions,
} from "expo-camera";
import { useRouter } from "expo-router";
import React, { useRef, useState, useEffect } from "react";
import {
  ActivityIndicator,
  Animated,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import RNFS from "react-native-fs";
import { useBeeStore } from "@/store/useBeeStore";
import { runMLPipeline } from "@/services/mlService";
import { VolumeManager } from "react-native-volume-manager";

const QUADRANT_LABELS = [
  "Top Left",
  "Top Right",
  "Bottom Left",
  "Bottom Right",
];
const CELL_SIZE = 60;

export default function CaptureScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const router = useRouter();
  const cameraRef = useRef<CameraView>(null);
  const { addImage, addScan, initializeData } = useBeeStore();

  const [hiveNo, setHiveNo] = useState(1);
  const [quadrantIndex, setQuadrantIndex] = useState(0);
  const [capturedPhotos, setCapturedPhotos] = useState<CameraCapturedPicture[]>(
    []
  );
  const [isProcessing, setIsProcessing] = useState(false);

  const pulseAnim = useRef(new Animated.Value(1)).current;
  const lastVolumePress = useRef(0);
  const handleCaptureRef = useRef<() => Promise<void>>(async () => {});

  useEffect(() => {
    initializeData();
  }, []);

  useEffect(() => {
    VolumeManager.showNativeVolumeUI({ enabled: false });

    const subscription = VolumeManager.addVolumeListener(() => {
      const now = Date.now();
      if (now - lastVolumePress.current > 300) {
        lastVolumePress.current = now;
        handleCaptureRef.current();
      }
    });

    return () => {
      VolumeManager.showNativeVolumeUI({ enabled: true });
      subscription.remove();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (isProcessing) return;
    pulseAnim.setValue(1);
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 0.3,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
      ])
    );
    anim.start();
    return () => anim.stop();
  }, [quadrantIndex, isProcessing]);

  const processPipeline = async (photos: CameraCapturedPicture[]) => {
    try {
      console.log("Starting ML pipeline with photos:", photos);
      const scansDir = `${RNFS.DocumentDirectoryPath}/scans`;
      const date = new Date();
      const dateStr = date.toLocaleDateString();
      const timestamp = Date.now();
      const imageIds: number[] = [];
      const confidences: number[] = [];

      for (let i = 0; i < photos.length; i++) {
        const photo = photos[i];
        const fileName = `${timestamp}_q${i + 1}.jpg`;
        const srcPath = photo.uri.startsWith("file://")
          ? photo.uri.slice(7)
          : photo.uri;
        await RNFS.copyFile(srcPath, `${scansDir}/${fileName}`);

        const imageId = addImage({
          ImageFileName: fileName,
          ImageName: `Scan ${dateStr} Q${i + 1}`,
          DateTaken: date.toISOString(),
          xco1: 0,
          yco1: 0,
          xco2: 0,
          yco2: 0,
        });
        imageIds.push(imageId);

        console.log("Calling runMLPipeline for image:", fileName);
        const confidence = await runMLPipeline(
          photo.uri,
          photo.width,
          photo.height
        );
        confidences.push(confidence);
      }

      const maxIndex = confidences.indexOf(Math.max(...confidences));
      for (let i = 0; i < imageIds.length; i++) {
        if (i === maxIndex) continue;
        addScan({
          HiveNo: hiveNo,
          ImageID: imageIds[i],
          Confidence: confidences[i],
        });
      }
      addScan({
        HiveNo: hiveNo,
        ImageID: imageIds[maxIndex],
        Confidence: confidences[maxIndex],
      });

      router.replace("/capture/results");
    } catch (err) {
      console.error("ML pipeline error:", err);
      setIsProcessing(false);
    }
  };

  const handleCapture = async () => {
    if (!cameraRef.current || isProcessing) return;
    const photo = await cameraRef.current.takePictureAsync();
    if (!photo) return;

    const newPhotos = [...capturedPhotos, photo];
    if (newPhotos.length < 4) {
      setCapturedPhotos(newPhotos);
      setQuadrantIndex(newPhotos.length);
    } else {
      setCapturedPhotos(newPhotos);
      setIsProcessing(true);
      processPipeline(newPhotos);
    }
  };
  handleCaptureRef.current = handleCapture;

  if (!permission) return <View style={styles.center} />;

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.permissionText}>
          We need camera access to scan frames.
        </Text>
        <Pressable style={styles.grantButton} onPress={requestPermission}>
          <Text style={styles.grantButtonText}>Grant Permission</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFillObject}
        facing="back"
      />

      <SafeAreaView style={styles.overlay} pointerEvents="box-none">
        <View style={styles.topBar}>
          <Pressable
            style={styles.backButton}
            onPress={() => !isProcessing && router.back()}
            disabled={isProcessing}
          >
            <Text style={styles.backButtonText}>Cancel</Text>
          </Pressable>
        </View>

        <View style={styles.bottomBar}>
          {/* Hive selector */}
          <View style={styles.hiveRow}>
            <Text style={styles.hiveLabel}>Hive number:</Text>
            <Pressable
              style={styles.hiveButton}
              onPress={() => setHiveNo((h) => Math.max(1, h - 1))}
            >
              <Text style={styles.hiveButtonText}>−</Text>
            </Pressable>
            <Text style={styles.hiveNumber}>{hiveNo}</Text>
            <Pressable
              style={styles.hiveButton}
              onPress={() => setHiveNo((h) => h + 1)}
            >
              <Text style={styles.hiveButtonText}>+</Text>
            </Pressable>
          </View>

          {/* Quadrant label */}
          <Text style={styles.quadrantLabel}>
            Photographing: {QUADRANT_LABELS[quadrantIndex]}
          </Text>

          {/* Quadrant progress grid */}
          <View style={styles.quadrantGrid}>
            {[0, 1, 2, 3].map((i) => {
              const isDone = i < quadrantIndex;
              const isCurrent = i === quadrantIndex;
              return (
                <Animated.View
                  key={i}
                  style={[
                    styles.quadrantCell,
                    isDone && styles.quadrantDone,
                    isCurrent && styles.quadrantCurrent,
                    !isDone && !isCurrent && styles.quadrantPending,
                    isCurrent && { opacity: pulseAnim },
                  ]}
                >
                  {isDone && <Text style={styles.doneCheck}>✓</Text>}
                </Animated.View>
              );
            })}
          </View>

          {/* Capture button */}
          <Pressable
            style={styles.captureOuter}
            onPress={handleCapture}
            disabled={isProcessing}
          >
            <View style={styles.captureInner} />
          </Pressable>
        </View>
      </SafeAreaView>

      {/* Processing overlay */}
      {isProcessing && (
        <View style={styles.processingOverlay}>
          <ActivityIndicator size="large" color="#E9B44C" />
          <Text style={styles.processingText}>Analysing frame…</Text>
        </View>
      )}
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

  topBar: { padding: 20, alignItems: "flex-start" },
  backButton: {
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  backButtonText: { color: "#fff", fontSize: 16, fontWeight: "600" },

  bottomBar: {
    paddingBottom: 40,
    paddingTop: 16,
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    gap: 12,
  },

  hiveRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  hiveLabel: { color: "#fff", fontSize: 16, fontWeight: "600" },
  hiveButton: {
    width: 44,
    height: 44,
    backgroundColor: "rgba(233,180,76,0.85)",
    borderRadius: 22,
    justifyContent: "center",
    alignItems: "center",
  },
  hiveButtonText: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "700",
    lineHeight: 28,
  },
  hiveNumber: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "700",
    minWidth: 30,
    textAlign: "center",
  },

  quadrantLabel: {
    color: "#E9B44C",
    fontSize: 15,
    fontWeight: "600",
  },

  quadrantGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    width: CELL_SIZE * 2 + 4,
    borderRadius: 6,
    overflow: "hidden",
    borderWidth: 2,
    borderColor: "#E9B44C",
  },
  quadrantCell: {
    width: CELL_SIZE,
    height: CELL_SIZE,
    justifyContent: "center",
    alignItems: "center",
  },
  quadrantDone: { backgroundColor: "#E9B44C" },
  quadrantCurrent: { backgroundColor: "#E9B44C" },
  quadrantPending: { backgroundColor: "rgba(80,80,80,0.5)" },
  doneCheck: { color: "#fff", fontSize: 26, fontWeight: "700" },

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

  processingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.75)",
    justifyContent: "center",
    alignItems: "center",
    gap: 16,
  },
  processingText: { color: "#fff", fontSize: 18, fontWeight: "600" },

  permissionText: { fontSize: 16, marginBottom: 20, textAlign: "center" },
  grantButton: { backgroundColor: "#E9B44C", padding: 15, borderRadius: 8 },
  grantButtonText: { color: "#fff", fontWeight: "bold" },
});
