import { CameraView, useCameraPermissions } from "expo-camera";
import { useRouter } from "expo-router";
import * as React from "react";
import { useRef, useState } from "react";
import { StyleSheet, Text, Pressable, View, ActivityIndicator } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { stitchVideo } from "../../services/stitcherApi";
import { useBeeStore } from "../../store/useBeeStore";

type RecordingState = "idle" | "recording" | "processing";

export default function CaptureScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [error, setError] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);
  const recordingPromise = useRef<Promise<{ uri: string } | undefined> | null>(null);
  const router = useRouter();
  const { addStitchedScan, referencePhotoUri } = useBeeStore();

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

  const startRecording = () => {
    if (!cameraRef.current || recordingState !== "idle") return;
    setError(null);
    setRecordingState("recording");

    // recordAsync resolves with { uri } once stopRecording() is called.
    recordingPromise.current = cameraRef.current.recordAsync({ maxDuration: 30 });
  };

  const stopAndProcess = async () => {
    if (!cameraRef.current || recordingState !== "recording") return;
    setRecordingState("processing");

    try {
      cameraRef.current.stopRecording();

      const video = await recordingPromise.current;
      recordingPromise.current = null;

      if (!video?.uri) throw new Error("No video URI returned from camera.");

      // Pass reference photo URI if one was captured
      const panoramaUri = await stitchVideo(
        video.uri,
        referencePhotoUri ?? undefined,
      );

      addStitchedScan({ panoramaUri });

      router.replace("/capture/results");
    } catch (e: any) {
      console.error("Stitching failed:", e);
      setError(e?.message ?? "Stitching failed. Please try again.");
      setRecordingState("idle");
    }
  };

  const isProcessing = recordingState === "processing";
  const isRecording = recordingState === "recording";

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFillObject}
        facing="back"
        mode="video"
        mute
      />

      <SafeAreaView style={styles.overlay} pointerEvents="box-none">
        {/* Top bar */}
        <View style={styles.topBar}>
          {!isProcessing && (
            <Pressable
              style={styles.backButton}
              onPress={() => router.push("/capture")}
              disabled={isRecording}
            >
              <Text style={[styles.backButtonText, isRecording && styles.dimmed]}>
                Cancel
              </Text>
            </Pressable>
          )}
          {isRecording && (
            <View style={styles.recordingIndicator}>
              <View style={styles.recordingDot} />
              <Text style={styles.recordingText}>Recording</Text>
            </View>
          )}
        </View>

        {/* Instruction */}
        {!isProcessing && (
          <View style={styles.instructionWrap} pointerEvents="none">
            <Text style={styles.instructionText}>
              {isRecording
                ? "Pan slowly across the hive frame, then tap Stop."
                : "Tap Record and pan across the hive frame."}
            </Text>
          </View>
        )}

        {/* Processing overlay */}
        {isProcessing && (
          <View style={styles.processingOverlay} pointerEvents="none">
            <ActivityIndicator size="large" color="#E9B44C" />
            <Text style={styles.processingText}>Stitching panorama…</Text>
            <Text style={styles.processingSubText}>This may take a few seconds.</Text>
          </View>
        )}

        {/* Error */}
        {error && (
          <View style={styles.errorBanner}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {/* Bottom controls */}
        {!isProcessing && (
          <View style={styles.bottomBar}>
            {!isRecording ? (
              <Pressable style={styles.captureOuter} onPress={startRecording}>
                <View style={styles.captureInner} />
              </Pressable>
            ) : (
              <Pressable style={styles.stopOuter} onPress={stopAndProcess}>
                <View style={styles.stopInner} />
              </Pressable>
            )}
            <Text style={styles.buttonLabel}>
              {isRecording ? "Stop" : "Record"}
            </Text>
          </View>
        )}
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

  // Top
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
  dimmed: { opacity: 0.4 },

  recordingIndicator: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 20,
  },
  recordingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "#E74C3C",
    marginRight: 7,
  },
  recordingText: { color: "#fff", fontWeight: "700", fontSize: 14 },

  // Instruction
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

  // Processing
  processingOverlay: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 12,
  },
  processingText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },
  processingSubText: {
    color: "rgba(255,255,255,0.7)",
    fontSize: 13,
  },

  // Error
  errorBanner: {
    marginHorizontal: 24,
    backgroundColor: "rgba(200,50,50,0.85)",
    borderRadius: 10,
    padding: 12,
  },
  errorText: { color: "#fff", fontSize: 14, textAlign: "center" },

  // Bottom
  bottomBar: { paddingBottom: 40, alignItems: "center", gap: 8 },
  buttonLabel: {
    color: "rgba(255,255,255,0.8)",
    fontSize: 13,
    fontWeight: "600",
  },

  // Record button (start)
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

  // Stop button
  stopOuter: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: "#E74C3C",
    justifyContent: "center",
    alignItems: "center",
  },
  stopInner: {
    width: 32,
    height: 32,
    borderRadius: 4,
    backgroundColor: "#E74C3C",
  },

  // Permission
  permissionText: { fontSize: 16, marginBottom: 20, textAlign: "center" },
  grantButton: { backgroundColor: "#E9B44C", padding: 15, borderRadius: 8 },
  grantButtonText: { color: "#fff", fontWeight: "bold" },
});