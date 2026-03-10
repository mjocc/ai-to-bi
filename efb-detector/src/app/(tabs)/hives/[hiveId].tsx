import * as MediaLibrary from "expo-media-library";
import { Stack, useLocalSearchParams } from "expo-router";
import React, { useEffect, useState } from "react";
import {
  Alert,
  Dimensions,
  Image,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { LineChart } from "react-native-chart-kit";
import { getImageUri, useBeeStore as useStore } from "@/store/useBeeStore";

const screenWidth = Dimensions.get("window").width;
const IMAGE_SIZE = (screenWidth - 48) / 2;

const CHART_HEIGHT = 220;
const CHART_TOP_PADDING = 55;
const CHART_BOTTOM_PADDING = 30;
const THRESHOLD = 80;
const THRESHOLD_TOP =
  CHART_TOP_PADDING +
  (CHART_HEIGHT - CHART_TOP_PADDING - CHART_BOTTOM_PADDING) *
    (1 - THRESHOLD / 100);

type ChartData = {
  labels: string[];
  datasets: {
    data: number[];
    color?: (opacity: number) => string;
    strokeWidth?: number;
    withDots?: boolean;
  }[];
};

type ActiveView = "chart" | "images";
type SortOption = "name_asc" | "name_desc" | "date_asc" | "date_desc";

export default function HiveScreen() {
  const { hiveId } = useLocalSearchParams<{ hiveId: string }>();
  const hiveNo = Number(hiveId);

  const {
    getScansByHive,
    getImagesWithHive,
    updateImageName,
    initializeData,
    deleteImage,
  } = useStore();

  const [activeView, setActiveView] = useState<ActiveView>("chart");
  const [chartData, setChartData] = useState<ChartData>({
    labels: [],
    datasets: [{ data: [] }],
  });
  const [imageRecords, setImageRecords] = useState<
    (ReturnType<typeof getImagesWithHive>[number] & { Confidence: number })[]
  >([]);
  const [editingImage, setEditingImage] = useState<
    ReturnType<typeof getImagesWithHive>[number] | null
  >(null);
  const [editName, setEditName] = useState("");
  const [fullscreenImage, setFullscreenImage] = useState<
    ReturnType<typeof getImagesWithHive>[number] | null
  >(null);
  const [showSortModal, setShowSortModal] = useState(false);
  const [sortOption, setSortOption] = useState<SortOption>("date_desc");

  useEffect(() => {
    const setup = async () => {
      await initializeData();
      loadChart();
      loadImages();
    };
    setup();
  }, [hiveNo]);

  const loadChart = () => {
    const scans = getScansByHive(hiveNo);
    const labels = scans.map((s) => s.ImageName);
    const data = scans.map((s) => s.Confidence);
    setChartData({
      labels,
      datasets: [
        { data: data.length > 0 ? data : [0] },
        {
          data: [100],
          color: () => "transparent",
          withDots: false,
          strokeWidth: 0,
        },
      ],
    });
  };

  const loadImages = () => {
    const all = getImagesWithHive();
    const scans = getScansByHive(hiveNo);
    const filtered = all
      .filter((img) => img.HiveNo === hiveNo)
      .map((img) => ({
        ...img,
        Confidence:
          scans.find((s) => s.ImageID === img.ImageID)?.Confidence ?? 0,
      }));
    setImageRecords(filtered);
  };

  const openEditModal = (
    image: ReturnType<typeof getImagesWithHive>[number]
  ) => {
    setEditingImage(image);
    setEditName(image.ImageName);
  };

  const saveImageName = () => {
    if (!editingImage) return;
    if (!editName.trim()) {
      Alert.alert("Invalid name", "Image name cannot be empty.");
      return;
    }
    updateImageName(editingImage.ImageID, editName.trim());
    setEditingImage(null);
    loadImages();
    loadChart();
  };

  const saveToPhotos = async () => {
    if (!fullscreenImage) return;
    const { status } = await MediaLibrary.requestPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission denied",
        "Please allow photo library access in your device settings to save images."
      );
      return;
    }
    try {
      await MediaLibrary.saveToLibraryAsync(
        getImageUri(fullscreenImage.ImageFileName)
      );
      Alert.alert("Saved", "Image saved to your photo library.");
    } catch {
      Alert.alert("Error", "Failed to save image.");
    }
  };

  const hasWarning = chartData.datasets[0].data.some((v) => v > THRESHOLD);

  const sparseLabels =
    chartData.labels.length > 4
      ? chartData.labels.map((label, i, arr) => {
          if (i === 0) return label;
          if (i === arr.length - 1)
            return label.length > 6 ? label.slice(0, 6) + "…" : label;
          return "";
        })
      : chartData.labels;

  const sparseChartData = { ...chartData, labels: sparseLabels };

  const sortedImages = [...imageRecords].sort((a, b) => {
    switch (sortOption) {
      case "name_asc":
        return a.ImageName.localeCompare(b.ImageName);
      case "name_desc":
        return b.ImageName.localeCompare(a.ImageName);
      case "date_asc":
        return (
          new Date(a.DateTaken).getTime() - new Date(b.DateTaken).getTime()
        );
      case "date_desc":
        return (
          new Date(b.DateTaken).getTime() - new Date(a.DateTaken).getTime()
        );
      default:
        return 0;
    }
  });

  const deleteSpecificScan = (img: (typeof imageRecords)[number]) => {
    Alert.alert(
      "Delete Scan",
      `Delete "${img.ImageName}"? This cannot be undone.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: () => {
            deleteImage(img.ImageID);
            loadChart();
            loadImages();
          },
        },
      ]
    );
  };

  const clearOldestScan = () => {
    const scans = getScansByHive(hiveNo);
    if (scans.length === 0) return;
    Alert.alert(
      "Clear Oldest Scan",
      `Remove the oldest scan "${scans[0].ImageName}" from Hive ${hiveNo}?`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Remove",
          style: "destructive",
          onPress: () => {
            deleteImage(scans[0].ImageID);
            loadChart();
            loadImages();
          },
        },
      ]
    );
  };

  const clearAllScans = () => {
    const scans = getScansByHive(hiveNo);
    if (scans.length === 0) return;
    Alert.alert(
      "Clear All Hive Data",
      `Remove all ${scans.length} scan(s) for Hive ${hiveNo}? This cannot be undone.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Clear All",
          style: "destructive",
          onPress: () => {
            scans.forEach((s) => deleteImage(s.ImageID));
            loadChart();
            loadImages();
          },
        },
      ]
    );
  };

  const chartConfig = {
    backgroundColor: "#F6C24E",
    backgroundGradientFrom: "#F6C24E",
    backgroundGradientTo: "#E3892B",
    decimalPlaces: 0,
    color: (opacity = 1) => `rgba(255,255,255,${opacity})`,
    labelColor: (opacity = 1) => `rgba(255,255,255,${opacity})`,
  };

  return (
    <View style={styles.container}>
      <Stack.Screen options={{ title: `Hive ${hiveNo}` }} />

      {/* Chart view */}
      {activeView === "chart" && (
        <ScrollView
          style={{ flex: 1, width: "100%" }}
          contentContainerStyle={{ alignItems: "center", paddingBottom: 16 }}
        >
          <Text style={styles.title}>Hive {hiveNo} — Confidence Over Time</Text>

          {chartData.datasets[0].data.length === 0 ? (
            <Text style={styles.placeholder}>No scans yet for this hive.</Text>
          ) : (
            <View style={styles.chartWrapper}>
              {hasWarning && (
                <Text style={styles.warningText}>
                  ⚠️ One or more scans exceed the {THRESHOLD}% confidence
                  threshold!
                </Text>
              )}
              <LineChart
                data={sparseChartData}
                width={screenWidth - 32}
                height={CHART_HEIGHT}
                chartConfig={chartConfig}
                bezier
                fromZero
                style={{ borderRadius: 12 }}
              />
              {hasWarning && (
                <View style={[styles.thresholdLine, { top: THRESHOLD_TOP }]} />
              )}
            </View>
          )}

          {/* Data management buttons */}
          <View style={styles.clearButtonRow}>
            <TouchableOpacity
              style={styles.clearOldestButton}
              onPress={clearOldestScan}
            >
              <Text style={styles.clearButtonText}>Clear Oldest</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.clearAllButton}
              onPress={clearAllScans}
            >
              <Text style={styles.clearButtonText}>Clear All</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      )}

      {/* Images view */}
      {activeView === "images" && (
        <View style={styles.imagesContainer}>
          <View style={styles.sortRow}>
            <TouchableOpacity
              style={styles.sortButton}
              onPress={() => setShowSortModal(true)}
            >
              <Text style={styles.sortButtonText}>Sort ▼</Text>
            </TouchableOpacity>
          </View>
          <ScrollView contentContainerStyle={styles.imageGrid}>
            {sortedImages.length === 0 ? (
              <Text style={styles.placeholder}>
                No images for this hive yet.
              </Text>
            ) : (
              sortedImages.map((img) => (
                <View key={img.ImageID} style={styles.imageCard}>
                  <TouchableOpacity onPress={() => setFullscreenImage(img)}>
                    <Image
                      source={{ uri: getImageUri(img.ImageFileName) }}
                      style={styles.imageThumbnail}
                      resizeMode="cover"
                    />
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => openEditModal(img)}>
                    <Text style={styles.imageName}>{img.ImageName} ✎</Text>
                  </TouchableOpacity>
                  <Text style={styles.imageDate}>{img.DateTaken}</Text>
                  <Text
                    style={[
                      styles.imageConfidence,
                      img.Confidence > THRESHOLD &&
                        styles.imageConfidenceWarning,
                    ]}
                  >
                    {img.Confidence.toFixed(1)}% confidence
                  </Text>
                  <TouchableOpacity
                    style={styles.deleteCardButton}
                    onPress={() => deleteSpecificScan(img)}
                  >
                    <Text style={styles.deleteCardButtonText}>🗑 Delete</Text>
                  </TouchableOpacity>
                </View>
              ))
            )}
          </ScrollView>
        </View>
      )}

      {/* Fullscreen image*/}
      {fullscreenImage && (
        <Modal
          visible
          animationType="fade"
          onRequestClose={() => setFullscreenImage(null)}
        >
          <View style={styles.fullscreenContainer}>
            <TouchableOpacity
              style={styles.fullscreenBackButton}
              onPress={() => setFullscreenImage(null)}
            >
              <Text style={styles.fullscreenBackText}>← Back</Text>
            </TouchableOpacity>
            <Image
              source={{ uri: getImageUri(fullscreenImage.ImageFileName) }}
              style={styles.fullscreenImage}
              resizeMode="contain"
            />
            <Text style={styles.fullscreenName}>
              {fullscreenImage.ImageName}
            </Text>
            <Text style={styles.fullscreenMeta}>
              {fullscreenImage.DateTaken}
            </Text>
            <TouchableOpacity style={styles.saveButton} onPress={saveToPhotos}>
              <Text style={styles.saveButtonText}>⬇ Save to Photos</Text>
            </TouchableOpacity>
          </View>
        </Modal>
      )}

      {/* Rename*/}
      <Modal
        visible={!!editingImage}
        transparent
        animationType="fade"
        onRequestClose={() => setEditingImage(null)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalBox}>
            <Text style={styles.modalTitle}>Rename Image</Text>
            <TextInput
              style={styles.modalInput}
              value={editName}
              onChangeText={setEditName}
              autoFocus
              selectTextOnFocus
            />
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.modalCancel]}
                onPress={() => setEditingImage(null)}
              >
                <Text style={styles.modalButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButton, styles.modalSave]}
                onPress={saveImageName}
              >
                <Text style={styles.modalButtonText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Sort */}
      <Modal
        visible={showSortModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowSortModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalBox}>
            <Text style={styles.modalTitle}>Sort By</Text>
            {(
              [
                { value: "name_asc", label: "Name (A → Z)" },
                { value: "name_desc", label: "Name (Z → A)" },
                { value: "date_asc", label: "Date (Oldest first)" },
                { value: "date_desc", label: "Date (Newest first)" },
              ] as { value: SortOption; label: string }[]
            ).map((opt) => (
              <TouchableOpacity
                key={opt.value}
                style={[
                  styles.sortOption,
                  sortOption === opt.value && styles.sortOptionActive,
                ]}
                onPress={() => {
                  setSortOption(opt.value);
                  setShowSortModal(false);
                }}
              >
                <Text
                  style={[
                    styles.sortOptionText,
                    sortOption === opt.value && styles.sortOptionTextActive,
                  ]}
                >
                  {opt.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </Modal>

      {/* Bottom toggle */}
      <View style={styles.toggleContainer}>
        <TouchableOpacity
          style={[
            styles.toggleButton,
            activeView === "chart" && styles.toggleButtonActive,
          ]}
          onPress={() => setActiveView("chart")}
        >
          <Text
            style={[
              styles.toggleText,
              activeView === "chart" && styles.toggleTextActive,
            ]}
          >
            Chart
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.toggleButton,
            activeView === "images" && styles.toggleButtonActive,
          ]}
          onPress={() => setActiveView("images")}
        >
          <Text
            style={[
              styles.toggleText,
              activeView === "images" && styles.toggleTextActive,
            ]}
          >
            Images
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 60,
    alignItems: "center",
  },
  title: {
    fontSize: 18,
    fontWeight: "700",
    color: "#E3892B",
    marginBottom: 16,
    textAlign: "center",
  },
  chartWrapper: {
    position: "relative",
    marginVertical: 16,
  },
  thresholdLine: {
    position: "absolute",
    left: 0,
    right: 0,
    height: 2,
    backgroundColor: "red",
  },
  warningText: {
    color: "red",
    fontWeight: "700",
    fontSize: 13,
    textAlign: "center",
    marginBottom: 8,
    marginHorizontal: 16,
  },
  imagesContainer: {
    flex: 1,
    width: "100%",
    paddingHorizontal: 16,
  },
  imageGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 16,
    paddingBottom: 16,
  },
  imageCard: {
    width: IMAGE_SIZE,
    alignItems: "center",
  },
  imageThumbnail: {
    width: IMAGE_SIZE,
    height: IMAGE_SIZE,
    borderRadius: 10,
    backgroundColor: "#E0E0E0",
  },
  imageName: {
    marginTop: 6,
    fontSize: 13,
    fontWeight: "600",
    textAlign: "center",
    color: "#E3892B",
    textDecorationLine: "underline",
  },
  imageDate: {
    fontSize: 11,
    color: "#aaa",
    textAlign: "center",
    marginTop: 2,
  },
  imageConfidence: {
    fontSize: 11,
    color: "#aaa",
    textAlign: "center",
    marginTop: 2,
  },
  imageConfidenceWarning: {
    color: "red",
    fontWeight: "700",
  },
  deleteCardButton: {
    marginTop: 6,
    backgroundColor: "#D9534F",
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  deleteCardButtonText: {
    color: "#fff",
    fontSize: 11,
    fontWeight: "700",
  },
  sortRow: {
    flexDirection: "row",
    justifyContent: "flex-end",
    marginBottom: 10,
  },
  sortButton: {
    backgroundColor: "#F6C24E",
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 8,
  },
  sortButtonText: {
    fontWeight: "600",
    fontSize: 13,
    color: "#fff",
  },
  sortOption: {
    width: "100%",
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderRadius: 8,
    marginBottom: 4,
  },
  sortOptionActive: {
    backgroundColor: "#FFF3D6",
  },
  sortOptionText: {
    fontSize: 15,
    color: "#555",
    textAlign: "center",
  },
  sortOptionTextActive: {
    color: "#E3892B",
    fontWeight: "700",
  },
  clearButtonRow: {
    flexDirection: "row",
    gap: 12,
    marginTop: 24,
    marginBottom: 8,
    paddingHorizontal: 16,
  },
  clearOldestButton: {
    flex: 1,
    backgroundColor: "#E3892B",
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: "center",
  },
  clearAllButton: {
    flex: 1,
    backgroundColor: "#D9534F",
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: "center",
  },
  clearButtonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 14,
  },
  placeholder: {
    fontSize: 16,
    color: "#aaa",
    marginTop: 40,
    textAlign: "center",
  },
  fullscreenContainer: {
    flex: 1,
    backgroundColor: "#000",
    justifyContent: "center",
    alignItems: "center",
  },
  fullscreenBackButton: {
    position: "absolute",
    top: 60,
    left: 20,
    zIndex: 10,
    backgroundColor: "rgba(255,255,255,0.15)",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  fullscreenBackText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  fullscreenImage: {
    width: screenWidth,
    height: screenWidth,
  },
  fullscreenName: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "700",
    marginTop: 20,
    textAlign: "center",
  },
  saveButton: {
    marginTop: 20,
    backgroundColor: "#F6C24E",
    paddingVertical: 10,
    paddingHorizontal: 28,
    borderRadius: 10,
  },
  saveButtonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 15,
  },
  fullscreenMeta: {
    color: "#aaa",
    fontSize: 13,
    marginTop: 6,
    textAlign: "center",
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "center",
    alignItems: "center",
  },
  modalBox: {
    backgroundColor: "#fff",
    borderRadius: 14,
    padding: 24,
    width: screenWidth - 64,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 6,
  },
  modalTitle: {
    fontSize: 17,
    fontWeight: "700",
    marginBottom: 14,
    textAlign: "center",
  },
  modalInput: {
    borderWidth: 1,
    borderColor: "#E0E0E0",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 15,
    marginBottom: 16,
  },
  modalButtons: {
    flexDirection: "row",
    gap: 10,
  },
  modalButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: "center",
  },
  modalCancel: {
    backgroundColor: "#F0F0F0",
  },
  modalSave: {
    backgroundColor: "#F6C24E",
  },
  modalButtonText: {
    fontWeight: "600",
    fontSize: 15,
  },
  toggleContainer: {
    flexDirection: "row",
    backgroundColor: "#E0E0E0",
    borderRadius: 10,
    padding: 4,
    marginBottom: 30,
  },
  toggleButton: {
    paddingVertical: 8,
    paddingHorizontal: 28,
    borderRadius: 8,
  },
  toggleButtonActive: {
    backgroundColor: "#F6C24E",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.15,
    shadowRadius: 2,
    elevation: 2,
  },
  toggleText: {
    fontSize: 15,
    fontWeight: "600",
    color: "#888",
  },
  toggleTextActive: {
    color: "#fff",
  },
});
