//I'm not sure if this was what was wanted but it should show the results of the scan
//Dodgy way of making it work by getting the results of the most recent scan saved to store
//So will need saving to the store beforehand
//Feel free to change / overwrite if you want

import { Stack, useRouter } from "expo-router";
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
import { SafeAreaView } from "react-native-safe-area-context";
import { getImageUri, useBeeStore as useStore } from "@/store/useBeeStore";

const screenWidth = Dimensions.get("window").width;
const THRESHOLD = 80;

export default function ResultsScreen() {
  const router = useRouter();
  const { getScansWithImageNames, images, updateImageName, initializeData } =
    useStore();

  const [editName, setEditName] = useState("");
  const [showEditModal, setShowEditModal] = useState(false);
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    initializeData().then(() => setInitialized(true));
  }, []);

  const scansWithNames = getScansWithImageNames();
  const latestScan =
    scansWithNames.length > 0
      ? scansWithNames[scansWithNames.length - 1]
      : null;

  const latestImage = latestScan
    ? images.find((img) => img.ImageID === latestScan.ImageID) ?? null
    : null;

  const openEditModal = () => {
    if (!latestImage) return;
    setEditName(latestImage.ImageName);
    setShowEditModal(true);
  };

  const saveName = () => {
    if (!latestImage) return;
    if (!editName.trim()) {
      Alert.alert("Invalid name", "Name cannot be empty.");
      return;
    }
    updateImageName(latestImage.ImageID, editName.trim());
    setShowEditModal(false);
  };

  const confidenceColor =
    latestScan && latestScan.Confidence > THRESHOLD ? "#D9534F" : "#4CAF50";

  const confidenceLabel =
    latestScan && latestScan.Confidence > THRESHOLD
      ? `⚠️ High — ${latestScan.Confidence.toFixed(1)}%`
      : latestScan
      ? `✓ Normal — ${latestScan.Confidence.toFixed(1)}%`
      : "—";

  return (
    <SafeAreaView style={styles.container}>
      <Stack.Screen options={{ title: "Latest Result" }} />

      <ScrollView
        style={{ flex: 1, width: "100%" }}
        contentContainerStyle={styles.scrollContent}
      >
        <Text style={styles.heading}>Highest Confidence Scan</Text>

        {!initialized || !latestScan || !latestImage ? (
          <Text style={styles.placeholder}>
            No results yet. Scan a hive to get started.
          </Text>
        ) : (
          <>
            <View style={styles.imageWrapper}>
              <Image
                source={{ uri: getImageUri(latestImage.ImageFileName) }}
                style={styles.image}
                resizeMode="cover"
              />
            </View>
            <View
              style={[styles.confidenceBadge, { borderColor: confidenceColor }]}
            >
              <Text style={styles.confidenceLabel}>EFB Confidence</Text>
              <Text
                style={[styles.confidenceValue, { color: confidenceColor }]}
              >
                {confidenceLabel}
              </Text>
              {latestScan.Confidence > THRESHOLD && (
                <Text style={styles.warningNote}>
                  This scan exceeds the {THRESHOLD}% threshold. Consider
                  treatment.
                </Text>
              )}
            </View>
            <TouchableOpacity
              style={[
                styles.efbLink,
                latestScan.Confidence > THRESHOLD
                  ? styles.efbLinkWarning
                  : styles.efbLinkSubtle,
              ]}
              onPress={() => router.push("/capture/efbInfo")}
              activeOpacity={0.7}
            >
              {latestScan.Confidence > THRESHOLD ? (
                <>
                  <Text style={styles.efbLinkWarningText}>
                    What should I do now?
                  </Text>
                  <Text style={styles.efbLinkWarningSubtext}>
                    Tap for EFB guidance and reporting contacts
                  </Text>
                </>
              ) : (
                <Text style={styles.efbLinkSubtleText}>
                  Learn more about EFB →
                </Text>
              )}
            </TouchableOpacity>
            <View style={styles.metaCard}>
              <Row label="Hive" value={`#${latestScan.HiveNo}`} />
              <Row label="Date" value={latestImage.DateTaken} />
              <Row
                label="Name"
                value={latestImage.ImageName}
                onPress={openEditModal}
                editable
              />
            </View>
            <TouchableOpacity
              style={styles.continueButton}
              onPress={() => router.replace("/(tabs)")}
            >
              <Text style={styles.continueButtonText}>← Back to Home</Text>
            </TouchableOpacity>
          </>
        )}
      </ScrollView>
      <Modal
        visible={showEditModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowEditModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalBox}>
            <Text style={styles.modalTitle}>Rename Scan</Text>
            <TextInput
              style={styles.modalInput}
              value={editName}
              onChangeText={setEditName}
              placeholder="Enter scan name"
              autoFocus
              returnKeyType="done"
              onSubmitEditing={saveName}
            />
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.modalCancel]}
                onPress={() => setShowEditModal(false)}
              >
                <Text style={styles.modalButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButton, styles.modalSave]}
                onPress={saveName}
              >
                <Text style={[styles.modalButtonText, { color: "#fff" }]}>
                  Save
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}
function Row({
  label,
  value,
  editable,
  onPress,
}: {
  label: string;
  value: string;
  editable?: boolean;
  onPress?: () => void;
}) {
  return (
    <View style={styles.metaRow}>
      <Text style={styles.metaLabel}>{label}</Text>
      {editable ? (
        <TouchableOpacity onPress={onPress} style={{ flex: 1 }}>
          <Text style={[styles.metaValue, styles.metaValueEditable]}>
            {value} ✎
          </Text>
        </TouchableOpacity>
      ) : (
        <Text style={styles.metaValue}>{value}</Text>
      )}
    </View>
  );
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
  },
  scrollContent: {
    alignItems: "center",
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  heading: {
    fontSize: 20,
    fontWeight: "700",
    color: "#E3892B",
    marginBottom: 20,
    textAlign: "center",
  },
  placeholder: {
    fontSize: 16,
    color: "#aaa",
    marginTop: 60,
    textAlign: "center",
  },
  imageWrapper: {
    borderRadius: 14,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 4,
    marginBottom: 20,
  },
  image: {
    width: screenWidth - 40,
    height: screenWidth - 40,
  },
  confidenceBadge: {
    width: "100%",
    borderWidth: 2,
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    alignItems: "center",
    backgroundColor: "#fff",
  },
  confidenceLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: "#888",
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: 4,
  },
  confidenceValue: {
    fontSize: 22,
    fontWeight: "800",
  },
  warningNote: {
    marginTop: 8,
    fontSize: 12,
    color: "#D9534F",
    textAlign: "center",
  },
  efbLink: {
    width: "100%",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    alignItems: "center",
  },
  efbLinkWarning: {
    backgroundColor: "#FDF0F0",
    borderWidth: 1.5,
    borderColor: "#D9534F",
  },
  efbLinkWarningText: {
    fontSize: 16,
    fontWeight: "700",
    color: "#D9534F",
  },
  efbLinkWarningSubtext: {
    fontSize: 12,
    color: "#B94440",
    marginTop: 4,
  },
  efbLinkSubtle: {
    backgroundColor: "#FFF8ED",
    borderWidth: 1,
    borderColor: "#E9B44C",
  },
  efbLinkSubtleText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#E3892B",
  },
  metaCard: {
    width: "100%",
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07,
    shadowRadius: 4,
    elevation: 2,
  },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#F0F0F0",
    gap: 8,
  },
  metaLabel: {
    fontSize: 14,
    color: "#888",
    fontWeight: "500",
    width: 48,
  },
  metaValue: {
    flex: 1,
    fontSize: 14,
    color: "#333",
    fontWeight: "600",
    textAlign: "right",
  },
  metaValueEditable: {
    color: "#E3892B",
    textDecorationLine: "underline",
  },
  continueButton: {
    backgroundColor: "#E3892B",
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 10,
    alignItems: "center",
    width: "100%",
  },
  continueButtonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 15,
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
    color: "#333",
  },
  modalInput: {
    borderWidth: 1,
    borderColor: "#E0E0E0",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 15,
    marginBottom: 16,
    color: "#333",
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
    color: "#555",
  },
});
