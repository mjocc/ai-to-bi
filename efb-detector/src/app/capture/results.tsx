/**
 * capture/results.tsx
 *
 * Displays the stitched panorama from the most recent scan.
 */

import { Stack, useRouter } from "expo-router";
import React, { useState } from "react";
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
import { useBeeStore } from "../../store/useBeeStore";

const screenWidth = Dimensions.get("window").width;

// const THRESHOLD = 80, or something;

export default function ResultsScreen() {
  const router = useRouter();
  const { getLatestStitchedScan, updateScanName } = useBeeStore();

  const [editName, setEditName] = useState("");
  const [showEditModal, setShowEditModal] = useState(false);

  // TODO: wire up confidence (and everything else)
  // const [confidence, setConfidence] = useState<number | null>(null);

  const scan = getLatestStitchedScan();

  const openEditModal = () => {
    if (!scan) return;
    setEditName(scan.name ?? "");
    setShowEditModal(true);
  };

  const saveName = () => {
    if (!scan) return;
    if (!editName.trim()) {
      Alert.alert("Invalid name", "Name cannot be empty.");
      return;
    }
    updateScanName(scan.id, editName.trim());
    setShowEditModal(false);
  };

  return (
    <View style={styles.container}>
      <Stack.Screen options={{ title: "Latest Result" }} />

      <ScrollView
        style={{ flex: 1, width: "100%" }}
        contentContainerStyle={styles.scrollContent}
      >
        <Text style={styles.heading}>Stitched Panorama</Text>

        {!scan ? (
          <Text style={styles.placeholder}>
            No results yet. Record a hive to get started.
          </Text>
        ) : (
          <>
            {/* Panorama image */}
            <View style={styles.imageWrapper}>
              <Image
                source={{ uri: scan.panoramaUri }}
                style={styles.image}
                resizeMode="contain"
              />
            </View>

            <View style={styles.placeholderBadge}>
              <Text style={styles.placeholderBadgeText}>
                🐝 Placeholder
              </Text>
            </View>

            {/* Meta */}
            <View style={styles.metaCard}>
              <Row label="Hive" value={`#${scan.hiveNo ?? "—"}`} />
              <Row label="Date" value={scan.date} />
              <Row
                label="Name"
                value={scan.name || "Unnamed scan"}
                editable
                onPress={openEditModal}
              />
            </View>

            <TouchableOpacity style={styles.renameButton} onPress={openEditModal}>
              <Text style={styles.renameButtonText}>✎ Rename This Scan</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.continueButton}
              onPress={() => router.replace("/(tabs)")}
            >
              <Text style={styles.continueButtonText}>Continue →</Text>
            </TouchableOpacity>
          </>
        )}
      </ScrollView>

      {/* Rename modal */}
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
    </View>
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
    backgroundColor: "#FAFAFA",
    alignItems: "center",
    paddingTop: 60,
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
    width: screenWidth - 40,
    borderRadius: 14,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 4,
    marginBottom: 20,
    backgroundColor: "#111",
  },
  image: {
    width: "100%",
    // fix height to avoid bad display
    height: (screenWidth - 40) * 0.45,
  },
  placeholderBadge: {
    width: "100%",
    borderWidth: 1,
    borderColor: "#E0E0E0",
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
    alignItems: "center",
    backgroundColor: "#fff",
  },
  placeholderBadgeText: {
    fontSize: 14,
    color: "#999",
    fontStyle: "italic",
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
  renameButton: {
    backgroundColor: "#F6C24E",
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 10,
    alignItems: "center",
    width: "100%",
    marginBottom: 12,
  },
  renameButtonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 15,
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
  modalButtons: { flexDirection: "row", gap: 10 },
  modalButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: "center",
  },
  modalCancel: { backgroundColor: "#F0F0F0" },
  modalSave: { backgroundColor: "#F6C24E" },
  modalButtonText: { fontWeight: "600", fontSize: 15, color: "#555" },
});