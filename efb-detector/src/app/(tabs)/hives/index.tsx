import { IconSymbol } from "@/components/icon-symbol";
import { Link } from "expo-router";
import React from "react";
import { FlatList, Pressable, StyleSheet, Text, View } from "react-native";

// TODO: make this use proper data from Zustand
const MOCK_HIVES = [
  {
    id: "1",
    name: "North Apiary - Hive A",
    lastScan: "2024-05-10",
    numFrames: 10,
  },
  {
    id: "2",
    name: "North Apiary - Hive B",
    lastScan: "2024-05-10",
    numFrames: 8,
  },
  { id: "3", name: "Cambridge Orchard", lastScan: "2024-05-08", numFrames: 12 },
  { id: "4", name: "River Side", lastScan: "2024-05-01", numFrames: 9 },
];

export default function HivesScreen() {
  const renderItem = ({ item }: { item: (typeof MOCK_HIVES)[0] }) => (
    <Link href={`/hives/${item.id}`} asChild>
      <Pressable style={styles.card}>
        <View style={styles.cardInfo}>
          <Text style={styles.hiveName}>{item.name}</Text>
          <View style={styles.cardInfoDetails}>
            <Text style={styles.date}>{item.lastScan}</Text>
            <Text style={styles.date}>{item.numFrames} frames</Text>
          </View>
        </View>
        <IconSymbol size={20} name="chevron.right" color="#ccc" />
      </Pressable>
    </Link>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Your Hives</Text>
      <FlatList
        data={MOCK_HIVES}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        contentContainerStyle={styles.listContent}
        ListEmptyComponent={
          <Text style={styles.empty}>
            No hives added yet. Start scanning to add some!
          </Text>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f8f9fa",
    paddingHorizontal: 20,
    marginTop: 30,
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    marginVertical: 20,
    color: "#333",
  },
  listContent: { paddingBottom: 20 },
  card: {
    backgroundColor: "#fff",
    padding: 16,
    borderRadius: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  cardInfo: { flex: 1 },
  cardInfoDetails: {
    flex: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 8,
    paddingRight: 10,
  },
  hiveName: { fontSize: 18, fontWeight: "600", color: "#333" },
  date: { fontSize: 14, color: "#888", marginTop: 4 },
  empty: { textAlign: "center", marginTop: 50, color: "#999" },
});
