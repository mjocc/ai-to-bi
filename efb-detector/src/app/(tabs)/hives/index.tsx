import { IconSymbol } from "@/components/icon-symbol";
import { Link } from "expo-router";
import React, { useEffect } from "react";
import { FlatList, Pressable, StyleSheet, Text, View } from "react-native";
import { useStore } from "../../../../store";

export default function HivesScreen() {
  const { getDistinctHiveNumbers, getScansByHive, initializeData } = useStore();

  useEffect(() => {
    initializeData();
  }, []);

  const hiveNumbers = getDistinctHiveNumbers();

  const hives = hiveNumbers.map((hiveNo) => {
    const scans = getScansByHive(hiveNo);
    const lastScan = scans.length > 0 ? scans[scans.length - 1].DateTaken : null;
    return {
      id: String(hiveNo),
      hiveNo,
      lastScan: lastScan ? new Date(lastScan).toLocaleDateString() : "No scans yet",
      numScans: scans.length,
    };
  });

  const renderItem = ({ item }: { item: typeof hives[0] }) => (
    <Link href={`/hives/${item.hiveNo}`} asChild>
      <Pressable style={styles.card}>
        <View style={styles.cardInfo}>
          <Text style={styles.hiveName}>Hive {item.hiveNo}</Text>
          <View style={styles.cardInfoDetails}>
            <Text style={styles.date}>Last scan: {item.lastScan}</Text>
            <Text style={styles.date}>{item.numScans} scan{item.numScans !== 1 ? "s" : ""}</Text>
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
        data={hives}
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