import { Stack } from "expo-router";
import { Linking, StyleSheet, Text, View } from "react-native";

export default function SettingsScreen() {
  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: "Settings",
        }}
      />

      <Text style={{ color: "red", fontSize: 20 }}>
        TODO: add box to enter BKA email address here and any other settings
      </Text>

      <Text style={styles.header}>Privacy</Text>

      <Text style={styles.text}>
        We process all data locally. We never see your photos, your location, or
        your scan results.
      </Text>

      <Text style={styles.header}>Credits & Licensing</Text>

      <View style={styles.creditItem}>
        <Text style={styles.label}>Honeybee Hero Image</Text>
        <Text style={styles.smallText}>
          "202203 western honey bee.svg" by DataBase Center for Life Science
          (DBCLS)
        </Text>
        <Text
          style={styles.link}
          onPress={() =>
            Linking.openURL(
              "https://commons.wikimedia.org/wiki/File:202203_western_honey_bee.svg"
            )
          }
        >
          View Source
        </Text>
        <Text
          style={styles.link}
          onPress={() =>
            Linking.openURL("https://creativecommons.org/licenses/by/4.0/")
          }
        >
          Licensed under CC BY 4.0
        </Text>
      </View>

      <View style={styles.creditItem}>
        <Text style={styles.label}>Software</Text>
        <Text style={styles.smallText}>
          This app is powered by Expo and React Native.
        </Text>
        <Text style={styles.smallText}>
          The following libraries are used under the MIT License: Expo Router,
          Zustand.
        </Text>
        <Text style={styles.smallText}>
          Full open-source license text for all dependencies is available upon
          request or at our project repository.
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: "#fff" },
  header: { fontSize: 22, fontWeight: "bold", marginBottom: 8, marginTop: 15 },
  creditItem: { marginBottom: 15 },
  label: { fontWeight: "600", fontSize: 16 },
  text: { color: "#333", marginBottom: 6, fontSize: 16 },
  smallText: { color: "#333", marginBottom: 6 },
  link: { color: "#007AFF", textDecorationLine: "underline", marginBottom: 4 },
});
