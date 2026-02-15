import { Stack } from "expo-router";
import React from "react";
import {
  Linking,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

export default function EFBInfoScreen() {
  const handleEmailNBU = () => {
    // TODO: ideally would also include images of the hive that we have stored
    const subject = "Suspected EFB case";
    const url = `mailto:nbu@apha.gov.uk?subject=${encodeURIComponent(subject)}`;

    Linking.openURL(url).catch((err) =>
      console.error("Failed to open email client", err)
    );
  };

  return (
    <ScrollView style={styles.container}>
      <Stack.Screen options={{ headerShadowVisible: false, title: "" }} />

      <View style={styles.content}>
        <Text style={styles.warningTitle}>EFB Suspected?</Text>
        <Text style={styles.description}>
          European foulbrood (EFB) is a statutory notifiable infection of honey
          bees. Any beekeeper in England or Wales who suspects the presence of
          AFB in a colony for which they are responsible is legally required to
          inform the NBU.
        </Text>

        <Pressable
          style={styles.infoLink}
          onPress={() =>
            Linking.openURL(
              "https://www.nationalbeeunit.com/diseases-and-pests/foulbroods-notifiable/how-to-spot-european-foul-brood"
            )
          }
        >
          <Text style={styles.infoLinkText}>
            How to spot EFB (Official NBU Guide) →
          </Text>
        </Pressable>

        <Pressable
          style={{ ...styles.infoLink, marginBottom: 50 }}
          onPress={() =>
            Linking.openURL(
              "https://www.nationalbeeunit.com/assets/PDFs/3_Resources_for_beekeepers/Advisory_leaflets/Foulbrood_2017_Web_version.pdf"
            )
          }
        >
          <Text style={styles.infoLinkText}>
            "Foulbrood Disease of Honey Bees" advisory leaflet (APHA/NBU) →
          </Text>
        </Pressable>

        {/* TODO: add bee inspector contact? or does it just go through NBU? */}

        <View style={styles.actionContainer}>
          <Pressable
            style={[styles.button, styles.nbuButton]}
            onPress={handleEmailNBU}
          >
            <Text style={styles.buttonText}>Contact National Bee Unit</Text>
            <Text style={styles.buttonSubtext}>Email: nbu@apha.gov.uk</Text>
          </Pressable>

          <Pressable
            style={[styles.button, styles.localButton]}
            onPress={() =>
              alert("Local Association contact not yet implemented -- TODO")
            }
          >
            <Text style={styles.buttonText}>Contact Local Association</Text>
            <Text style={styles.buttonSubtext}>TODO Beekeepers' Assoc</Text>
          </Pressable>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fff" },
  content: { padding: 20, paddingTop: 0 },
  warningTitle: {
    fontSize: 36,
    fontWeight: "bold",
    color: "#e74c3c",
    marginBottom: 10,
    textAlign: "center",
  },
  description: {
    fontSize: 18,
    color: "#333",
    lineHeight: 22,
    marginBottom: 30,
    fontWeight: "700",
  },
  infoLink: {
    padding: 15,
    backgroundColor: "#f0f0f0",
    borderRadius: 8,
    marginBottom: 15,
  },
  infoLinkText: { color: "#007AFF", fontWeight: "600", textAlign: "center" },
  actionContainer: { gap: 15 },
  button: { padding: 20, borderRadius: 12, alignItems: "center" },
  nbuButton: { backgroundColor: "#E9B44C" },
  localButton: { backgroundColor: "#333" },
  buttonText: { fontSize: 18, fontWeight: "bold", color: "#fff" },
  buttonSubtext: { fontSize: 12, color: "#fff", opacity: 0.8, marginTop: 4 },
});
