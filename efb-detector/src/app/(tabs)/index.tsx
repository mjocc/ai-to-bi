import { Link, useFocusEffect, useRouter } from "expo-router";
import { useCallback, useRef } from "react";
import { Image, Pressable, StyleSheet, Text, View } from "react-native";
import { VolumeManager } from "react-native-volume-manager";

export default function Index() {
  const router = useRouter();
  const lastPress = useRef(0);

  useFocusEffect(
    useCallback(() => {
      VolumeManager.showNativeVolumeUI({ enabled: false });

      const subscription = VolumeManager.addVolumeListener(() => {
        const now = Date.now();
        if (now - lastPress.current > 300) {
          lastPress.current = now;
          router.push("/capture");
        }
      });

      return () => {
        VolumeManager.showNativeVolumeUI({ enabled: true });
        subscription.remove();
      };
    }, [])
  );

  return (
    <View style={styles.container}>
      <Image
        source={require("../../../assets/images/honeybee.png")}
        style={styles.beeImage}
        resizeMode="contain"
      />

      <View style={styles.textContainer}>
        <Text style={styles.title}>AI to BI</Text>
        <Text style={styles.subtitle}>Mobile App for EFB Detection</Text>
      </View>

      <Link href="/capture" asChild>
        <Pressable style={styles.button}>
          <Text style={styles.buttonText}>Start a scan</Text>
        </Pressable>
      </Link>

      <View style={styles.tipPill}>
        <Text style={styles.tipText}>📢 Press a volume button to start a scan</Text>
      </View>

      <Link href="/capture/efbInfo" asChild>
        <Pressable style={styles.linkContainer}>
          <Text style={styles.linkText}>What to do if you suspect EFB?</Text>
        </Pressable>
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  beeImage: {
    width: "80%",
    height: 300,
    marginBottom: 20,
  },
  textContainer: {
    alignItems: "center",
    marginBottom: 40,
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#333",
  },
  subtitle: {
    fontSize: 18,
    color: "#666",
  },
  button: {
    backgroundColor: "#E9B44C",
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 30,
    elevation: 3,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  buttonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
  },
  tipPill: {
    marginTop: 20,
    backgroundColor: "#FFF3D6",
    borderRadius: 20,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: "#F6C24E",
  },
  tipText: {
    color: "#B8760A",
    fontSize: 13,
    fontWeight: "500",
  },
  linkContainer: {
    marginTop: 20,
    padding: 10,
  },
  linkText: {
    color: "#666",
    fontSize: 16,
    textDecorationLine: "underline",
  },
});
