import { Link } from "expo-router";
import { Image, Pressable, StyleSheet, Text, View } from "react-native";

export default function Index() {
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
    elevation: 3, // Android shadow
    shadowColor: "#000", // iOS shadow
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  buttonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
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
