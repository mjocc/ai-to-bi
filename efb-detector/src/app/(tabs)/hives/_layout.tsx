import { Stack } from "expo-router";

export default function HivesLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="index" options={{ title: "Hives" }} />
      <Stack.Screen name="[hiveId]" options={{ title: "Hive Details" }} />
    </Stack>
  );
}
