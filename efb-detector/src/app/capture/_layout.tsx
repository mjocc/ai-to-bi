import { Stack } from "expo-router";
import React from "react";

export default function CaptureLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="index" options={{ title: "Scan" }} />
      <Stack.Screen
        name="efbInfo"
        options={{ title: "EFB Guidance", headerShown: true, headerBackTitle: "Back" }}
      />
      <Stack.Screen name="results" options={{ title: "Results" }} />
    </Stack>
  );
}
