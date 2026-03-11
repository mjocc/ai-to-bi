import { useEffect } from "react";
import { Stack } from "expo-router";
import { preloadModels } from "@/services/mlService";

export default function RootLayout() {
  useEffect(() => {
    preloadModels();
  }, []);

  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: "#fff" },
        headerTitleStyle: {
          fontWeight: "900",
          fontSize: 25,
          color: "#333",
        },
        // headerShadowVisible: false,
        headerTintColor: "#E9B44C",
      }}
    >
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen
        name="capture"
        options={{
          headerShown: false,
          presentation: "fullScreenModal", // slides up to cover everything
          animation: "slide_from_bottom",
        }}
      />
    </Stack>
  );
}
