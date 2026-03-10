import { Link, Tabs } from "expo-router";
import React, { useEffect } from "react";
import { test } from "@/services/classify_larvae_test";

import { IconSymbol } from "@/components/icon-symbol";
import { Pressable } from "react-native";

export default function TabLayout() {
  //useEffect(() => {
  //    test();
  //  }, []);
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: "#E9B44C",
        tabBarInactiveTintColor: "#666",
        headerShown: false,
        tabBarStyle: { backgroundColor: "#fff" },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Home",
          tabBarIcon: ({ color }) => (
            <IconSymbol size={28} name="house.fill" color={color} />
          ),
          headerTransparent: true,
          headerTitle: "",
          headerShown: true,
          headerRight: () => (
            <Link href="/settings" asChild>
              <Pressable style={{ marginRight: 35, marginTop: -30 }}>
                <IconSymbol size={34} name="gearshape.fill" color="#666" />
              </Pressable>
            </Link>
          ),
        }}
      />
      <Tabs.Screen
        name="hives"
        options={{
          title: "Hives",
          tabBarIcon: ({ color }) => (
            <IconSymbol size={28} name="circle.grid.hex" color={color} />
          ),
        }}
      />
    </Tabs>
  );
}
