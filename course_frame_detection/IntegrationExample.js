/**
 * Example Integration - Beekeeping App
 * 
 * This file shows how to integrate the FrameDetection component
 * into your larger beekeeping application with disease detection,
 * data collection, and beekeeper guidance features.
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ScrollView,
} from 'react-native';
import FrameDetection from './FrameDetection';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';

/**
 * Main Inspection Screen
 * Coordinates frame detection with disease detection and data collection
 */
const InspectionScreen = ({ navigation }) => {
  const [currentStep, setCurrentStep] = useState('frame_detection');
  const [frameDetectionResult, setFrameDetectionResult] = useState(null);
  const [inspectionData, setInspectionData] = useState({
    timestamp: new Date().toISOString(),
    location: null,
    hiveId: null,
    inspectorId: null,
  });

  /**
   * Handle frame detection completion
   * This is the callback passed to FrameDetection component
   */
  const handleFrameDetection = async (result) => {
    console.log('Frame detection completed:', result);
    
    setFrameDetectionResult(result);
    
    // Update inspection data
    setInspectionData(prev => ({
      ...prev,
      frameDetection: result,
      hasFrame: result.hasFrame,
      frameConfidence: result.confidence,
    }));

    // If frame detected, proceed to disease detection
    if (result.hasFrame && result.confidence > 0.7) {
      Alert.alert(
        'Frame Detected',
        'Would you like to check for diseases?',
        [
          {
            text: 'Not Now',
            style: 'cancel',
            onPress: () => saveInspection(result),
          },
          {
            text: 'Check Diseases',
            onPress: () => proceedToDiseaseDetection(result),
          },
        ]
      );
    } else if (!result.hasFrame) {
      Alert.alert(
        'No Frame Detected',
        'Please ensure the frame is clearly visible in the image.',
        [
          { text: 'Try Again', onPress: () => resetDetection() },
          { text: 'Save Anyway', onPress: () => saveInspection(result) },
        ]
      );
    }
  };

  /**
   * Proceed to disease detection (implemented by other team member)
   */
  const proceedToDiseaseDetection = (frameResult) => {
    // Navigate to disease detection screen
    // Pass the frame image and detection results
    navigation.navigate('DiseaseDetection', {
      imageUri: frameResult.imageUri,
      frameData: frameResult,
      inspectionData: inspectionData,
    });
  };

  /**
   * Save inspection data to local storage and sync when online
   */
  const saveInspection = async (result) => {
    try {
      // Generate unique inspection ID
      const inspectionId = `inspection_${Date.now()}`;
      
      // Prepare complete inspection data
      const completeData = {
        id: inspectionId,
        ...inspectionData,
        frameDetection: result,
        status: 'pending_sync',
      };

      // Save to local storage
      await AsyncStorage.setItem(
        `@inspection:${inspectionId}`,
        JSON.stringify(completeData)
      );

      console.log('Inspection saved locally:', inspectionId);

      // Try to sync if online
      const netState = await NetInfo.fetch();
      if (netState.isConnected) {
        await syncInspectionToServer(completeData);
      }

      Alert.alert(
        'Saved',
        'Inspection data saved successfully',
        [{ text: 'OK', onPress: () => navigation.goBack() }]
      );

    } catch (error) {
      console.error('Error saving inspection:', error);
      Alert.alert('Error', 'Failed to save inspection data');
    }
  };

  /**
   * Sync inspection data to research server
   */
  const syncInspectionToServer = async (data) => {
    try {
      // Replace with your actual API endpoint
      const API_ENDPOINT = 'https://cambridge-beekeeping.org/api/inspections';

      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${await getAuthToken()}`,
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        // Mark as synced
        await AsyncStorage.setItem(
          `@inspection:${data.id}`,
          JSON.stringify({ ...data, status: 'synced' })
        );
        console.log('Inspection synced to server');
      } else {
        throw new Error('Server returned error');
      }
    } catch (error) {
      console.error('Sync failed:', error);
      // Will retry next time app is online
    }
  };

  /**
   * Get authentication token (implement based on your auth system)
   */
  const getAuthToken = async () => {
    // Implement your auth logic
    return await AsyncStorage.getItem('@auth:token');
  };

  /**
   * Database callback for research data collection
   */
  const saveToResearchDatabase = async (detectionData) => {
    try {
      // Prepare research data
      const researchData = {
        type: 'frame_detection',
        timestamp: detectionData.timestamp,
        result: detectionData.hasFrame,
        confidence: detectionData.confidence,
        imageMetadata: {
          size: detectionData.imageSize,
          location: inspectionData.location,
        },
        // Anonymized data for research
        anonymousId: await getAnonymousId(),
      };

      // Save to research database (separate from personal data)
      await AsyncStorage.setItem(
        `@research:${Date.now()}`,
        JSON.stringify(researchData)
      );

      console.log('Research data saved');
    } catch (error) {
      console.error('Error saving research data:', error);
    }
  };

  /**
   * Get or create anonymous research ID
   */
  const getAnonymousId = async () => {
    let id = await AsyncStorage.getItem('@research:anonymousId');
    if (!id) {
      id = `anon_${Math.random().toString(36).substr(2, 9)}`;
      await AsyncStorage.setItem('@research:anonymousId', id);
    }
    return id;
  };

  /**
   * Reset detection and start over
   */
  const resetDetection = () => {
    setFrameDetectionResult(null);
    setCurrentStep('frame_detection');
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Hive Inspection</Text>
        <Text style={styles.headerSubtitle}>
          {currentStep === 'frame_detection' 
            ? 'Step 1: Frame Detection' 
            : 'Step 2: Disease Detection'}
        </Text>
      </View>

      {/* Main Content */}
      <View style={styles.content}>
        {currentStep === 'frame_detection' && (
          <FrameDetection
            onDetectionComplete={handleFrameDetection}
            saveToDatabase={saveToResearchDatabase}
          />
        )}
      </View>

      {/* Footer with inspection info */}
      {inspectionData.hiveId && (
        <View style={styles.footer}>
          <Text style={styles.footerText}>
            Hive: {inspectionData.hiveId} | {new Date().toLocaleDateString()}
          </Text>
        </View>
      )}
    </View>
  );
};

/**
 * Integration with Navigation
 */
const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen 
          name="Home" 
          component={HomeScreen}
          options={{ title: 'Cambridge Beekeeping' }}
        />
        <Stack.Screen 
          name="Inspection" 
          component={InspectionScreen}
          options={{ title: 'Hive Inspection' }}
        />
        <Stack.Screen 
          name="DiseaseDetection" 
          component={DiseaseDetectionScreen}
          options={{ title: 'Disease Detection' }}
        />
        <Stack.Screen 
          name="History" 
          component={InspectionHistoryScreen}
          options={{ title: 'Inspection History' }}
        />
        <Stack.Screen 
          name="Guidance" 
          component={BeekeeperGuidanceScreen}
          options={{ title: 'Beekeeper Guidance' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

/**
 * Offline Sync Service
 * Runs in background to sync pending inspections
 */
const OfflineSyncService = {
  /**
   * Get all pending inspections
   */
  getPendingInspections: async () => {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const inspectionKeys = keys.filter(k => k.startsWith('@inspection:'));
      
      const inspections = await AsyncStorage.multiGet(inspectionKeys);
      
      return inspections
        .map(([key, value]) => JSON.parse(value))
        .filter(inspection => inspection.status === 'pending_sync');
    } catch (error) {
      console.error('Error getting pending inspections:', error);
      return [];
    }
  },

  /**
   * Sync all pending inspections
   */
  syncAll: async () => {
    const netState = await NetInfo.fetch();
    if (!netState.isConnected) {
      console.log('Offline - sync deferred');
      return;
    }

    const pending = await OfflineSyncService.getPendingInspections();
    console.log(`Syncing ${pending.length} pending inspections`);

    for (const inspection of pending) {
      try {
        await syncInspectionToServer(inspection);
      } catch (error) {
        console.error(`Failed to sync ${inspection.id}:`, error);
      }
    }
  },

  /**
   * Start periodic sync
   */
  startPeriodicSync: () => {
    // Sync every 5 minutes when app is active
    setInterval(() => {
      OfflineSyncService.syncAll();
    }, 5 * 60 * 1000);

    // Also sync when network status changes
    NetInfo.addEventListener(state => {
      if (state.isConnected) {
        OfflineSyncService.syncAll();
      }
    });
  },
};

/**
 * Utility: Export inspection data for research
 */
const exportResearchData = async () => {
  try {
    const keys = await AsyncStorage.getAllKeys();
    const researchKeys = keys.filter(k => k.startsWith('@research:'));
    
    const data = await AsyncStorage.multiGet(researchKeys);
    const researchData = data.map(([key, value]) => JSON.parse(value));

    // Export to CSV or JSON
    const csv = convertToCSV(researchData);
    
    // Save or share file
    // Implementation depends on your file sharing method
    
    return csv;
  } catch (error) {
    console.error('Error exporting research data:', error);
  }
};

const convertToCSV = (data) => {
  if (data.length === 0) return '';
  
  const headers = Object.keys(data[0]).join(',');
  const rows = data.map(obj => Object.values(obj).join(','));
  
  return [headers, ...rows].join('\n');
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8F9FA',
  },
  header: {
    backgroundColor: '#F5A623',
    padding: 20,
    paddingTop: 60,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#FFFFFF',
    opacity: 0.9,
  },
  content: {
    flex: 1,
  },
  footer: {
    backgroundColor: '#FFFFFF',
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: '#E9ECEF',
  },
  footerText: {
    textAlign: 'center',
    color: '#6C757D',
    fontSize: 14,
  },
});

export default InspectionScreen;
export { OfflineSyncService, exportResearchData };
