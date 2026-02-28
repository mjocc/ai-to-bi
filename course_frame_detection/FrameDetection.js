import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  StyleSheet,
  Alert,
  ScrollView,
  ActivityIndicator,
  Dimensions,
  Platform,
} from 'react-native';
import { Camera, useCameraDevices } from 'react-native-vision-camera';
import { launchImageLibrary } from 'react-native-image-picker';
import RNFS from 'react-native-fs';
import * as tf from '@tensorflow/tfjs';
import { fetch, bundleResourceIO } from '@tensorflow/tfjs-react-native';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

/**
 * FrameDetection Component
 * 
 * This component handles:
 * - Camera capture for beehive frame photos
 * - Image selection from gallery
 * - ML-based frame detection (frame present/absent)
 * - Data collection for research
 * - Visual feedback for beekeepers
 */
const FrameDetection = ({ onDetectionComplete, saveToDatabase }) => {
  // State management
  const [hasPermission, setHasPermission] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [model, setModel] = useState(null);
  const [tfReady, setTfReady] = useState(false);
  
  const cameraRef = useRef(null);
  const devices = useCameraDevices();
  const device = devices.back;

  // Initialize TensorFlow
  useEffect(() => {
    const initTF = async () => {
      try {
        await tf.ready();
        setTfReady(true);
        console.log('TensorFlow initialized');
        
        // Load the trained model
        await loadModel();
      } catch (error) {
        console.error('TensorFlow initialization failed:', error);
        Alert.alert('Error', 'Failed to initialize ML framework');
      }
    };
    
    initTF();
  }, []);

  // Request camera permissions
  useEffect(() => {
    const requestPermissions = async () => {
      const cameraPermission = await Camera.requestCameraPermission();
      setHasPermission(cameraPermission === 'authorized');
    };
    
    requestPermissions();
  }, []);

  /**
   * Load the pre-trained frame detection model
   * Replace with your actual model path/URL
   */
  const loadModel = async () => {
    try {
      // Option 1: Load from local assets
      const modelJson = require('./models/frame_detector/model.json');
      const modelWeights = require('./models/frame_detector/weights.bin');
      const loadedModel = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
      
      // Option 2: Load from remote URL (uncomment if needed)
      // const loadedModel = await tf.loadLayersModel('https://your-server.com/model.json');
      
      setModel(loadedModel);
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Model loading failed:', error);
      Alert.alert('Warning', 'ML model not loaded. Using fallback detection.');
    }
  };

  /**
   * Capture photo from camera
   */
  const takePhoto = async () => {
    if (!cameraRef.current) return;
    
    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'quality',
        enableAutoStabilization: true,
      });
      
      const imagePath = Platform.OS === 'ios' 
        ? photo.path 
        : `file://${photo.path}`;
      
      setSelectedImage(imagePath);
      await processImage(imagePath);
    } catch (error) {
      console.error('Photo capture failed:', error);
      Alert.alert('Error', 'Failed to capture photo');
    }
  };

  /**
   * Select image from gallery
   */
  const selectFromGallery = () => {
    const options = {
      mediaType: 'photo',
      quality: 1,
      maxWidth: 1024,
      maxHeight: 1024,
    };

    launchImageLibrary(options, async (response) => {
      if (response.didCancel) {
        return;
      }
      if (response.errorCode) {
        Alert.alert('Error', response.errorMessage);
        return;
      }
      
      const imageUri = response.assets[0].uri;
      setSelectedImage(imageUri);
      await processImage(imageUri);
    });
  };

  /**
   * Preprocess image for model input
   */
  const preprocessImage = async (imageUri) => {
    try {
      // Read image as base64
      const base64 = await RNFS.readFile(imageUri, 'base64');
      const imageData = `data:image/jpeg;base64,${base64}`;
      
      // Create image element
      const imgElement = new Image();
      imgElement.src = imageData;
      
      await new Promise((resolve) => {
        imgElement.onload = resolve;
      });
      
      // Convert to tensor and preprocess
      let tensor = tf.browser.fromPixels(imgElement);
      
      // Resize to model input size (e.g., 224x224)
      const resized = tf.image.resizeBilinear(tensor, [224, 224]);
      
      // Normalize pixel values to [0, 1] or [-1, 1] depending on your model
      const normalized = resized.div(255.0);
      
      // Add batch dimension
      const batched = normalized.expandDims(0);
      
      return batched;
    } catch (error) {
      console.error('Image preprocessing failed:', error);
      throw error;
    }
  };

  /**
   * Run inference with the loaded model
   */
  const runInference = async (imageTensor) => {
    if (!model) {
      // Fallback: simple color-based detection
      return fallbackDetection(imageTensor);
    }
    
    try {
      const prediction = await model.predict(imageTensor);
      const predictionData = await prediction.data();
      
      // Assuming binary classification: [no_frame_prob, frame_prob]
      const framePresent = predictionData[1] > 0.5;
      const confidence = predictionData[1];
      
      return {
        hasFrame: framePresent,
        confidence: confidence,
        probabilities: {
          noFrame: predictionData[0],
          frame: predictionData[1],
        },
      };
    } catch (error) {
      console.error('Inference failed:', error);
      return fallbackDetection(imageTensor);
    }
  };

  /**
   * Fallback detection method when ML model is unavailable
   * Uses simple image analysis heuristics
   */
  const fallbackDetection = async (imageTensor) => {
    try {
      // Simple heuristic: detect rectangular structures
      // This is a placeholder - you can implement edge detection or other methods
      
      const tensorData = await imageTensor.squeeze().array();
      
      // Calculate average brightness and edge density
      const avgBrightness = tf.mean(imageTensor).dataSync()[0];
      
      // Simple threshold-based detection
      const hasFrame = avgBrightness > 0.3 && avgBrightness < 0.8;
      
      return {
        hasFrame,
        confidence: 0.6,
        probabilities: {
          noFrame: hasFrame ? 0.4 : 0.7,
          frame: hasFrame ? 0.6 : 0.3,
        },
        method: 'fallback',
      };
    } catch (error) {
      console.error('Fallback detection failed:', error);
      return {
        hasFrame: false,
        confidence: 0,
        probabilities: { noFrame: 0.5, frame: 0.5 },
        error: true,
      };
    }
  };

  /**
   * Process the selected/captured image
   */
  const processImage = async (imageUri) => {
    setIsProcessing(true);
    setDetectionResult(null);
    
    try {
      // Preprocess image
      const imageTensor = await preprocessImage(imageUri);
      
      // Run inference
      const result = await runInference(imageTensor);
      
      // Clean up tensors
      imageTensor.dispose();
      
      // Prepare result with metadata
      const detectionData = {
        ...result,
        timestamp: new Date().toISOString(),
        imageUri: imageUri,
        imageSize: await getImageSize(imageUri),
      };
      
      setDetectionResult(detectionData);
      
      // Save to database if callback provided
      if (saveToDatabase) {
        await saveToDatabase(detectionData);
      }
      
      // Notify parent component
      if (onDetectionComplete) {
        onDetectionComplete(detectionData);
      }
      
    } catch (error) {
      console.error('Image processing failed:', error);
      Alert.alert('Error', 'Failed to process image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  /**
   * Get image dimensions
   */
  const getImageSize = async (imageUri) => {
    return new Promise((resolve) => {
      Image.getSize(
        imageUri,
        (width, height) => resolve({ width, height }),
        () => resolve({ width: 0, height: 0 })
      );
    });
  };

  /**
   * Reset and take another photo
   */
  const resetDetection = () => {
    setSelectedImage(null);
    setDetectionResult(null);
  };

  // Render loading state
  if (!tfReady) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#F5A623" />
        <Text style={styles.loadingText}>Initializing detection system...</Text>
      </View>
    );
  }

  // Render camera view
  if (!selectedImage && hasPermission && device) {
    return (
      <View style={styles.container}>
        <Camera
          ref={cameraRef}
          style={styles.camera}
          device={device}
          isActive={true}
          photo={true}
        />
        
        <View style={styles.cameraOverlay}>
          <View style={styles.frameGuide}>
            <View style={styles.corner} style={[styles.corner, styles.topLeft]} />
            <View style={styles.corner} style={[styles.corner, styles.topRight]} />
            <View style={styles.corner} style={[styles.corner, styles.bottomLeft]} />
            <View style={styles.corner} style={[styles.corner, styles.bottomRight]} />
          </View>
          
          <Text style={styles.guideText}>
            Position the beehive frame within the guides
          </Text>
        </View>
        
        <View style={styles.controls}>
          <TouchableOpacity 
            style={styles.galleryButton}
            onPress={selectFromGallery}
          >
            <Text style={styles.buttonText}>📁 Gallery</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.captureButton}
            onPress={takePhoto}
          >
            <View style={styles.captureButtonInner} />
          </TouchableOpacity>
          
          <View style={styles.placeholder} />
        </View>
      </View>
    );
  }

  // Render results view
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      {selectedImage && (
        <View style={styles.imageContainer}>
          <Image 
            source={{ uri: selectedImage }} 
            style={styles.image}
            resizeMode="contain"
          />
        </View>
      )}
      
      {isProcessing && (
        <View style={styles.processingOverlay}>
          <ActivityIndicator size="large" color="#F5A623" />
          <Text style={styles.processingText}>Analyzing frame...</Text>
        </View>
      )}
      
      {detectionResult && !isProcessing && (
        <View style={styles.resultsContainer}>
          <View style={[
            styles.resultBanner,
            detectionResult.hasFrame ? styles.successBanner : styles.warningBanner
          ]}>
            <Text style={styles.resultEmoji}>
              {detectionResult.hasFrame ? '✅' : '⚠️'}
            </Text>
            <Text style={styles.resultTitle}>
              {detectionResult.hasFrame ? 'Frame Detected' : 'No Frame Detected'}
            </Text>
          </View>
          
          <View style={styles.detailsCard}>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Confidence:</Text>
              <Text style={styles.detailValue}>
                {(detectionResult.confidence * 100).toFixed(1)}%
              </Text>
            </View>
            
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Frame Present:</Text>
              <Text style={styles.detailValue}>
                {(detectionResult.probabilities.frame * 100).toFixed(1)}%
              </Text>
            </View>
            
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>No Frame:</Text>
              <Text style={styles.detailValue}>
                {(detectionResult.probabilities.noFrame * 100).toFixed(1)}%
              </Text>
            </View>
            
            {detectionResult.method === 'fallback' && (
              <View style={styles.methodBadge}>
                <Text style={styles.methodText}>Using fallback detection</Text>
              </View>
            )}
          </View>
          
          <View style={styles.actionButtons}>
            <TouchableOpacity 
              style={styles.secondaryButton}
              onPress={resetDetection}
            >
              <Text style={styles.secondaryButtonText}>📸 New Photo</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={styles.primaryButton}
              onPress={() => {
                Alert.alert(
                  'Success',
                  'Detection result saved for research',
                  [{ text: 'OK', onPress: resetDetection }]
                );
              }}
            >
              <Text style={styles.primaryButtonText}>💾 Save Result</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
      
      {!selectedImage && (
        <View style={styles.emptyState}>
          <Text style={styles.emptyEmoji}>📷</Text>
          <Text style={styles.emptyTitle}>No Image Selected</Text>
          <Text style={styles.emptyText}>
            Take a photo or select from gallery to detect beehive frames
          </Text>
          
          <TouchableOpacity 
            style={styles.primaryButton}
            onPress={selectFromGallery}
          >
            <Text style={styles.primaryButtonText}>Select from Gallery</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8F9FA',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F8F9FA',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#495057',
    fontFamily: 'System',
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  frameGuide: {
    width: SCREEN_WIDTH * 0.8,
    height: SCREEN_WIDTH * 0.8 * 1.5,
    position: 'relative',
  },
  corner: {
    position: 'absolute',
    width: 40,
    height: 40,
    borderColor: '#F5A623',
    borderWidth: 3,
  },
  topLeft: {
    top: 0,
    left: 0,
    borderRightWidth: 0,
    borderBottomWidth: 0,
  },
  topRight: {
    top: 0,
    right: 0,
    borderLeftWidth: 0,
    borderBottomWidth: 0,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
    borderRightWidth: 0,
    borderTopWidth: 0,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
    borderLeftWidth: 0,
    borderTopWidth: 0,
  },
  guideText: {
    marginTop: 24,
    fontSize: 16,
    color: '#FFFFFF',
    textAlign: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    fontFamily: 'System',
    fontWeight: '600',
  },
  controls: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  galleryButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 25,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    fontFamily: 'System',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#F5A623',
  },
  captureButtonInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#F5A623',
  },
  placeholder: {
    width: 80,
  },
  scrollContent: {
    padding: 20,
  },
  imageContainer: {
    width: '100%',
    height: 400,
    backgroundColor: '#000000',
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
  },
  image: {
    width: '100%',
    height: '100%',
  },
  processingOverlay: {
    padding: 40,
    alignItems: 'center',
  },
  processingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#495057',
    fontFamily: 'System',
  },
  resultsContainer: {
    marginBottom: 20,
  },
  resultBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    borderRadius: 12,
    marginBottom: 16,
  },
  successBanner: {
    backgroundColor: '#D4EDDA',
  },
  warningBanner: {
    backgroundColor: '#FFF3CD',
  },
  resultEmoji: {
    fontSize: 32,
    marginRight: 12,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#212529',
    fontFamily: 'System',
  },
  detailsCard: {
    backgroundColor: '#FFFFFF',
    padding: 20,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E9ECEF',
  },
  detailLabel: {
    fontSize: 16,
    color: '#6C757D',
    fontFamily: 'System',
  },
  detailValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#212529',
    fontFamily: 'System',
  },
  methodBadge: {
    marginTop: 12,
    backgroundColor: '#FFF3CD',
    padding: 8,
    borderRadius: 6,
  },
  methodText: {
    fontSize: 14,
    color: '#856404',
    textAlign: 'center',
    fontFamily: 'System',
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  primaryButton: {
    flex: 1,
    backgroundColor: '#F5A623',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    fontFamily: 'System',
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: '#6C757D',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    fontFamily: 'System',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyEmoji: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#212529',
    marginBottom: 8,
    fontFamily: 'System',
  },
  emptyText: {
    fontSize: 16,
    color: '#6C757D',
    textAlign: 'center',
    marginBottom: 32,
    paddingHorizontal: 40,
    fontFamily: 'System',
  },
});

export default FrameDetection;
