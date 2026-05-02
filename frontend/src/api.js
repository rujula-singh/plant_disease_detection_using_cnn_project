// api.js — Handles backend communication for plant disease prediction
import axios from "axios";

// Base URL for the FastAPI backend
const API_BASE = "http://127.0.0.1:8000";

/**
 * Sends an image file to the backend for disease prediction.
 * @param {File} imageFile - The image file to analyze
 * @returns {Promise<{status: string, disease: string, confidence: number}>}
 */
export const predictDisease = async (imageFile) => {
  // Create FormData and append the image file
  const formData = new FormData();
  formData.append("file", imageFile);

  try {
    // POST request to the /predict endpoint
    // Let axios auto-set Content-Type with the correct multipart boundary
    const response = await axios.post(`${API_BASE}/predict`, formData);
    return response.data;
  } catch (error) {
    console.error("Error predicting disease:", error);
    throw error;
  }
};

/**
 * Sends multiple image files to the backend in a single bulk request.
 * @param {File[]} imageFiles - Array of image files to analyze
 * @returns {Promise<{status: string, results: Array<{filename: string, status: string, disease: string, confidence: number}>}>}
 */
export const predictBulk = async (imageFiles) => {
  const formData = new FormData();
  imageFiles.forEach((file) => formData.append("files", file));

  try {
    const response = await axios.post(`${API_BASE}/predict/bulk`, formData);
    return response.data;
  } catch (error) {
    console.error("Error in bulk prediction:", error);
    throw error;
  }
};
