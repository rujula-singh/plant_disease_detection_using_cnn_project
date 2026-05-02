// App.jsx — Main application component with per-image prediction UX
import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Import components
import Navbar from "./components/Navbar";
import UploadArea from "./components/UploadArea";
import ImagePreview from "./components/ImagePreview";
import Footer from "./components/Footer";

// Import API function
import { predictDisease, predictBulk } from "./api";

function App() {
  // ==================== STATE ====================
  // Each image object: { file, preview, prediction, confidence, loading, failed, failError }
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [error, setError] = useState("");
  const [predictingAll, setPredictingAll] = useState(false);
  // Bulk prediction summary: { total, success, failed } — shown after a bulk run
  const [bulkSummary, setBulkSummary] = useState(null);

  // Dark mode — check localStorage or system preference
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem("darkMode");
    if (saved !== null) return JSON.parse(saved);
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  // ==================== EFFECTS ====================

  // Toggle "dark" class on <html> and persist to localStorage
  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem("darkMode", JSON.stringify(darkMode));
  }, [darkMode]);

  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => images.forEach((img) => URL.revokeObjectURL(img.preview));
  }, []);

  // ==================== HANDLERS ====================

  const handleImagesSelected = useCallback((files) => {
    const newImages = files.map((file) => ({
      file,
      preview: URL.createObjectURL(file),
      prediction: null,
      confidence: null,
      loading: false,
      failed: false,
      failError: null,
    }));
    setImages((prev) => [...prev, ...newImages]);
    setError("");
    setBulkSummary(null);
  }, []);

  const handleRemoveImage = useCallback((indexToRemove) => {
    setImages((prev) => {
      URL.revokeObjectURL(prev[indexToRemove].preview);
      return prev.filter((_, i) => i !== indexToRemove);
    });
    setCurrentIndex((prev) => {
      const newLength = images.length - 1;
      if (newLength === 0) return 0;
      if (prev >= newLength) return newLength - 1;
      return prev;
    });
  }, [images.length]);

  // Predict a single image by index
  const handlePredictOne = useCallback(async (index) => {
    setError("");
    setBulkSummary(null);
    // Set loading for this image
    setImages((prev) =>
      prev.map((img, i) => (i === index ? { ...img, loading: true, failed: false, failError: null } : img))
    );

    try {
      const data = await predictDisease(images[index].file);
      setImages((prev) =>
        prev.map((img, i) =>
          i === index
            ? {
                ...img,
                prediction: data.disease || "Unknown",
                confidence: data.confidence || 0,
                loading: false,
                failed: false,
                failError: null,
              }
            : img
        )
      );
    } catch {
      setImages((prev) =>
        prev.map((img, i) =>
          i === index
            ? {
                ...img,
                prediction: null,
                confidence: 0,
                loading: false,
                failed: true,
                failError: "Failed to analyze image",
              }
            : img
        )
      );
      setError("Failed to connect to the server. Make sure the backend is running.");
    }
  }, [images]);

  // Predict all images at once via a single bulk API call
  const handlePredictAll = useCallback(async () => {
    if (images.length === 0) return;
    setError("");
    setBulkSummary(null);
    setPredictingAll(true);

    // Mark all images as loading
    setImages((prev) =>
      prev.map((img) => ({ ...img, loading: true, failed: false, failError: null }))
    );

    try {
      // Send all files in a single request
      const data = await predictBulk(images.map((img) => img.file));
      const { total, success, failed, results } = data;

      setImages((prev) =>
        prev.map((img, i) => {
          const result = results[i];
          if (result?.status === "success") {
            return {
              ...img,
              prediction: result.prediction || "Unknown",
              confidence: result.confidence || 0,
              loading: false,
              failed: false,
              failError: null,
            };
          }
          // Per-image failure — mark as failed but don't block others
          return {
            ...img,
            prediction: null,
            confidence: 0,
            loading: false,
            failed: true,
            failError: result?.error || "Error analyzing image",
          };
        })
      );

      // Set the bulk summary for the banner
      setBulkSummary({ total, success, failed });

      if (failed > 0 && failed < total) {
        setError(`${failed} of ${total} images could not be analyzed. Successfully predicted the rest.`);
      } else if (failed === total) {
        setError("All images failed to analyze. Please check the images and try again.");
      }
    } catch {
      setImages((prev) =>
        prev.map((img) => ({ ...img, loading: false }))
      );
      setError("Failed to connect to the server. Make sure the backend is running.");
    } finally {
      setPredictingAll(false);
    }
  }, [images]);

  const handleReset = () => {
    images.forEach((img) => URL.revokeObjectURL(img.preview));
    setImages([]);
    setCurrentIndex(0);
    setError("");
    setBulkSummary(null);
  };

  // Check if any image is currently loading
  const anyLoading = images.some((img) => img.loading);

  // ==================== RENDER ====================

  return (
    <div className="min-h-screen flex flex-col transition-colors duration-400 bg-gradient-to-br from-slate-50 via-blue-50/20 to-cyan-50/10 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      {/* Navbar */}
      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center px-4 sm:px-6 lg:px-8 py-10 gap-10 max-w-5xl mx-auto w-full">

        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="text-center space-y-4 max-w-2xl"
        >
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-slate-800 dark:text-white">
            Plant Disease{" "}
            <span className="bg-gradient-to-r from-blue-500 to-cyan-400 bg-clip-text text-transparent">
              Detection
            </span>{" "}
            System
          </h2>
          <p className="text-slate-500 dark:text-slate-400 text-base sm:text-lg leading-relaxed max-w-lg mx-auto">
            Upload photos of your plants and let our AI identify diseases instantly.
            Powered by deep learning for accurate diagnoses.
          </p>
        </motion.div>

        {/* Upload Area — shown when no images, or as "add more" when images exist */}
        {images.length === 0 && (
          <UploadArea onImagesSelected={handleImagesSelected} disabled={anyLoading} />
        )}

        {/* Image Preview + Actions */}
        {images.length > 0 && (
          <>
            <ImagePreview
              images={images}
              currentIndex={currentIndex}
              setCurrentIndex={setCurrentIndex}
              onRemoveImage={handleRemoveImage}
              onPredictOne={handlePredictOne}
            />

            {/* Add more images */}
            <UploadArea onImagesSelected={handleImagesSelected} disabled={anyLoading} />

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="flex flex-wrap gap-4 justify-center"
            >
              {/* Predict All button — only shown if more than 1 image */}
              {images.length > 1 && (
                <motion.button
                  whileHover={{ scale: 1.03, y: -1 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={handlePredictAll}
                  disabled={anyLoading || predictingAll}
                  className="px-8 py-3.5 rounded-2xl font-semibold text-white text-sm bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 shadow-lg shadow-blue-500/25 dark:shadow-blue-500/15 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                >
                  🔬 Predict All ({images.length})
                </motion.button>
              )}

              {/* Reset button */}
              <motion.button
                whileHover={{ scale: 1.03, y: -1 }}
                whileTap={{ scale: 0.97 }}
                onClick={handleReset}
                disabled={anyLoading}
                className="px-8 py-3.5 rounded-2xl font-semibold text-sm transition-all duration-200 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed bg-white text-slate-600 border border-slate-200 hover:bg-slate-50 hover:border-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700 dark:hover:bg-slate-700 dark:hover:border-slate-600"
              >
                🗑️ Clear All
              </motion.button>
            </motion.div>
          </>
        )}

        {/* Bulk Prediction Summary Banner */}
        <AnimatePresence>
          {bulkSummary && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="w-full max-w-2xl rounded-2xl border overflow-hidden bg-white/80 backdrop-blur-sm border-slate-200 shadow-lg dark:bg-slate-800/80 dark:border-slate-700 dark:shadow-xl dark:shadow-slate-950/40"
            >
              {/* Banner header */}
              <div className="px-5 py-3.5 border-b border-slate-100 dark:border-slate-700/60 flex items-center gap-2.5">
                <span className="w-7 h-7 rounded-lg bg-blue-100 dark:bg-blue-500/20 flex items-center justify-center text-sm">📊</span>
                <h3 className="text-sm font-bold text-slate-700 dark:text-slate-200">Bulk Prediction Summary</h3>
              </div>

              {/* Stats grid */}
              <div className="grid grid-cols-3 gap-4 p-5">
                {/* Total */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="text-center p-4 rounded-xl bg-slate-50 dark:bg-slate-700/50 border border-slate-100 dark:border-slate-600/50"
                >
                  <p className="text-2xl font-extrabold text-slate-700 dark:text-white">{bulkSummary.total}</p>
                  <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 mt-1 uppercase tracking-wider">Total</p>
                </motion.div>

                {/* Success */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="text-center p-4 rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200/60 dark:border-emerald-500/20"
                >
                  <p className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">{bulkSummary.success}</p>
                  <p className="text-xs font-semibold text-emerald-500/70 dark:text-emerald-500/50 mt-1 uppercase tracking-wider">Success</p>
                </motion.div>

                {/* Failed */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className={`text-center p-4 rounded-xl border ${
                    bulkSummary.failed > 0
                      ? "bg-red-50 dark:bg-red-500/10 border-red-200/60 dark:border-red-500/20"
                      : "bg-slate-50 dark:bg-slate-700/50 border-slate-100 dark:border-slate-600/50"
                  }`}
                >
                  <p className={`text-2xl font-extrabold ${
                    bulkSummary.failed > 0
                      ? "text-red-600 dark:text-red-400"
                      : "text-slate-400 dark:text-slate-500"
                  }`}>{bulkSummary.failed}</p>
                  <p className={`text-xs font-semibold mt-1 uppercase tracking-wider ${
                    bulkSummary.failed > 0
                      ? "text-red-500/70 dark:text-red-500/50"
                      : "text-slate-400 dark:text-slate-500"
                  }`}>Failed</p>
                </motion.div>
              </div>

              {/* All-success congratulations bar */}
              {bulkSummary.failed === 0 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  className="px-5 pb-4"
                >
                  <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200/50 dark:border-emerald-500/15">
                    <span className="text-sm">✅</span>
                    <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">All images analyzed successfully!</span>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full max-w-2xl p-4 rounded-2xl text-center border bg-red-50 border-red-200 text-red-700 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-400"
            >
              ⚠️ {error}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
}

export default App;
