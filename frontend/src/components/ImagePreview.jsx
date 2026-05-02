// ImagePreview.jsx — Carousel preview with per-image Predict button and inline results
import { motion, AnimatePresence } from "framer-motion";

const ImagePreview = ({ images, currentIndex, setCurrentIndex, onRemoveImage, onPredictOne }) => {
  if (!images || images.length === 0) return null;

  const goToPrev = () => setCurrentIndex((prev) => (prev > 0 ? prev - 1 : images.length - 1));
  const goToNext = () => setCurrentIndex((prev) => (prev < images.length - 1 ? prev + 1 : 0));

  const currentImage = images[currentIndex];

  // Confidence color system
  const getConfidenceColors = (confidence) => {
    if (confidence >= 0.8) return {
      badge: "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/15 dark:text-emerald-400 dark:border-emerald-500/20",
      bar: "from-emerald-400 to-green-500",
    };
    if (confidence >= 0.5) return {
      badge: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-500/15 dark:text-amber-400 dark:border-amber-500/20",
      bar: "from-amber-400 to-yellow-500",
    };
    return {
      badge: "bg-red-50 text-red-700 border-red-200 dark:bg-red-500/15 dark:text-red-400 dark:border-red-500/20",
      bar: "from-red-400 to-rose-500",
    };
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="w-full max-w-2xl mx-auto"
    >
      {/* Header with counter badge */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-300 flex items-center gap-2">
          <span className="w-6 h-6 rounded-lg bg-blue-100 dark:bg-blue-500/20 flex items-center justify-center text-xs">📸</span>
          Uploaded Images
        </h3>
        <span className="text-xs font-bold px-3 py-1.5 rounded-full bg-blue-50 text-blue-600 dark:bg-blue-500/15 dark:text-blue-400 border border-blue-200/50 dark:border-blue-500/20">
          {currentIndex + 1} / {images.length}
        </span>
      </div>

      {/* Main carousel card */}
      <div className="relative rounded-2xl overflow-hidden border shadow-lg transition-all duration-300 bg-white border-slate-200 dark:bg-slate-800 dark:border-slate-700 dark:shadow-xl dark:shadow-slate-950/50">
        {/* Image display area */}
        <div className="relative h-72 sm:h-80 md:h-96 flex items-center justify-center bg-slate-50 dark:bg-slate-900/50">
          <AnimatePresence mode="wait">
            <motion.img
              key={currentIndex}
              src={currentImage?.preview}
              alt={`Upload ${currentIndex + 1}`}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.25 }}
              className="max-h-full max-w-full object-contain p-4"
            />
          </AnimatePresence>

          {/* Per-image loading overlay */}
          <AnimatePresence>
            {currentImage?.loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex flex-col items-center justify-center bg-white/70 dark:bg-slate-900/70 backdrop-blur-sm z-10"
              >
                <motion.div
                  className="w-12 h-12 border-4 rounded-full border-slate-200 dark:border-slate-700 border-t-blue-500 dark:border-t-blue-400"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                />
                <motion.p
                  className="text-slate-500 dark:text-slate-400 font-medium text-sm mt-3"
                  animate={{ opacity: [0.4, 1, 0.4] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  Analyzing...
                </motion.p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Remove button */}
          <motion.button
            whileHover={{ scale: 1.15 }}
            whileTap={{ scale: 0.85 }}
            onClick={() => onRemoveImage(currentIndex)}
            className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center rounded-full bg-red-500 hover:bg-red-600 text-white text-xs font-bold shadow-lg shadow-red-500/30 transition-colors duration-200 z-20"
            title="Remove this image"
          >
            ✕
          </motion.button>
        </div>

        {/* Bottom section: Predict button + inline result + failure state */}
        <div className="px-5 py-4 border-t border-slate-100 dark:border-slate-700/60">
          {/* File name */}
          <p className="text-xs text-slate-400 dark:text-slate-500 truncate font-medium mb-3" title={currentImage?.file?.name}>
            📄 {currentImage?.file?.name || `Image ${currentIndex + 1}`}
          </p>

          {/* Predict button for this image (shown only if no prediction AND not failed) */}
          {!currentImage?.prediction && !currentImage?.failed && (
            <motion.button
              whileHover={{ scale: 1.02, y: -1 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onPredictOne(currentIndex)}
              disabled={currentImage?.loading}
              className="w-full py-3 rounded-xl font-semibold text-white text-sm bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 shadow-md shadow-blue-500/20 dark:shadow-blue-500/10 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              {currentImage?.loading ? "⏳ Analyzing..." : "🔬 Predict Disease"}
            </motion.button>
          )}

          {/* Per-image failure state */}
          <AnimatePresence>
            {currentImage?.failed && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.35 }}
                className="space-y-3"
              >
                {/* Error badge */}
                <div className="flex items-start gap-3 p-3.5 rounded-xl bg-red-50 dark:bg-red-500/10 border border-red-200/60 dark:border-red-500/20">
                  <span className="w-8 h-8 flex-shrink-0 rounded-lg bg-red-100 dark:bg-red-500/20 flex items-center justify-center text-sm mt-0.5">❌</span>
                  <div className="min-w-0">
                    <p className="text-sm font-bold text-red-700 dark:text-red-400">Prediction Failed</p>
                    <p className="text-xs text-red-500/80 dark:text-red-400/60 mt-0.5 break-words">
                      {currentImage.failError || "Unknown error"}
                    </p>
                  </div>
                </div>

                {/* Retry button */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => onPredictOne(currentIndex)}
                  disabled={currentImage?.loading}
                  className="w-full py-2.5 rounded-xl font-medium text-sm transition-all duration-200 bg-red-50 text-red-600 hover:bg-red-100 border border-red-200/50 dark:bg-red-500/10 dark:text-red-400 dark:hover:bg-red-500/20 dark:border-red-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  🔄 Retry Prediction
                </motion.button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Inline result when prediction exists */}
          <AnimatePresence>
            {currentImage?.prediction && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.35 }}
                className="space-y-3"
              >
                {/* Disease name */}
                <h4 className="text-lg font-bold text-slate-800 dark:text-white leading-tight">
                  🦠 {currentImage.prediction}
                </h4>

                {/* Confidence bar */}
                {currentImage.confidence > 0 && (() => {
                  const percent = (currentImage.confidence * 100).toFixed(1);
                  const colors = getConfidenceColors(currentImage.confidence);
                  return (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-slate-500 dark:text-slate-400">Confidence</span>
                        <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${colors.badge}`}>
                          {percent}%
                        </span>
                      </div>
                      <div className="w-full rounded-full h-2.5 overflow-hidden bg-slate-200 dark:bg-slate-700">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${percent}%` }}
                          transition={{ duration: 0.8, ease: "easeOut" }}
                          className={`h-full rounded-full bg-gradient-to-r ${colors.bar}`}
                        />
                      </div>
                    </div>
                  );
                })()}

                {/* Re-predict button */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => onPredictOne(currentIndex)}
                  disabled={currentImage?.loading}
                  className="w-full py-2.5 rounded-xl font-medium text-sm transition-all duration-200 bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-700 dark:text-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  🔄 Re-predict
                </motion.button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Navigation buttons */}
        {images.length > 1 && (
          <>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={goToPrev}
              className="absolute left-3 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-xl shadow-lg transition-all duration-200 bg-white/90 hover:bg-white text-slate-700 border border-slate-200 dark:bg-slate-800/90 dark:hover:bg-slate-700 dark:text-slate-200 dark:border-slate-600 dark:shadow-slate-950/30 backdrop-blur-sm"
            >
              ◀
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={goToNext}
              className="absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-xl shadow-lg transition-all duration-200 bg-white/90 hover:bg-white text-slate-700 border border-slate-200 dark:bg-slate-800/90 dark:hover:bg-slate-700 dark:text-slate-200 dark:border-slate-600 dark:shadow-slate-950/30 backdrop-blur-sm"
            >
              ▶
            </motion.button>
          </>
        )}
      </div>

      {/* Thumbnail strip */}
      {images.length > 1 && (
        <div className="flex gap-2 mt-4 overflow-x-auto pb-2 scrollbar-thin">
          {images.map((img, idx) => (
            <motion.button
              key={idx}
              whileHover={{ scale: 1.08 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setCurrentIndex(idx)}
              className={`
                relative flex-shrink-0 w-16 h-16 rounded-xl overflow-hidden border-2 transition-all duration-200
                ${idx === currentIndex
                  ? "border-blue-500 shadow-md shadow-blue-500/25 ring-2 ring-blue-500/20"
                  : "border-slate-200 dark:border-slate-700 opacity-50 hover:opacity-100 hover:border-slate-400 dark:hover:border-slate-500"
                }
              `}
            >
              <img src={img.preview} alt={`Thumb ${idx + 1}`} className="w-full h-full object-cover" />
              {/* Prediction status indicator on thumbnail */}
              {img.loading && (
                <div className="absolute inset-0 bg-white/60 dark:bg-slate-900/60 flex items-center justify-center">
                  <motion.div
                    className="w-5 h-5 border-2 rounded-full border-slate-300 border-t-blue-500"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                </div>
              )}
              {img.prediction && !img.loading && (
                <div className="absolute bottom-0.5 right-0.5 w-4 h-4 rounded-full bg-emerald-500 flex items-center justify-center text-[8px] text-white font-bold shadow">
                  ✓
                </div>
              )}
              {img.failed && !img.loading && (
                <div className="absolute bottom-0.5 right-0.5 w-4 h-4 rounded-full bg-red-500 flex items-center justify-center text-[8px] text-white font-bold shadow">
                  ✕
                </div>
              )}
            </motion.button>
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default ImagePreview;
