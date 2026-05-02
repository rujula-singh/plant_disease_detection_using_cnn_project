// UploadArea.jsx — Premium drag & drop + file picker for image uploads
import { useState, useRef } from "react";
import { motion } from "framer-motion";

const UploadArea = ({ onImagesSelected, disabled }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Handle files from either drag-drop or file picker
  const handleFiles = (files) => {
    if (disabled) return;
    const imageFiles = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (imageFiles.length > 0) onImagesSelected(imageFiles);
  };

  const handleDragOver = (e) => { e.preventDefault(); if (!disabled) setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e) => { e.preventDefault(); setIsDragging(false); handleFiles(e.dataTransfer.files); };
  const handleFileChange = (e) => { handleFiles(e.target.files); e.target.value = ""; };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="w-full max-w-2xl mx-auto"
    >
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`
          group relative cursor-pointer rounded-2xl border-2 border-dashed p-12
          flex flex-col items-center justify-center gap-5
          transition-all duration-300 ease-in-out
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          ${isDragging
            ? "border-blue-400 bg-blue-50 dark:bg-blue-500/10 dark:border-blue-400 scale-[1.02] shadow-xl shadow-blue-500/10"
            : "border-slate-300 bg-white/60 shadow-sm hover:border-blue-400 hover:bg-blue-50/50 hover:shadow-lg dark:border-slate-600 dark:bg-slate-800/60 dark:hover:border-blue-400 dark:hover:bg-blue-500/5 dark:hover:shadow-blue-500/5"
          }
        `}
      >
        {/* Upload icon */}
        <motion.div
          animate={isDragging ? { scale: [1, 1.2, 1], y: [0, -8, 0] } : {}}
          transition={{ duration: 0.6, repeat: Infinity }}
          className={`w-16 h-16 rounded-2xl flex items-center justify-center text-3xl transition-all duration-300 ${
            isDragging
              ? "bg-blue-100 dark:bg-blue-500/20"
              : "bg-slate-100 group-hover:bg-blue-100 dark:bg-slate-700 dark:group-hover:bg-blue-500/20"
          }`}
        >
          {isDragging ? "📥" : "☁️"}
        </motion.div>

        {/* Instructions */}
        <div className="text-center space-y-2">
          <p className="text-lg font-semibold text-slate-700 dark:text-slate-200">
            {isDragging ? "Drop your images here!" : "Drag & drop plant images"}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            or{" "}
            <span className="text-blue-500 dark:text-blue-400 font-semibold underline underline-offset-2 decoration-blue-500/40">
              click to browse
            </span>
          </p>
          <p className="text-xs text-slate-400 dark:text-slate-500 pt-1">
            Supports JPG, PNG, WEBP • Multiple files allowed
          </p>
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
    </motion.div>
  );
};

export default UploadArea;
