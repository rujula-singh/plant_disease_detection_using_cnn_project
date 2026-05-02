// ResultCard.jsx — Rich prediction result card with confidence visualization
import { motion } from "framer-motion";

const ResultCard = ({ imageName, imageUrl, disease, confidence, index }) => {
  const percent = (confidence * 100).toFixed(1);

  // Color system based on confidence level
  const getColors = () => {
    if (confidence >= 0.8) return {
      badge: "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/15 dark:text-emerald-400 dark:border-emerald-500/20",
      bar: "from-emerald-400 to-green-500",
      glow: "shadow-emerald-500/20",
    };
    if (confidence >= 0.5) return {
      badge: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-500/15 dark:text-amber-400 dark:border-amber-500/20",
      bar: "from-amber-400 to-yellow-500",
      glow: "shadow-amber-500/20",
    };
    return {
      badge: "bg-red-50 text-red-700 border-red-200 dark:bg-red-500/15 dark:text-red-400 dark:border-red-500/20",
      bar: "from-red-400 to-rose-500",
      glow: "shadow-red-500/20",
    };
  };

  const colors = getColors();

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, delay: index * 0.1, ease: "easeOut" }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className="rounded-2xl overflow-hidden transition-all duration-300 bg-white border border-slate-200 shadow-md hover:shadow-xl dark:bg-slate-800 dark:border-slate-700 dark:shadow-xl dark:shadow-slate-950/40 dark:hover:shadow-2xl dark:hover:shadow-slate-950/60"
    >
      {/* Image thumbnail with overlay gradient */}
      <div className="relative h-48 overflow-hidden bg-slate-100 dark:bg-slate-900">
        <img
          src={imageUrl}
          alt={imageName}
          className="w-full h-full object-cover transition-transform duration-500 hover:scale-110"
        />
        {/* Bottom gradient overlay */}
        <div className="absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-black/30 to-transparent" />
      </div>

      {/* Result content */}
      <div className="p-5 space-y-4">
        {/* File name tag */}
        <p className="text-xs text-slate-400 dark:text-slate-500 truncate font-medium" title={imageName}>
          📄 {imageName}
        </p>

        {/* Disease name */}
        <h3 className="text-lg font-bold text-slate-800 dark:text-white leading-tight">
          🦠 {disease}
        </h3>

        {/* Confidence section */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-slate-500 dark:text-slate-400">Confidence</span>
            <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${colors.badge}`}>
              {percent}%
            </span>
          </div>

          {/* Animated progress bar */}
          <div className="w-full rounded-full h-2.5 overflow-hidden bg-slate-200 dark:bg-slate-700">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${percent}%` }}
              transition={{ duration: 1, delay: index * 0.1 + 0.3, ease: "easeOut" }}
              className={`h-full rounded-full bg-gradient-to-r ${colors.bar} shadow-sm ${colors.glow}`}
            />
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ResultCard;
