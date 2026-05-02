// Loader.jsx — Premium animated loading spinner
import { motion } from "framer-motion";

const Loader = () => {
  return (
    <div className="flex flex-col items-center justify-center gap-5 py-12">
      {/* Spinning ring with glow */}
      <div className="relative">
        <motion.div
          className="w-16 h-16 border-4 rounded-full border-slate-200 dark:border-slate-700 border-t-blue-500 dark:border-t-blue-400"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        />
        {/* Inner glow dot */}
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            className="w-3 h-3 rounded-full bg-blue-500 dark:bg-blue-400"
            animate={{ scale: [1, 1.4, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        </div>
      </div>
      {/* Pulsing text */}
      <motion.p
        className="text-slate-500 dark:text-slate-400 font-medium text-sm"
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        Analyzing plant image...
      </motion.p>
    </div>
  );
};

export default Loader;
