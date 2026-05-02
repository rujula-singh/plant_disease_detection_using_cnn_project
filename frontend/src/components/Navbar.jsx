// Navbar.jsx — Glassmorphism navbar with dark mode toggle
import { motion } from "framer-motion";

const Navbar = ({ darkMode, setDarkMode }) => {
  return (
    <nav className="w-full px-6 py-4 flex items-center justify-between sticky top-0 z-50 border-b transition-all duration-300 bg-white/70 border-gray-200/80 shadow-sm dark:bg-slate-900/70 dark:border-slate-700/50 dark:shadow-lg dark:shadow-slate-950/50 backdrop-blur-xl">
      {/* App Logo & Title */}
      <div className="flex items-center gap-3">
        {/* Glowing icon container */}
        <div className="w-9 h-9 rounded-xl flex items-center justify-center bg-gradient-to-br from-blue-500 to-cyan-400 shadow-md shadow-blue-500/30 dark:shadow-blue-500/20">
          <span className="text-lg text-white">🌿</span>
        </div>
        <h1 className="text-xl font-bold tracking-tight text-slate-800 dark:text-white">
          Plant<span className="bg-gradient-to-r from-blue-500 to-cyan-400 bg-clip-text text-transparent">Guard</span> AI
        </h1>
      </div>

      {/* Dark Mode Toggle — smooth animated switch */}
      <motion.button
        whileTap={{ scale: 0.85, rotate: 15 }}
        whileHover={{ scale: 1.05 }}
        onClick={() => setDarkMode(!darkMode)}
        className="relative w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300 bg-slate-100 hover:bg-slate-200 text-slate-600 dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-yellow-300 shadow-sm dark:shadow-slate-700/30"
        aria-label="Toggle dark mode"
      >
        <motion.span
          key={darkMode ? "sun" : "moon"}
          initial={{ rotate: -90, opacity: 0 }}
          animate={{ rotate: 0, opacity: 1 }}
          exit={{ rotate: 90, opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="text-lg"
        >
          {darkMode ? "☀️" : "🌙"}
        </motion.span>
      </motion.button>
    </nav>
  );
};

export default Navbar;
