// Footer.jsx — Clean footer with proper dark mode
const Footer = () => {
  return (
    <footer className="w-full py-6 px-6 text-center border-t transition-all duration-300 bg-white/60 border-slate-200 dark:bg-slate-900/60 dark:border-slate-800 backdrop-blur-sm">
      <p className="text-sm text-slate-500 dark:text-slate-400">
        Built with 💚 using React + Tailwind CSS
      </p>
      <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
        PlantGuard AI &copy; {new Date().getFullYear()} — Plant Disease Detection System
      </p>
    </footer>
  );
};

export default Footer;
