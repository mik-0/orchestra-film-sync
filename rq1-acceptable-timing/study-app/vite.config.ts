import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vercel-ready Vite config. Media lives in public/media and is served at /media.
export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173 },
});
