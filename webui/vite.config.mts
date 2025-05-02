import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { hmrPlugin, presets } from "vite-plugin-web-components-hmr";
import { defineConfig } from "vite";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [
    hmrPlugin({
      include: ["./src/**/*.ts"],
      presets: [presets.lit],
    }),
  ],
});
