import { puppeteerLauncher } from "@web/test-runner-puppeteer";

const filteredLogs = ["Running in dev mode", "Lit is in dev mode"];

export default /** @type {import("@web/test-runner").TestRunnerConfig} */ ({
  /** Test files to run */
  files: "dist/**/*.test.js",
  browsers: [puppeteerLauncher({ concurrency: 1 })],

  /** Resolve bare module imports */
  nodeResolve: {
    exportConditions: ["browser", "development"],
  },

  /** Filter out lit dev mode logs */
  filterBrowserLogs(log) {
    for (const arg of log.args) {
      if (
        typeof arg === "string" &&
        filteredLogs.some((l) => arg.includes(l))
      ) {
        return false;
      }
    }
    return true;
  },
});
