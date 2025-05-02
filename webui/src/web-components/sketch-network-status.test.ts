import { test, expect } from "@sand4rt/experimental-ct-web";
import { SketchNetworkStatus } from "./sketch-network-status";

// Test for when no error message is present - component should not render
test("does not display anything when no error is provided", async ({
  mount,
}) => {
  const component = await mount(SketchNetworkStatus, {
    props: {
      connection: "connected",
    },
  });

  // The component should be empty
  await expect(component.locator(".status-container")).not.toBeVisible();
});

// Test that error message is displayed correctly
test("displays error message when provided", async ({ mount }) => {
  const errorMsg = "Connection error";
  const component = await mount(SketchNetworkStatus, {
    props: {
      connection: "disconnected",
      error: errorMsg,
    },
  });

  await expect(component.locator(".status-text")).toBeVisible();
  await expect(component.locator(".status-text")).toContainText(errorMsg);
});
