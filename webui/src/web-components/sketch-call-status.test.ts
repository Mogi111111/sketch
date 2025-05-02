import { test, expect } from "@sand4rt/experimental-ct-web";
import { SketchCallStatus } from "./sketch-call-status";

test("initializes with zero LLM calls and empty tool calls by default", async ({
  mount,
}) => {
  const component = await mount(SketchCallStatus, {});

  // Check properties via component's evaluate method
  const llmCalls = await component.evaluate(
    (el: SketchCallStatus) => el.llmCalls,
  );
  expect(llmCalls).toBe(0);

  const toolCalls = await component.evaluate(
    (el: SketchCallStatus) => el.toolCalls,
  );
  expect(toolCalls).toEqual([]);

  // Check that indicators are not active
  await expect(component.locator(".llm-indicator")).not.toHaveClass(/active/);
  await expect(component.locator(".tool-indicator")).not.toHaveClass(/active/);
});

test("displays the correct state for active LLM calls", async ({ mount }) => {
  const component = await mount(SketchCallStatus, {
    props: {
      llmCalls: 3,
      toolCalls: [],
    },
  });

  // Check that LLM indicator is active
  await expect(component.locator(".llm-indicator")).toHaveClass(/active/);

  // Check that LLM indicator has the correct background and color
  await expect(component.locator(".llm-indicator.active")).toBeVisible();

  // Check that tool indicator is not active
  await expect(component.locator(".tool-indicator")).not.toHaveClass(/active/);
});

test("displays the correct state for active tool calls", async ({ mount }) => {
  const component = await mount(SketchCallStatus, {
    props: {
      llmCalls: 0,
      toolCalls: ["bash", "think"],
    },
  });

  // Check that tool indicator is active
  await expect(component.locator(".tool-indicator")).toHaveClass(/active/);

  // Check that tool indicator has the correct background and color
  await expect(component.locator(".tool-indicator.active")).toBeVisible();

  // Check that LLM indicator is not active
  await expect(component.locator(".llm-indicator")).not.toHaveClass(/active/);
});

test("displays both indicators when both call types are active", async ({
  mount,
}) => {
  const component = await mount(SketchCallStatus, {
    props: {
      llmCalls: 1,
      toolCalls: ["patch"],
    },
  });

  // Check that both indicators are active
  await expect(component.locator(".llm-indicator")).toHaveClass(/active/);
  await expect(component.locator(".tool-indicator")).toHaveClass(/active/);

  // Check that both active indicators are visible with their respective styles
  await expect(component.locator(".llm-indicator.active")).toBeVisible();
  await expect(component.locator(".tool-indicator.active")).toBeVisible();
});

test("has correct tooltip text for LLM calls", async ({ mount }) => {
  // Test with singular
  let component = await mount(SketchCallStatus, {
    props: {
      llmCalls: 1,
      toolCalls: [],
    },
  });

  await expect(component.locator(".llm-indicator")).toHaveAttribute(
    "title",
    "1 LLM call in progress",
  );

  await component.unmount();

  // Test with plural
  component = await mount(SketchCallStatus, {
    props: {
      llmCalls: 2,
      toolCalls: [],
    },
  });

  await expect(component.locator(".llm-indicator")).toHaveAttribute(
    "title",
    "2 LLM calls in progress",
  );
});

test("has correct tooltip text for tool calls", async ({ mount }) => {
  // Test with singular
  let component = await mount(SketchCallStatus, {
    props: {
      llmCalls: 0,
      toolCalls: ["bash"],
    },
  });

  await expect(component.locator(".tool-indicator")).toHaveAttribute(
    "title",
    "1 tool call in progress: bash",
  );

  await component.unmount();

  // Test with plural
  component = await mount(SketchCallStatus, {
    props: {
      llmCalls: 0,
      toolCalls: ["bash", "think"],
    },
  });

  await expect(component.locator(".tool-indicator")).toHaveAttribute(
    "title",
    "2 tool calls in progress: bash, think",
  );
});
