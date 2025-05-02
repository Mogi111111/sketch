import { test, expect } from "@sand4rt/experimental-ct-web";
import { SketchContainerStatus } from "./sketch-container-status";
import { State } from "../types";

// Mock complete state for testing
const mockCompleteState: State = {
  hostname: "test-host",
  working_dir: "/test/dir",
  initial_commit: "abcdef1234567890",
  message_count: 42,
  os: "linux",
  title: "Test Session",
  total_usage: {
    input_tokens: 1000,
    output_tokens: 2000,
    cache_read_input_tokens: 300,
    cache_creation_input_tokens: 400,
    total_cost_usd: 0.25,
    start_time: "",
    messages: 0,
    tool_uses: {},
  },
  outstanding_llm_calls: 0,
  outstanding_tool_calls: [],
  session_id: "test-session-id",
  ssh_available: false,
  in_container: true,
  first_message_index: 0,
};

test("render props", async ({ mount }) => {
  const component = await mount(SketchContainerStatus, {
    props: {
      state: mockCompleteState,
    },
  });
  await expect(component.locator("#hostname")).toContainText(
    mockCompleteState.hostname,
  );
  // Check that all expected elements exist
  await expect(component.locator("#workingDir")).toContainText(
    mockCompleteState.working_dir,
  );
  await expect(component.locator("#initialCommit")).toContainText(
    mockCompleteState.initial_commit.substring(0, 8),
  );

  await expect(component.locator("#messageCount")).toContainText(
    mockCompleteState.message_count + "",
  );
  const expectedTotalInputTokens =
    mockCompleteState.total_usage.input_tokens +
    mockCompleteState.total_usage.cache_read_input_tokens +
    mockCompleteState.total_usage.cache_creation_input_tokens;
  await expect(component.locator("#inputTokens")).toContainText(
    expectedTotalInputTokens.toLocaleString(),
  );
  await expect(component.locator("#outputTokens")).toContainText(
    mockCompleteState.total_usage.output_tokens.toLocaleString(),
  );
  await expect(component.locator("#totalCost")).toContainText(
    "$" + mockCompleteState.total_usage.total_cost_usd.toFixed(2),
  );
});

test("renders with undefined state", async ({ mount }) => {
  const component = await mount(SketchContainerStatus, {});

  // Elements should exist but be empty
  await expect(component.locator("#hostname")).toContainText("");
  await expect(component.locator("#workingDir")).toContainText("");
  await expect(component.locator("#initialCommit")).toContainText("");
  await expect(component.locator("#messageCount")).toContainText("");
  await expect(component.locator("#inputTokens")).toContainText("0");
  await expect(component.locator("#outputTokens")).toContainText("");
  await expect(component.locator("#totalCost")).toContainText("$0.00");
});

test("renders with partial state data", async ({ mount }) => {
  const partialState: Partial<State> = {
    hostname: "partial-host",
    message_count: 10,
    os: "linux",
    title: "Partial Test",
    session_id: "partial-session",
    ssh_available: false,
    total_usage: {
      input_tokens: 500,
      start_time: "",
      messages: 0,
      output_tokens: 0,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
      total_cost_usd: 0,
      tool_uses: {},
    },
  };

  const component = await mount(SketchContainerStatus, {
    props: {
      state: partialState as State,
    },
  });

  // Check that elements with data are properly populated
  await expect(component.locator("#hostname")).toContainText("partial-host");
  await expect(component.locator("#messageCount")).toContainText("10");

  const expectedTotalInputTokens =
    partialState.total_usage.input_tokens +
    partialState.total_usage.cache_read_input_tokens +
    partialState.total_usage.cache_creation_input_tokens;
  await expect(component.locator("#inputTokens")).toContainText(
    expectedTotalInputTokens.toLocaleString(),
  );

  // Check that elements without data are empty
  await expect(component.locator("#workingDir")).toContainText("");
  await expect(component.locator("#initialCommit")).toContainText("");
  await expect(component.locator("#outputTokens")).toContainText("0");
  await expect(component.locator("#totalCost")).toContainText("$0.00");
});

test("handles cost formatting correctly", async ({ mount }) => {
  // Test with different cost values
  const testCases = [
    { cost: 0, expected: "$0.00" },
    { cost: 0.1, expected: "$0.10" },
    { cost: 1.234, expected: "$1.23" },
    { cost: 10.009, expected: "$10.01" },
  ];

  for (const testCase of testCases) {
    const stateWithCost = {
      ...mockCompleteState,
      total_usage: {
        ...mockCompleteState.total_usage,
        total_cost_usd: testCase.cost,
      },
    };

    const component = await mount(SketchContainerStatus, {
      props: {
        state: stateWithCost,
      },
    });
    await expect(component.locator("#totalCost")).toContainText(
      testCase.expected,
    );
    await component.unmount();
  }
});

test("truncates commit hash to 8 characters", async ({ mount }) => {
  const stateWithLongCommit = {
    ...mockCompleteState,
    initial_commit: "1234567890abcdef1234567890abcdef12345678",
  };

  const component = await mount(SketchContainerStatus, {
    props: {
      state: stateWithLongCommit,
    },
  });

  await expect(component.locator("#initialCommit")).toContainText("12345678");
});

test("has correct link elements", async ({ mount }) => {
  const component = await mount(SketchContainerStatus, {
    props: {
      state: mockCompleteState,
    },
  });

  // Check for logs link
  const logsLink = component.locator("a").filter({ hasText: "Logs" });
  await expect(logsLink).toHaveAttribute("href", "logs");

  // Check for download link
  const downloadLink = component.locator("a").filter({ hasText: "Download" });
  await expect(downloadLink).toHaveAttribute("href", "download");
});
