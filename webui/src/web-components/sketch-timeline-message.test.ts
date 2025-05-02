import { test, expect } from "@sand4rt/experimental-ct-web";
import { SketchTimelineMessage } from "./sketch-timeline-message";
import {
  AgentMessage,
  CodingAgentMessageType,
  GitCommit,
  Usage,
} from "../types";

// Helper function to create mock timeline messages
function createMockMessage(props: Partial<AgentMessage> = {}): AgentMessage {
  return {
    idx: props.idx || 0,
    type: props.type || "agent",
    content: props.content || "Hello world",
    timestamp: props.timestamp || "2023-05-15T12:00:00Z",
    elapsed: props.elapsed || 1500000000, // 1.5 seconds in nanoseconds
    end_of_turn: props.end_of_turn || false,
    conversation_id: props.conversation_id || "conv123",
    tool_calls: props.tool_calls || [],
    commits: props.commits || [],
    usage: props.usage,
    ...props,
  };
}

test("renders with basic message content", async ({ mount }) => {
  const message = createMockMessage({
    type: "agent",
    content: "This is a test message",
  });

  const component = await mount(SketchTimelineMessage, {
    props: {
      message: message,
    },
  });

  await expect(component.locator(".message-text")).toBeVisible();
  await expect(component.locator(".message-text")).toContainText(
    "This is a test message",
  );
});

test.skip("renders with correct message type classes", async ({ mount }) => {
  const messageTypes: CodingAgentMessageType[] = [
    "user",
    "agent",
    "error",
    "budget",
    "tool",
    "commit",
    "auto",
  ];

  for (const type of messageTypes) {
    const message = createMockMessage({ type });

    const component = await mount(SketchTimelineMessage, {
      props: {
        message: message,
      },
    });

    await expect(component.locator(".message")).toBeVisible();
    await expect(component.locator(`.message.${type}`)).toBeVisible();
  }
});

test("renders end-of-turn marker correctly", async ({ mount }) => {
  const message = createMockMessage({
    end_of_turn: true,
  });

  const component = await mount(SketchTimelineMessage, {
    props: {
      message: message,
    },
  });

  await expect(component.locator(".message")).toBeVisible();
  await expect(component.locator(".message.end-of-turn")).toBeVisible();
});

test("formats timestamps correctly", async ({ mount }) => {
  const message = createMockMessage({
    timestamp: "2023-05-15T12:00:00Z",
  });

  const component = await mount(SketchTimelineMessage, {
    props: {
      message: message,
    },
  });

  await expect(component.locator(".message-timestamp")).toBeVisible();
  // Should include a formatted date like "May 15, 2023"
  await expect(component.locator(".message-timestamp")).toContainText(
    "May 15, 2023",
  );
  // Should include elapsed time
  await expect(component.locator(".message-timestamp")).toContainText(
    "(1.50s)",
  );
});

test("renders markdown content correctly", async ({ mount }) => {
  const markdownContent =
    "# Heading\n\n- List item 1\n- List item 2\n\n`code block`";
  const message = createMockMessage({
    content: markdownContent,
  });

  const component = await mount(SketchTimelineMessage, {
    props: {
      message: message,
    },
  });

  await expect(component.locator(".markdown-content")).toBeVisible();

  // Check HTML content
  const html = await component
    .locator(".markdown-content")
    .evaluate((element) => element.innerHTML);
  expect(html).toContain("<h1>Heading</h1>");
  expect(html).toContain("<ul>");
  expect(html).toContain("<li>List item 1</li>");
  expect(html).toContain("<code>code block</code>");
});

test("displays usage information when available", async ({ mount }) => {
  const usage: Usage = {
    input_tokens: 150,
    output_tokens: 300,
    cost_usd: 0.025,
    cache_read_input_tokens: 50,
    cache_creation_input_tokens: 0,
  };

  const message = createMockMessage({
    usage,
  });

  const component = await mount(SketchTimelineMessage, {
    props: {
      message: message,
    },
  });

  await expect(component.locator(".message-usage")).toBeVisible();
  await expect(component.locator(".message-usage")).toContainText(
    "200".toLocaleString(),
  ); // In (150 + 50 cache)
  await expect(component.locator(".message-usage")).toContainText(
    "300".toLocaleString(),
  ); // Out
  await expect(component.locator(".message-usage")).toContainText("$0.03"); // Cost
});

test("renders commit information correctly", async ({ mount }) => {
  const commits: GitCommit[] = [
    {
      hash: "1234567890abcdef",
      subject: "Fix bug in application",
      body: "This fixes a major bug in the application\n\nSigned-off-by: Developer",
      pushed_branch: "main",
    },
  ];

  const message = createMockMessage({
    commits,
  });

  const component = await mount(SketchTimelineMessage, {
    props: {
      message: message,
    },
  });

  await expect(component.locator(".commits-container")).toBeVisible();
  await expect(component.locator(".commits-header")).toBeVisible();
  await expect(component.locator(".commits-header")).toContainText("1 new");

  await expect(component.locator(".commit-hash")).toBeVisible();
  await expect(component.locator(".commit-hash")).toHaveText("12345678"); // First 8 chars

  await expect(component.locator(".pushed-branch")).toBeVisible();
  await expect(component.locator(".pushed-branch")).toContainText("main");
});

test("dispatches show-commit-diff event when commit diff button is clicked", async ({
  mount,
}) => {
  const commits: GitCommit[] = [
    {
      hash: "1234567890abcdef",
      subject: "Fix bug in application",
      body: "This fixes a major bug in the application",
      pushed_branch: "main",
    },
  ];

  const message = createMockMessage({
    commits,
  });

  const component = await mount(SketchTimelineMessage, {
    props: {
      message: message,
    },
  });

  await expect(component.locator(".commit-diff-button")).toBeVisible();

  // Set up promise to wait for the event
  const eventPromise = component.evaluate((el) => {
    return new Promise((resolve) => {
      el.addEventListener(
        "show-commit-diff",
        (event) => {
          resolve((event as CustomEvent).detail);
        },
        { once: true },
      );
    });
  });

  // Click the diff button
  await component.locator(".commit-diff-button").click();

  // Wait for the event and check its details
  const detail = await eventPromise;
  expect(detail["commitHash"]).toBe("1234567890abcdef");
});

test.skip("handles message type icon display correctly", async ({ mount }) => {
  // First message of a type should show icon
  const firstMessage = createMockMessage({
    type: "user",
    idx: 0,
  });

  // Second message of same type should not show icon
  const secondMessage = createMockMessage({
    type: "user",
    idx: 1,
  });

  // Test first message (should show icon)
  const firstComponent = await mount(SketchTimelineMessage, {
    props: {
      message: firstMessage,
    },
  });

  await expect(firstComponent.locator(".message-icon")).toBeVisible();
  await expect(firstComponent.locator(".message-icon")).toHaveText("U");

  // Test second message with previous message of same type
  const secondComponent = await mount(SketchTimelineMessage, {
    props: {
      message: secondMessage,
      previousMessage: firstMessage,
    },
  });

  await expect(secondComponent.locator(".message-icon")).not.toBeVisible();
});

test("formats numbers correctly", async ({ mount }) => {
  const component = await mount(SketchTimelineMessage, {});

  // Test accessing public method via evaluate
  const result1 = await component.evaluate((el: SketchTimelineMessage) =>
    el.formatNumber(1000),
  );
  expect(result1).toBe("1,000");

  const result2 = await component.evaluate((el: SketchTimelineMessage) =>
    el.formatNumber(null, "N/A"),
  );
  expect(result2).toBe("N/A");

  const result3 = await component.evaluate((el: SketchTimelineMessage) =>
    el.formatNumber(undefined, "--"),
  );
  expect(result3).toBe("--");
});

test("formats currency values correctly", async ({ mount }) => {
  const component = await mount(SketchTimelineMessage, {});

  // Test with different precisions
  const result1 = await component.evaluate((el: SketchTimelineMessage) =>
    el.formatCurrency(10.12345, "$0.00", true),
  );
  expect(result1).toBe("$10.1235"); // message level (4 decimals)

  const result2 = await component.evaluate((el: SketchTimelineMessage) =>
    el.formatCurrency(10.12345, "$0.00", false),
  );
  expect(result2).toBe("$10.12"); // total level (2 decimals)

  const result3 = await component.evaluate((el: SketchTimelineMessage) =>
    el.formatCurrency(null, "N/A"),
  );
  expect(result3).toBe("N/A");

  const result4 = await component.evaluate((el: SketchTimelineMessage) =>
    el.formatCurrency(undefined, "--"),
  );
  expect(result4).toBe("--");
});
