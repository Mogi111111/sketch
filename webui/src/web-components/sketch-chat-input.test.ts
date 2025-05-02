import { test, expect } from "@sand4rt/experimental-ct-web";
import { SketchChatInput } from "./sketch-chat-input";

test("initializes with empty content by default", async ({ mount }) => {
  const component = await mount(SketchChatInput, {});

  // Check public property via component's evaluate method
  const content = await component.evaluate((el: SketchChatInput) => el.content);
  expect(content).toBe("");

  // Check textarea value
  await expect(component.locator("#chatInput")).toHaveValue("");
});

test("initializes with provided content", async ({ mount }) => {
  const testContent = "Hello, world!";
  const component = await mount(SketchChatInput, {
    props: {
      content: testContent,
    },
  });

  // Check public property via component's evaluate method
  const content = await component.evaluate((el: SketchChatInput) => el.content);
  expect(content).toBe(testContent);

  // Check textarea value
  await expect(component.locator("#chatInput")).toHaveValue(testContent);
});

test("updates content when typing in the textarea", async ({ mount }) => {
  const component = await mount(SketchChatInput, {});
  const newValue = "New message";

  // Fill the textarea with new content
  await component.locator("#chatInput").fill(newValue);

  // Check that the content property was updated
  const content = await component.evaluate((el: SketchChatInput) => el.content);
  expect(content).toBe(newValue);
});

test("sends message when clicking the send button", async ({ mount }) => {
  const testContent = "Test message";
  const component = await mount(SketchChatInput, {
    props: {
      content: testContent,
    },
  });

  // Set up promise to wait for the event
  const eventPromise = component.evaluate((el) => {
    return new Promise((resolve) => {
      el.addEventListener(
        "send-chat",
        (event) => {
          resolve((event as CustomEvent).detail);
        },
        { once: true },
      );
    });
  });

  // Click the send button
  await component.locator("#sendChatButton").click();

  // Wait for the event and check its details
  const detail: any = await eventPromise;
  expect(detail.message).toBe(testContent);
});

test.skip("sends message when pressing Enter (without shift)", async ({
  mount,
}) => {
  const testContent = "Test message";
  const component = await mount(SketchChatInput, {
    props: {
      content: testContent,
    },
  });

  // Set up promise to wait for the event
  const eventPromise = component.evaluate((el) => {
    return new Promise((resolve) => {
      el.addEventListener(
        "send-chat",
        (event) => {
          resolve((event as CustomEvent).detail);
        },
        { once: true },
      );
    });
  });

  // Press Enter in the textarea
  await component.locator("#chatInput").press("Enter");

  // Wait for the event and check its details
  const detail: any = await eventPromise;
  expect(detail.message).toBe(testContent);

  // Check that content was cleared
  const content = await component.evaluate((el: SketchChatInput) => el.content);
  expect(content).toBe("");
});

test.skip("does not send message when pressing Shift+Enter", async ({
  mount,
}) => {
  const testContent = "Test message";
  const component = await mount(SketchChatInput, {
    props: {
      content: testContent,
    },
  });

  // Set up to track if event fires
  let eventFired = false;
  await component.evaluate((el) => {
    el.addEventListener("send-chat", () => {
      (window as any).__eventFired = true;
    });
    (window as any).__eventFired = false;
  });

  // Press Shift+Enter in the textarea
  await component.locator("#chatInput").press("Shift+Enter");

  // Wait a short time and check if event fired
  await new Promise((resolve) => setTimeout(resolve, 50));
  eventFired = await component.evaluate(() => (window as any).__eventFired);
  expect(eventFired).toBe(false);

  // Check that content was not cleared
  const content = await component.evaluate((el: SketchChatInput) => el.content);
  expect(content).toBe(testContent);
});

test("resizes when user enters more text than will fit", async ({ mount }) => {
  const testContent = "Test message\n\n\n\n\n\n\n\n\n\n\n\n\nends here.";
  const component = await mount(SketchChatInput, {
    props: {
      content: "",
    },
  });
  const origHeight = await component.evaluate(
    (el: SketchChatInput) => el.chatInput.style.height,
  );

  // Enter very tall text in the textarea
  await component.locator("#chatInput").fill(testContent);

  // Wait for the requestAnimationFrame to complete
  await component.evaluate(() => new Promise(requestAnimationFrame));

  // Check that textarea resized
  const newHeight = await component.evaluate(
    (el: SketchChatInput) => el.chatInput.style.height,
  );
  expect(Number.parseInt(newHeight)).toBeGreaterThan(
    Number.parseInt(origHeight),
  );
});
