import { http, HttpResponse, delay } from "msw";
import { initialState, initialMessages } from "../../../fixtures/dummy";

// Mock state updates for long-polling simulation
let currentState = { ...initialState };
const EMPTY_CONVERSATION =
  new URL(window.location.href).searchParams.get("emptyConversation") === "1";
const ADD_NEW_MESSAGES =
  new URL(window.location.href).searchParams.get("addNewMessages") === "1";

const messages = EMPTY_CONVERSATION ? [] : [...initialMessages];

export const handlers = [
  // Unified state endpoint that handles both regular and polling requests
  http.get("*/state", async ({ request }) => {
    const url = new URL(request.url);
    const isPoll = url.searchParams.get("poll") === "true";

    if (!isPoll) {
      // Regular state request
      return HttpResponse.json(currentState);
    }

    // This is a long-polling request
    await delay(ADD_NEW_MESSAGES ? 2000 : 60000); // Simulate waiting for changes

    if (ADD_NEW_MESSAGES) {
      // Simulate adding new messages
      messages.push({
        type: "agent",
        end_of_turn: false,
        content: "Here's a message",
        timestamp: "2025-04-24T10:32:29.072661+01:00",
        conversation_id: "37s-g6xg",
        usage: {
          input_tokens: 5,
          cache_creation_input_tokens: 250,
          cache_read_input_tokens: 4017,
          output_tokens: 92,
          cost_usd: 0.0035376,
        },
        start_time: "2025-04-24T10:32:26.99749+01:00",
        end_time: "2025-04-24T10:32:29.072654+01:00",
        elapsed: 2075193375,
        turnDuration: 28393844125,
        idx: messages.length,
      });

      // Update the state with new messages
      currentState = {
        ...currentState,
        message_count: messages.length,
      };
    }

    return HttpResponse.json(currentState);
  }),

  // Messages endpoint
  http.get("*/messages", ({ request }) => {
    const url = new URL(request.url);
    const startIndex = parseInt(url.searchParams.get("start") || "0");

    return HttpResponse.json(messages.slice(startIndex));
  }),
];
