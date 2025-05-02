//go:build goexperiment.synctest

package loop

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"testing/synctest"

	"sketch.dev/ant"
)

func TestLoop_OneTurn_Basic(t *testing.T) {
	synctest.Run(func() {
		mockConvo := NewMockConvo(t)

		agent := &Agent{
			convo:  mockConvo,
			inbox:  make(chan string, 1),
			outbox: make(chan AgentMessage, 1),
		}
		userMsg := ant.UserStringMessage("hi")
		userMsgResponse := &ant.MessageResponse{}
		mockConvo.ExpectCall("SendMessage", userMsg).Return(userMsgResponse, nil)

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		go agent.Loop(ctx)

		agent.UserMessage(ctx, "hi")

		// This makes sure the SendMessage call happens before we assert the expectations.
		synctest.Wait()

		// Verify results
		mockConvo.AssertExpectations(t)
	})
}

func TestLoop_ToolCall_Basic(t *testing.T) {
	synctest.Run(func() {
		mockConvo := NewMockConvo(t)

		agent := &Agent{
			convo:  mockConvo,
			inbox:  make(chan string, 1),
			outbox: make(chan AgentMessage, 1),
		}
		userMsg := ant.Message{
			Role: ant.MessageRoleUser,
			Content: []ant.Content{
				{Type: ant.ContentTypeText, Text: "hi"},
			},
		}
		userMsgResponse := &ant.MessageResponse{
			StopReason: ant.StopReasonToolUse,
			Content: []ant.Content{
				{
					Type:      ant.ContentTypeToolUse,
					ID:        "tool1",
					ToolName:  "test_tool",
					ToolInput: []byte(`{"param":"value"}`),
				},
			},
			Usage: ant.Usage{
				InputTokens:  100,
				OutputTokens: 200,
			},
		}

		toolUseContents := []ant.Content{
			{
				Type:       ant.ContentTypeToolResult,
				ToolUseID:  "tool1",
				Text:       "",
				ToolResult: "This is a tool result",
				ToolError:  false,
			},
		}
		toolUseResultsMsg := ant.Message{
			Role:    ant.MessageRoleUser,
			Content: toolUseContents,
		}
		toolUseResponse := &ant.MessageResponse{
			StopReason: ant.StopReasonEndTurn,
			Content: []ant.Content{
				{
					Type: ant.ContentTypeText,
					Text: "tool_use contents accepted",
				},
			},
			Usage: ant.Usage{
				InputTokens:  50,
				OutputTokens: 75,
			},
		}

		// Set up the mock response for tool results
		mockConvo.ExpectCall("SendMessage", userMsg).Return(userMsgResponse, nil)
		mockConvo.ExpectCall("ToolResultContents", userMsgResponse).Return(toolUseContents, nil)
		mockConvo.ExpectCall("SendMessage", toolUseResultsMsg).Return(toolUseResponse, nil)

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		go agent.Loop(ctx)

		agent.UserMessage(ctx, "hi")

		// This makes sure the SendMessage call happens before we assert the expectations.
		synctest.Wait()

		// Verify results
		mockConvo.AssertExpectations(t)
	})
}

func TestLoop_ToolCall_UserCancelsDuringToolResultContents(t *testing.T) {
	synctest.Run(func() {
		mockConvo := NewMockConvo(t)

		agent := &Agent{
			convo:  mockConvo,
			inbox:  make(chan string, 1),
			outbox: make(chan AgentMessage, 10), // don't let anything block on outbox.
		}
		userMsg := ant.UserStringMessage("hi")
		userMsgResponse := &ant.MessageResponse{
			StopReason: ant.StopReasonToolUse,
			Content: []ant.Content{
				{
					Type:      ant.ContentTypeToolUse,
					ID:        "tool1",
					ToolName:  "test_tool",
					ToolInput: []byte(`{"param":"value"}`),
				},
			},
			Usage: ant.Usage{
				InputTokens:  100,
				OutputTokens: 200,
			},
		}
		toolUseResultsMsg := ant.UserStringMessage(cancelToolUseMessage)
		toolUseResponse := &ant.MessageResponse{
			StopReason: ant.StopReasonEndTurn,
			Content: []ant.Content{
				{
					Type: ant.ContentTypeText,
					Text: "tool_use contents accepted",
				},
			},
			Usage: ant.Usage{
				InputTokens:  50,
				OutputTokens: 75,
			},
		}

		// Set up the mock response for tool results

		userCancelError := fmt.Errorf("user canceled")
		// This allows the test to block the InnerLoop goroutine that invokes ToolResultsContents so
		// we can force its context to cancel while it's blocked.
		waitForToolResultContents := make(chan any, 1)

		mockConvo.ExpectCall("SendMessage", userMsg).Return(userMsgResponse, nil)
		mockConvo.ExpectCall("ToolResultContents",
			userMsgResponse).BlockAndReturn(waitForToolResultContents, []ant.Content{}, userCancelError)
		mockConvo.ExpectCall("SendMessage", toolUseResultsMsg).Return(toolUseResponse, nil)

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		go agent.Loop(ctx)

		// This puts one message into agent.inbox, which should un-block the GatherMessages call
		// at the top of agent.InnerLoop.
		agent.UserMessage(ctx, "hi")

		// This makes sure the first SendMessage call happens before we proceed with the cancel.
		synctest.Wait()

		// The goroutine executing ToolResultContents call should be blocked, simulating a long
		// running operation that the user wishes to cancel while it's still in progress.
		// This call invokes that InnerLoop context's cancel() func.
		agent.CancelInnerLoop(userCancelError)

		// This tells the goroutine that's in mockConvo.ToolResultContents to proceed.
		waitForToolResultContents <- nil

		// This makes sure the final SendMessage call happens before we assert the expectations.
		synctest.Wait()

		// Verify results
		mockConvo.AssertExpectations(t)
	})
}

func TestLoop_ToolCall_UserCancelsDuringToolResultContents_AndContinuesToChat(t *testing.T) {
	synctest.Run(func() {
		mockConvo := NewMockConvo(t)

		agent := &Agent{
			convo:  mockConvo,
			inbox:  make(chan string, 1),
			outbox: make(chan AgentMessage, 10), // don't let anything block on outbox.
		}
		userMsg := ant.Message{
			Role: ant.MessageRoleUser,
			Content: []ant.Content{
				{Type: ant.ContentTypeText, Text: "hi"},
			},
		}
		userMsgResponse := &ant.MessageResponse{
			StopReason: ant.StopReasonToolUse,
			Content: []ant.Content{
				{
					Type:      ant.ContentTypeToolUse,
					ID:        "tool1",
					ToolName:  "test_tool",
					ToolInput: []byte(`{"param":"value"}`),
				},
			},
			Usage: ant.Usage{
				InputTokens:  100,
				OutputTokens: 200,
			},
		}
		toolUseResultsMsg := ant.Message{
			Role: ant.MessageRoleUser,
			Content: []ant.Content{
				{Type: ant.ContentTypeText, Text: cancelToolUseMessage},
			},
		}
		toolUseResultResponse := &ant.MessageResponse{
			StopReason: ant.StopReasonEndTurn,
			Content: []ant.Content{
				{
					Type: ant.ContentTypeText,
					Text: "awaiting further instructions",
				},
			},
			Usage: ant.Usage{
				InputTokens:  50,
				OutputTokens: 75,
			},
		}
		userFollowUpMsg := ant.Message{
			Role: ant.MessageRoleUser,
			Content: []ant.Content{
				{Type: ant.ContentTypeText, Text: "that was the wrong thing to do"},
			},
		}
		userFollowUpResponse := &ant.MessageResponse{
			StopReason: ant.StopReasonEndTurn,
			Content: []ant.Content{
				{
					Type: ant.ContentTypeText,
					Text: "sorry about that",
				},
			},
			Usage: ant.Usage{
				InputTokens:  100,
				OutputTokens: 200,
			},
		}
		// Set up the mock response for tool results

		userCancelError := fmt.Errorf("user canceled")
		// This allows the test to block the InnerLoop goroutine that invokes ToolResultsContents so
		// we can force its context to cancel while it's blocked.
		waitForToolResultContents := make(chan any, 1)

		mockConvo.ExpectCall("SendMessage", userMsg).Return(userMsgResponse, nil)
		mockConvo.ExpectCall("ToolResultContents",
			userMsgResponse).BlockAndReturn(waitForToolResultContents, []ant.Content{}, userCancelError)
		mockConvo.ExpectCall("SendMessage", toolUseResultsMsg).Return(toolUseResultResponse, nil)

		mockConvo.ExpectCall("SendMessage", userFollowUpMsg).Return(userFollowUpResponse, nil)

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		go agent.Loop(ctx)

		// This puts one message into agent.inbox, which should un-block the GatherMessages call
		// at the top of agent.InnerLoop.
		agent.UserMessage(ctx, "hi")

		// This makes sure the first SendMessage call happens before we proceed with the cancel.
		synctest.Wait()

		// The goroutine executing ToolResultContents call should be blocked, simulating a long
		// running operation that the user wishes to cancel while it's still in progress.
		// This call invokes that InnerLoop context's cancel() func.
		agent.CancelInnerLoop(userCancelError)

		// This tells the goroutine that's in mockConvo.ToolResultContents to proceed.
		waitForToolResultContents <- nil

		// Allow InnerLoop to handle the cancellation logic before continuing the conversation.
		synctest.Wait()

		agent.UserMessage(ctx, "that was the wrong thing to do")

		synctest.Wait()

		// Verify results
		mockConvo.AssertExpectations(t)
	})
}

func TestInnerLoop_UserCancels(t *testing.T) {
	synctest.Run(func() {
		mockConvo := NewMockConvo(t)

		agent := &Agent{
			convo:  mockConvo,
			inbox:  make(chan string, 1),
			outbox: make(chan AgentMessage, 10), // don't block on outbox
		}

		// Define test message
		// This simulates something that would result in claude  responding with tool_use responses.
		userMsg := ant.UserStringMessage("use test_tool for something")
		// Mock initial response with tool use
		userMsgResponse := &ant.MessageResponse{
			StopReason: ant.StopReasonToolUse,
			Content: []ant.Content{
				{
					Type:      ant.ContentTypeToolUse,
					ID:        "tool1",
					ToolName:  "test_tool",
					ToolInput: []byte(`{"param":"value"}`),
				},
			},
			Usage: ant.Usage{
				InputTokens:  100,
				OutputTokens: 200,
			},
		}
		canceledToolUseContents := []ant.Content{
			{
				Type:       ant.ContentTypeToolResult,
				ToolUseID:  "tool1",
				ToolError:  true,
				ToolResult: "user canceled this tool_use",
			},
		}
		canceledToolUseMsg := ant.Message{
			Role:    ant.MessageRoleUser,
			Content: append(canceledToolUseContents, ant.StringContent(cancelToolUseMessage)),
		}
		// Set up expected behaviors
		waitForSendMessage := make(chan any)
		mockConvo.ExpectCall("SendMessage", userMsg).BlockAndReturn(waitForSendMessage, userMsgResponse, nil)

		mockConvo.ExpectCall("ToolResultCancelContents", userMsgResponse).Return(canceledToolUseContents, nil)
		mockConvo.ExpectCall("SendMessage", canceledToolUseMsg).Return(
			&ant.MessageResponse{
				StopReason: ant.StopReasonToolUse,
			}, nil)

		ctx, cancel := context.WithCancelCause(context.Background())

		// Run one iteration of InnerLoop
		go agent.InnerLoop(ctx)

		// Send a message to the agent's inbox
		agent.UserMessage(ctx, "use test_tool for something")

		synctest.Wait()

		// cancel the context before we even call InnerLoop with it, so it will
		// be .Done() the first time it checks.
		cancel(fmt.Errorf("user canceled"))

		// unblock the InnerLoop goroutine's SendMessage call
		waitForSendMessage <- nil

		synctest.Wait()

		// Verify results
		mockConvo.AssertExpectations(t)

		// Get all messages from outbox and verify their types/content
		var messages []AgentMessage

		// Collect messages until outbox is empty or we have 10 messages
		for i := 0; i < 10; i++ {
			select {
			case msg := <-agent.outbox:
				messages = append(messages, msg)
			default:
				// No more messages
				i = 10 // Exit the loop
			}
		}

		// Print out the messages we got for debugging
		t.Logf("Received %d messages from outbox", len(messages))
		for i, msg := range messages {
			t.Logf("Message %d: Type=%s, Content=%s, EndOfTurn=%t", i, msg.Type, msg.Content, msg.EndOfTurn)
			if msg.ToolName != "" {
				t.Logf("  Tool: Name=%s, Input=%s, Result=%s, Error=%v",
					msg.ToolName, msg.ToolInput, msg.ToolResult, msg.ToolError)
			}
		}

		// Basic checks
		if len(messages) < 1 {
			t.Errorf("Should have at least one message, got %d", len(messages))
		}

		// The main thing we want to verify: when user cancels, the response processing stops
		// and appropriate messages are sent

		// Check if we have an error message about cancellation
		hasCancelErrorMessage := false
		for _, msg := range messages {
			if msg.Type == ErrorMessageType && msg.Content == userCancelMessage {
				hasCancelErrorMessage = true
				break
			}
		}

		// Check if we have a tool message with error
		hasToolError := false
		for _, msg := range messages {
			if msg.Type == ToolUseMessageType &&
				msg.ToolError && strings.Contains(msg.ToolResult, "user canceled") {
				hasToolError = true
				break
			}
		}

		// We should have at least one of these messages
		if !(hasCancelErrorMessage || hasToolError) {
			t.Errorf("Should have either an error message or a tool with error about cancellation")
		}
	})
}

func TestInnerLoop_UserDoesNotCancel(t *testing.T) {
	mockConvo := NewMockConvo(t)

	agent := &Agent{
		convo:  mockConvo,
		inbox:  make(chan string, 100),
		outbox: make(chan AgentMessage, 100),
	}

	// Define test message
	// This simulates something that would result in claude
	// responding with tool_use responses.
	testMsg := "use test_tool for something"

	// Mock initial response with tool use
	initialResponse := &ant.MessageResponse{
		StopReason: ant.StopReasonToolUse,
		Content: []ant.Content{
			{
				Type:      ant.ContentTypeToolUse,
				ID:        "tool1",
				ToolName:  "test_tool",
				ToolInput: []byte(`{"param":"value"}`),
			},
		},
		Usage: ant.Usage{
			InputTokens:  100,
			OutputTokens: 200,
		},
	}

	// Set up expected behaviors
	mockConvo.ExpectCall("SendMessage", nil).Return(initialResponse, nil)

	toolUseContents := []ant.Content{
		{
			Type:       ant.ContentTypeToolResult,
			ToolUseID:  "tool1",
			Text:       "",
			ToolResult: "This is a tool result",
			ToolError:  false,
		},
	}
	toolUseResponse := &ant.MessageResponse{
		// StopReason: ant.StopReasonEndTurn,
		Content: []ant.Content{
			{
				Type: ant.ContentTypeText,
				Text: "tool_use contents accepted",
			},
		},
		Usage: ant.Usage{
			InputTokens:  50,
			OutputTokens: 75,
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setting up the mock response for tool results
	mockConvo.ExpectCall("ToolResultContents", initialResponse).Return(toolUseContents, nil)
	mockConvo.ExpectCall("SendMessage", nil).Return(toolUseResponse, nil)
	// mockConvo, as a mock, isn't able to run the loop in ant.Convo that makes this agent.OnToolResult callback.
	// So we "mock" it out here by calling it explicitly, in order to make sure it calls .pushToOutbox with this message.
	// This is not a good situation.
	// ant.Convo and loop.Agent seem to be excessively coupled, and aware of each others' internal details.
	// TODO: refactor (or clarify in docs somewhere) the boundary between what ant.Convo is responsible
	// for vs what loop.Agent is responsible for.
	antConvo := &ant.Convo{}
	res := ""
	agent.OnToolResult(ctx, antConvo, "tool1", nil, toolUseContents[0], &res, nil)

	// Send a message to the agent's inbox
	agent.UserMessage(ctx, testMsg)

	// Run one iteration of InnerLoop
	agent.InnerLoop(ctx)

	// Verify results
	mockConvo.AssertExpectations(t)

	// Get all messages from outbox and verify their types/content
	var messages []AgentMessage

	// Collect messages until outbox is empty or we have 10 messages
	for i := 0; i < 10; i++ {
		select {
		case msg := <-agent.outbox:
			messages = append(messages, msg)
		default:
			// No more messages
			i = 10 // Exit the loop
		}
	}

	// Print out the messages we got for debugging
	t.Logf("Received %d messages from outbox", len(messages))
	for i, msg := range messages {
		t.Logf("Message %d: Type=%s, Content=%s, EndOfTurn=%t", i, msg.Type, msg.Content, msg.EndOfTurn)
		if msg.ToolName != "" {
			t.Logf("  Tool: Name=%s, Input=%s, Result=%s, Error=%v",
				msg.ToolName, msg.ToolInput, msg.ToolResult, msg.ToolError)
		}
	}

	// Basic checks
	if len(messages) < 1 {
		t.Errorf("Should have at least one message, got %d", len(messages))
	}

	// The main thing we want to verify: when user cancels, the response processing stops
	// and appropriate messages are sent

	// Check if we have an error message about cancellation
	hasCancelErrorMessage := false
	for _, msg := range messages {
		if msg.Type == ErrorMessageType && msg.Content == userCancelMessage {
			hasCancelErrorMessage = true
			break
		}
	}

	// Check if we have a tool message with error
	hasToolError := false
	for _, msg := range messages {
		if msg.Type == ToolUseMessageType &&
			msg.ToolError && strings.Contains(msg.ToolResult, "user canceled") {
			hasToolError = true
			break
		}
	}

	if hasCancelErrorMessage || hasToolError {
		t.Errorf("Should not have either an error message nor a tool with error about cancellation")
	}
}
