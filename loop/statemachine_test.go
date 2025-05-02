package loop

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestStateMachine(t *testing.T) {
	ctx := context.Background()
	sm := NewStateMachine()

	// Check initial state
	if sm.CurrentState() != StateReady {
		t.Errorf("Initial state should be StateReady, got %s", sm.CurrentState())
	}

	// Test valid transition
	err := sm.Transition(ctx, StateWaitingForUserInput, "Starting inner loop")
	if err != nil {
		t.Errorf("Error transitioning to StateWaitingForUserInput: %v", err)
	}
	if sm.CurrentState() != StateWaitingForUserInput {
		t.Errorf("Current state should be StateWaitingForUserInput, got %s", sm.CurrentState())
	}
	if sm.PreviousState() != StateReady {
		t.Errorf("Previous state should be StateReady, got %s", sm.PreviousState())
	}

	// Test invalid transition
	err = sm.Transition(ctx, StateRunningAutoformatters, "Invalid transition")
	if err == nil {
		t.Error("Expected error for invalid transition but got nil")
	}

	// Verify state didn't change after invalid transition
	if sm.CurrentState() != StateWaitingForUserInput {
		t.Errorf("State should not have changed after invalid transition, got %s", sm.CurrentState())
	}

	// Test complete flow
	transitions := []struct {
		state State
		event string
	}{
		{StateSendingToLLM, "Sending user message to LLM"},
		{StateProcessingLLMResponse, "Processing LLM response"},
		{StateToolUseRequested, "LLM requested tool use"},
		{StateCheckingForCancellation, "Checking for user cancellation"},
		{StateRunningTool, "Running tool"},
		{StateCheckingGitCommits, "Checking for git commits"},
		{StateCheckingBudget, "Checking budget"},
		{StateGatheringAdditionalMessages, "Gathering additional messages"},
		{StateSendingToolResults, "Sending tool results"},
		{StateProcessingLLMResponse, "Processing LLM response"},
		{StateEndOfTurn, "End of turn"},
		{StateWaitingForUserInput, "Waiting for next user input"},
	}

	for i, tt := range transitions {
		err := sm.Transition(ctx, tt.state, tt.event)
		if err != nil {
			t.Errorf("[%d] Error transitioning to %s: %v", i, tt.state, err)
		}
		if sm.CurrentState() != tt.state {
			t.Errorf("[%d] Current state should be %s, got %s", i, tt.state, sm.CurrentState())
		}
	}

	// Check if history was recorded correctly
	history := sm.History()
	expectedHistoryLen := len(transitions) + 1 // +1 for the initial transition
	if len(history) != expectedHistoryLen {
		t.Errorf("Expected history length %d, got %d", expectedHistoryLen, len(history))
	}

	// Check error state detection
	err = sm.Transition(ctx, StateError, "An error occurred")
	if err != nil {
		t.Errorf("Error transitioning to StateError: %v", err)
	}
	if !sm.IsInErrorState() {
		t.Error("IsInErrorState() should return true when in StateError")
	}
	if !sm.IsInTerminalState() {
		t.Error("IsInTerminalState() should return true when in StateError")
	}

	// Test reset
	sm.Reset()
	if sm.CurrentState() != StateReady {
		t.Errorf("After reset, state should be StateReady, got %s", sm.CurrentState())
	}
}

func TestTimeInState(t *testing.T) {
	sm := NewStateMachine()

	// Ensure time in state increases
	time.Sleep(50 * time.Millisecond)
	timeInState := sm.TimeInState()
	if timeInState < 50*time.Millisecond {
		t.Errorf("Expected TimeInState() > 50ms, got %v", timeInState)
	}
}

func TestTransitionEvent(t *testing.T) {
	ctx := context.Background()
	sm := NewStateMachine()

	// Test transition with custom event
	event := TransitionEvent{
		Description: "Test event",
		Data:        map[string]string{"key": "value"},
		Timestamp:   time.Now(),
	}

	err := sm.TransitionWithEvent(ctx, StateWaitingForUserInput, event)
	if err != nil {
		t.Errorf("Error in TransitionWithEvent: %v", err)
	}

	// Check the event was recorded in history
	history := sm.History()
	if len(history) != 1 {
		t.Fatalf("Expected history length 1, got %d", len(history))
	}
	if history[0].Event.Description != "Test event" {
		t.Errorf("Expected event description 'Test event', got '%s'", history[0].Event.Description)
	}
}

func TestConcurrentTransitions(t *testing.T) {
	sm := NewStateMachine()
	ctx := context.Background()

	// Start with waiting for user input
	sm.Transition(ctx, StateWaitingForUserInput, "Initial state")

	// Set up a channel to receive transition events
	events := make(chan StateTransition, 100)
	removeListener := sm.AddTransitionListener(events)
	defer removeListener()

	// Launch goroutines to perform concurrent transitions
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(10)

	go func() {
		wg.Wait()
		close(done)
	}()

	// Launch 10 goroutines that attempt to transition the state machine
	for i := 0; i < 10; i++ {
		go func(idx int) {
			defer wg.Done()

			// Each goroutine tries to make a valid transition from the current state
			for j := 0; j < 10; j++ {
				currentState := sm.CurrentState()
				var nextState State

				// Choose a valid next state based on current state
				switch currentState {
				case StateWaitingForUserInput:
					nextState = StateSendingToLLM
				case StateSendingToLLM:
					nextState = StateProcessingLLMResponse
				case StateProcessingLLMResponse:
					nextState = StateToolUseRequested
				case StateToolUseRequested:
					nextState = StateCheckingForCancellation
				case StateCheckingForCancellation:
					nextState = StateRunningTool
				case StateRunningTool:
					nextState = StateCheckingGitCommits
				case StateCheckingGitCommits:
					nextState = StateCheckingBudget
				case StateCheckingBudget:
					nextState = StateGatheringAdditionalMessages
				case StateGatheringAdditionalMessages:
					nextState = StateSendingToolResults
				case StateSendingToolResults:
					nextState = StateProcessingLLMResponse
				default:
					// If in a state we don't know how to handle, reset to a known state
					sm.ForceTransition(ctx, StateWaitingForUserInput, "Reset for test")
					continue
				}

				// Try to transition and record success/failure
				err := sm.Transition(ctx, nextState, fmt.Sprintf("Transition from goroutine %d", idx))
				if err != nil {
					// This is expected in concurrent scenarios - another goroutine might have
					// changed the state between our check and transition attempt
					time.Sleep(5 * time.Millisecond) // Back off a bit
				}
			}
		}(i)
	}

	// Collect events until all goroutines are done
	transitions := make([]StateTransition, 0)
loop:
	for {
		select {
		case evt := <-events:
			transitions = append(transitions, evt)
		case <-done:
			// Collect any remaining events
			for len(events) > 0 {
				transitions = append(transitions, <-events)
			}
			break loop
		}
	}

	// Get final history from state machine
	history := sm.History()

	// We may have missed some events due to channel buffer size and race conditions
	// That's okay for this test - the main point is to verify thread safety
	t.Logf("Collected %d events, history contains %d transitions",
		len(transitions), len(history))

	// Verify that all transitions in history are valid
	for i := 1; i < len(history); i++ {
		prev := history[i-1]
		curr := history[i]

		// Skip validating transitions if they're forced
		if strings.HasPrefix(curr.Event.Description, "Forced transition") {
			continue
		}

		if prev.To != curr.From {
			t.Errorf("Invalid transition chain at index %d: %s->%s followed by %s->%s",
				i, prev.From, prev.To, curr.From, curr.To)
		}
	}
}

func TestForceTransition(t *testing.T) {
	sm := NewStateMachine()
	ctx := context.Background()

	// Set to a regular state
	sm.Transition(ctx, StateWaitingForUserInput, "Initial state")

	// Force transition to a state that would normally be invalid
	sm.ForceTransition(ctx, StateError, "Testing force transition")

	// Check that the transition happened despite being invalid
	if sm.CurrentState() != StateError {
		t.Errorf("Force transition failed, state is %s instead of %s",
			sm.CurrentState(), StateError)
	}

	// Check that it was recorded in history
	history := sm.History()
	lastTransition := history[len(history)-1]

	if lastTransition.From != StateWaitingForUserInput || lastTransition.To != StateError {
		t.Errorf("Force transition not properly recorded in history: %v", lastTransition)
	}
}

func TestTransitionListeners(t *testing.T) {
	sm := NewStateMachine()
	ctx := context.Background()

	// Create a channel to receive transitions
	events := make(chan StateTransition, 10)

	// Add a listener
	removeListener := sm.AddTransitionListener(events)

	// Make a transition
	sm.Transition(ctx, StateWaitingForUserInput, "Testing listeners")

	// Check that the event was received
	select {
	case evt := <-events:
		if evt.To != StateWaitingForUserInput {
			t.Errorf("Received wrong transition: %v", evt)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Timeout waiting for transition event")
	}

	// Remove the listener
	removeListener()

	// Make another transition
	sm.Transition(ctx, StateSendingToLLM, "After removing listener")

	// Verify no event was received
	select {
	case evt := <-events:
		t.Errorf("Received transition after removing listener: %v", evt)
	case <-time.After(100 * time.Millisecond):
		// This is expected
	}
}
