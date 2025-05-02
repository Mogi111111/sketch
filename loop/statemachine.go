package loop

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// State represents the possible states of the Agent state machine
type State int

const (
	// StateUnknown is the default state
	StateUnknown State = iota
	// StateReady is the initial state when the agent is initialized and ready to operate
	StateReady
	// StateWaitingForUserInput occurs when the agent is waiting for a user message to start a turn
	StateWaitingForUserInput
	// StateSendingToLLM occurs when the agent is sending message(s) to the LLM
	StateSendingToLLM
	// StateProcessingLLMResponse occurs when the agent is processing a response from the LLM
	StateProcessingLLMResponse
	// StateEndOfTurn occurs when processing is completed without tool use, and the turn ends
	StateEndOfTurn
	// StateToolUseRequested occurs when the LLM has requested to use a tool
	StateToolUseRequested
	// StateCheckingForCancellation occurs when the agent checks if user requested cancellation
	StateCheckingForCancellation
	// StateRunningTool occurs when the agent is executing the requested tool
	StateRunningTool
	// StateCheckingGitCommits occurs when the agent checks for new git commits after tool execution
	StateCheckingGitCommits
	// StateRunningAutoformatters occurs when the agent runs code formatters on new commits
	StateRunningAutoformatters
	// StateCheckingBudget occurs when the agent verifies if budget limits are exceeded
	StateCheckingBudget
	// StateGatheringAdditionalMessages occurs when the agent collects user messages that arrived during tool execution
	StateGatheringAdditionalMessages
	// StateSendingToolResults occurs when the agent sends tool results back to the LLM
	StateSendingToolResults
	// StateCancelled occurs when an operation was cancelled by the user
	StateCancelled
	// StateBudgetExceeded occurs when the budget limit was reached
	StateBudgetExceeded
	// StateError occurs when an error occurred during processing
	StateError
)

// String returns a string representation of the State for logging and debugging
func (s State) String() string {
	switch s {
	case StateUnknown:
		return "Unknown"
	case StateReady:
		return "Ready"
	case StateWaitingForUserInput:
		return "WaitingForUserInput"
	case StateSendingToLLM:
		return "SendingToLLM"
	case StateProcessingLLMResponse:
		return "ProcessingLLMResponse"
	case StateEndOfTurn:
		return "EndOfTurn"
	case StateToolUseRequested:
		return "ToolUseRequested"
	case StateCheckingForCancellation:
		return "CheckingForCancellation"
	case StateRunningTool:
		return "RunningTool"
	case StateCheckingGitCommits:
		return "CheckingGitCommits"
	case StateRunningAutoformatters:
		return "RunningAutoformatters"
	case StateCheckingBudget:
		return "CheckingBudget"
	case StateGatheringAdditionalMessages:
		return "GatheringAdditionalMessages"
	case StateSendingToolResults:
		return "SendingToolResults"
	case StateCancelled:
		return "Cancelled"
	case StateBudgetExceeded:
		return "BudgetExceeded"
	case StateError:
		return "Error"
	default:
		return fmt.Sprintf("Unknown(%d)", int(s))
	}
}

// TransitionEvent represents an event that causes a state transition
type TransitionEvent struct {
	// Description provides a human-readable description of the event
	Description string
	// Data can hold any additional information about the event
	Data interface{}
	// Timestamp is when the event occurred
	Timestamp time.Time
}

// StateTransition represents a transition from one state to another
type StateTransition struct {
	From  State
	To    State
	Event TransitionEvent
}

// StateMachine manages the Agent's states and transitions
type StateMachine struct {
	// mu protects all fields of the StateMachine from concurrent access
	mu sync.RWMutex
	// currentState is the current state of the state machine
	currentState State
	// previousState is the previous state of the state machine
	previousState State
	// stateEnteredAt is when the current state was entered
	stateEnteredAt time.Time
	// transitions maps from states to the states they can transition to
	transitions map[State]map[State]bool
	// history records the history of state transitions
	history []StateTransition
	// maxHistorySize limits the number of transitions to keep in history
	maxHistorySize int
	// eventListeners are notified when state transitions occur
	eventListeners []chan<- StateTransition
	// onTransition is a callback function that's called when a transition occurs
	onTransition func(ctx context.Context, from, to State, event TransitionEvent)
}

// NewStateMachine creates a new state machine initialized to StateReady
func NewStateMachine() *StateMachine {
	sm := &StateMachine{
		currentState:   StateReady,
		previousState:  StateUnknown,
		stateEnteredAt: time.Now(),
		transitions:    make(map[State]map[State]bool),
		maxHistorySize: 100,
		eventListeners: make([]chan<- StateTransition, 0),
	}

	// Initialize valid transitions
	sm.initTransitions()

	return sm
}

// SetMaxHistorySize sets the maximum number of transitions to keep in history
func (sm *StateMachine) SetMaxHistorySize(size int) {
	if size < 1 {
		size = 1 // Ensure we keep at least one entry
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.maxHistorySize = size

	// Trim history if needed
	if len(sm.history) > sm.maxHistorySize {
		sm.history = sm.history[len(sm.history)-sm.maxHistorySize:]
	}
}

// AddTransitionListener adds a listener channel that will be notified of state transitions
// Returns a function that can be called to remove the listener
func (sm *StateMachine) AddTransitionListener(listener chan<- StateTransition) func() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.eventListeners = append(sm.eventListeners, listener)

	// Return a function to remove this listener
	return func() {
		sm.RemoveTransitionListener(listener)
	}
}

// RemoveTransitionListener removes a previously added listener
func (sm *StateMachine) RemoveTransitionListener(listener chan<- StateTransition) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	for i, l := range sm.eventListeners {
		if l == listener {
			// Remove by swapping with the last element and then truncating
			lastIdx := len(sm.eventListeners) - 1
			sm.eventListeners[i] = sm.eventListeners[lastIdx]
			sm.eventListeners = sm.eventListeners[:lastIdx]
			break
		}
	}
}

// SetTransitionCallback sets a function to be called on every state transition
func (sm *StateMachine) SetTransitionCallback(callback func(ctx context.Context, from, to State, event TransitionEvent)) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.onTransition = callback
}

// ClearTransitionCallback removes any previously set transition callback
func (sm *StateMachine) ClearTransitionCallback() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.onTransition = nil
}

// initTransitions initializes the map of valid state transitions
func (sm *StateMachine) initTransitions() {
	// Helper function to add transitions
	addTransition := func(from State, to ...State) {
		// Initialize the map for this state if it doesn't exist
		if _, exists := sm.transitions[from]; !exists {
			sm.transitions[from] = make(map[State]bool)
		}

		// Add all of the 'to' states
		for _, toState := range to {
			sm.transitions[from][toState] = true
		}
	}

	// Define valid transitions based on the state machine diagram

	// Initial state
	addTransition(StateReady, StateWaitingForUserInput)

	// Main flow
	addTransition(StateWaitingForUserInput, StateSendingToLLM, StateError)
	addTransition(StateSendingToLLM, StateProcessingLLMResponse, StateError)
	addTransition(StateProcessingLLMResponse, StateEndOfTurn, StateToolUseRequested, StateError)
	addTransition(StateEndOfTurn, StateWaitingForUserInput)

	// Tool use flow
	addTransition(StateToolUseRequested, StateCheckingForCancellation)
	addTransition(StateCheckingForCancellation, StateRunningTool, StateCancelled)
	addTransition(StateRunningTool, StateCheckingGitCommits, StateError)
	addTransition(StateCheckingGitCommits, StateRunningAutoformatters, StateCheckingBudget)
	addTransition(StateRunningAutoformatters, StateCheckingBudget)
	addTransition(StateCheckingBudget, StateGatheringAdditionalMessages, StateBudgetExceeded)
	addTransition(StateGatheringAdditionalMessages, StateSendingToolResults, StateError)
	addTransition(StateSendingToolResults, StateProcessingLLMResponse, StateError)

	// Terminal states to new turn
	addTransition(StateCancelled, StateWaitingForUserInput)
	addTransition(StateBudgetExceeded, StateWaitingForUserInput)
	addTransition(StateError, StateWaitingForUserInput)
}

// Transition attempts to transition from the current state to the given state
func (sm *StateMachine) Transition(ctx context.Context, newState State, event string) error {
	if sm == nil {
		return fmt.Errorf("nil StateMachine pointer")
	}
	transitionEvent := TransitionEvent{
		Description: event,
		Timestamp:   time.Now(),
	}
	return sm.TransitionWithEvent(ctx, newState, transitionEvent)
}

// TransitionWithEvent attempts to transition from the current state to the given state
// with the provided event information
func (sm *StateMachine) TransitionWithEvent(ctx context.Context, newState State, event TransitionEvent) error {
	// First check if the transition is valid without holding the write lock
	sm.mu.RLock()
	currentState := sm.currentState
	canTransition := false
	if validToStates, exists := sm.transitions[currentState]; exists {
		canTransition = validToStates[newState]
	}
	sm.mu.RUnlock()

	if !canTransition {
		return fmt.Errorf("invalid transition from %s to %s", currentState, newState)
	}

	// Acquire write lock for the actual transition
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Double-check that the state hasn't changed since we checked
	if sm.currentState != currentState {
		// State changed between our check and lock acquisition
		// Re-check if the transition is still valid
		if validToStates, exists := sm.transitions[sm.currentState]; !exists || !validToStates[newState] {
			return fmt.Errorf("concurrent state change detected: invalid transition from current %s to %s",
				sm.currentState, newState)
		}
	}

	// Calculate duration in current state
	duration := time.Since(sm.stateEnteredAt)

	// Record the transition
	transition := StateTransition{
		From:  sm.currentState,
		To:    newState,
		Event: event,
	}

	// Update state
	sm.previousState = sm.currentState
	sm.currentState = newState
	sm.stateEnteredAt = time.Now()

	// Add to history
	sm.history = append(sm.history, transition)

	// Trim history if it exceeds maximum size
	if len(sm.history) > sm.maxHistorySize {
		sm.history = sm.history[len(sm.history)-sm.maxHistorySize:]
	}

	// Make a local copy of any callback functions to invoke outside the lock
	var onTransition func(ctx context.Context, from, to State, event TransitionEvent)
	var eventListenersCopy []chan<- StateTransition
	if sm.onTransition != nil {
		onTransition = sm.onTransition
	}
	if len(sm.eventListeners) > 0 {
		eventListenersCopy = make([]chan<- StateTransition, len(sm.eventListeners))
		copy(eventListenersCopy, sm.eventListeners)
	}

	// Log the transition
	slog.InfoContext(ctx, "State transition",
		"from", sm.previousState,
		"to", sm.currentState,
		"event", event.Description,
		"duration", duration)

	// Release the lock before notifying listeners to avoid deadlocks
	sm.mu.Unlock()

	// Notify listeners if any
	if onTransition != nil {
		onTransition(ctx, sm.previousState, sm.currentState, event)
	}

	for _, ch := range eventListenersCopy {
		select {
		case ch <- transition:
			// Successfully sent
		default:
			// Channel buffer full or no receiver, log and continue
			slog.WarnContext(ctx, "Failed to notify state transition listener",
				"from", sm.previousState, "to", sm.currentState)
		}
	}

	// Re-acquire the lock that we explicitly released above
	sm.mu.Lock()
	return nil
}

// CurrentState returns the current state
func (sm *StateMachine) CurrentState() State {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.currentState
}

// PreviousState returns the previous state
func (sm *StateMachine) PreviousState() State {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.previousState
}

// TimeInState returns how long the machine has been in the current state
func (sm *StateMachine) TimeInState() time.Duration {
	sm.mu.RLock()
	enteredAt := sm.stateEnteredAt
	sm.mu.RUnlock()
	return time.Since(enteredAt)
}

// CanTransition returns true if a transition from the from state to the to state is valid
func (sm *StateMachine) CanTransition(from, to State) bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	if validToStates, exists := sm.transitions[from]; exists {
		return validToStates[to]
	}
	return false
}

// History returns the transition history of the state machine
func (sm *StateMachine) History() []StateTransition {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Return a copy to prevent modification
	historyCopy := make([]StateTransition, len(sm.history))
	copy(historyCopy, sm.history)
	return historyCopy
}

// Reset resets the state machine to the initial ready state
func (sm *StateMachine) Reset() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.currentState = StateReady
	sm.previousState = StateUnknown
	sm.stateEnteredAt = time.Now()
}

// IsInTerminalState returns whether the current state is a terminal state
func (sm *StateMachine) IsInTerminalState() bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	switch sm.currentState {
	case StateEndOfTurn, StateCancelled, StateBudgetExceeded, StateError:
		return true
	default:
		return false
	}
}

// IsInErrorState returns whether the current state is an error state
func (sm *StateMachine) IsInErrorState() bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	switch sm.currentState {
	case StateError, StateCancelled, StateBudgetExceeded:
		return true
	default:
		return false
	}
}

// ForceTransition forces a transition regardless of whether it's valid according to the state machine rules
// This should be used only in critical situations like cancellation or error recovery
func (sm *StateMachine) ForceTransition(ctx context.Context, newState State, reason string) {
	event := TransitionEvent{
		Description: fmt.Sprintf("Forced transition: %s", reason),
		Timestamp:   time.Now(),
	}

	sm.mu.Lock()

	// Calculate duration in current state
	duration := time.Since(sm.stateEnteredAt)

	// Record the transition
	transition := StateTransition{
		From:  sm.currentState,
		To:    newState,
		Event: event,
	}

	// Update state
	sm.previousState = sm.currentState
	sm.currentState = newState
	sm.stateEnteredAt = time.Now()

	// Add to history
	sm.history = append(sm.history, transition)

	// Trim history if it exceeds maximum size
	if len(sm.history) > sm.maxHistorySize {
		sm.history = sm.history[len(sm.history)-sm.maxHistorySize:]
	}

	// Make a local copy of any callback functions to invoke outside the lock
	var onTransition func(ctx context.Context, from, to State, event TransitionEvent)
	var eventListenersCopy []chan<- StateTransition
	if sm.onTransition != nil {
		onTransition = sm.onTransition
	}
	if len(sm.eventListeners) > 0 {
		eventListenersCopy = make([]chan<- StateTransition, len(sm.eventListeners))
		copy(eventListenersCopy, sm.eventListeners)
	}

	// Log the transition
	slog.WarnContext(ctx, "Forced state transition",
		"from", sm.previousState,
		"to", sm.currentState,
		"reason", reason,
		"duration", duration)

	// Release the lock before notifying listeners to avoid deadlocks
	sm.mu.Unlock()

	// Notify listeners if any
	if onTransition != nil {
		onTransition(ctx, sm.previousState, sm.currentState, event)
	}

	for _, ch := range eventListenersCopy {
		select {
		case ch <- transition:
			// Successfully sent
		default:
			// Channel buffer full or no receiver, log and continue
			slog.WarnContext(ctx, "Failed to notify state transition listener for forced transition",
				"from", sm.previousState, "to", sm.currentState)
		}
	}

	// Re-acquire the lock
	sm.mu.Lock()
	defer sm.mu.Unlock()
}
