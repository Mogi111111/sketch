package loop

import (
	"context"
	"testing"
	"time"
)

// TestIteratorBasic tests basic iterator functionality
func TestIteratorBasic(t *testing.T) {
	// Create an agent with some predefined messages
	agent := &Agent{
		subscribers: []chan *AgentMessage{},
	}

	// Add some test messages to the history
	agent.mu.Lock()
	agent.history = []AgentMessage{
		{Type: AgentMessageType, Content: "Message 1", Idx: 0},
		{Type: AgentMessageType, Content: "Message 2", Idx: 1},
		{Type: AgentMessageType, Content: "Message 3", Idx: 2},
	}
	agent.mu.Unlock()

	// Create an iterator starting from the beginning
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	it := agent.NewIterator(ctx, 0)
	defer it.Close()

	// Read all messages and verify
	for i := range 3 {
		msg := it.Next()
		if msg == nil {
			t.Fatalf("Expected message %d but got nil", i)
		}
		expectedNum := i + 1
		expectedContent := "Message " + string(rune('0')+rune(expectedNum))
		if msg.Content != expectedContent {
			t.Errorf("Expected message %d to be 'Message %d', got '%s'", i+1, i+1, msg.Content)
		}
	}
}

// TestIteratorStartFromMiddle tests starting an iterator from a specific index
func TestIteratorStartFromMiddle(t *testing.T) {
	// Create an agent with some predefined messages
	agent := &Agent{
		subscribers: []chan *AgentMessage{},
	}

	// Add some test messages to the history
	agent.mu.Lock()
	agent.history = []AgentMessage{
		{Type: AgentMessageType, Content: "Message 1", Idx: 0},
		{Type: AgentMessageType, Content: "Message 2", Idx: 1},
		{Type: AgentMessageType, Content: "Message 3", Idx: 2},
	}
	agent.mu.Unlock()

	// Create an iterator starting from index 1
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	it := agent.NewIterator(ctx, 1)
	defer it.Close()

	// We should get Message 2 and 3
	msg := it.Next()
	if msg == nil || msg.Content != "Message 2" {
		t.Errorf("Expected 'Message 2', got %v", msg)
	}

	msg = it.Next()
	if msg == nil || msg.Content != "Message 3" {
		t.Errorf("Expected 'Message 3', got %v", msg)
	}
}

// TestIteratorWithNewMessages tests that the iterator properly waits for and receives new messages
func TestIteratorWithNewMessages(t *testing.T) {
	// Create an agent
	agent := &Agent{
		subscribers: []chan *AgentMessage{},
	}

	// Create an iterator
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	it := agent.NewIterator(ctx, 0)
	defer it.Close()

	// Use channels to synchronize instead of sleeps
	msg1Added := make(chan struct{})
	msg2Ready := make(chan struct{})

	// Add messages in another goroutine
	go func() {
		// Add first message immediately
		agent.pushToOutbox(context.Background(), AgentMessage{Type: AgentMessageType, Content: "New message 1"})

		// Signal that message 1 is added
		close(msg1Added)

		// Wait for signal that we're ready for message 2
		<-msg2Ready

		// Add second message
		agent.pushToOutbox(context.Background(), AgentMessage{Type: AgentMessageType, Content: "New message 2"})
	}()

	// Read first message
	msg := it.Next()
	if msg == nil || msg.Content != "New message 1" {
		t.Errorf("Expected 'New message 1', got %v", msg)
	}

	// Signal that we're ready for message 2
	close(msg2Ready)

	// Read second message
	msg = it.Next()
	if msg == nil || msg.Content != "New message 2" {
		t.Errorf("Expected 'New message 2', got %v", msg)
	}
}

// TestIteratorClose tests that closing an iterator removes it from the subscribers list
func TestIteratorClose(t *testing.T) {
	// Create an agent
	agent := &Agent{
		subscribers: []chan *AgentMessage{},
	}

	// Create an iterator
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	it := agent.NewIterator(ctx, 0)

	// Verify iterator was added to subscribers after it tries to get a message
	// that doesn't exist yet
	agent.mu.Lock()
	agent.history = []AgentMessage{} // ensure history is empty
	agent.mu.Unlock()

	// Start a goroutine to call Next() which should subscribe
	done := make(chan struct{})
	go func() {
		// This will block after subscribing
		it.Next()
		close(done)
	}()

	// Give a short time for the goroutine to run and subscribe
	time.Sleep(10 * time.Millisecond)

	// Check that we have a subscriber
	agent.mu.Lock()
	subscriberCount := len(agent.subscribers)
	agent.mu.Unlock()

	if subscriberCount != 1 {
		t.Errorf("Expected 1 subscriber, got %d", subscriberCount)
	}

	// Close the iterator
	it.Close()

	// Add a message to trigger the goroutine to exit (in case it's still waiting)
	agent.pushToOutbox(context.Background(), AgentMessage{Type: AgentMessageType, Content: "Test message"})

	// Wait for the goroutine to finish
	select {
	case <-done:
		// Good, it finished
	case <-time.After(100 * time.Millisecond):
		t.Error("Timed out waiting for iterator goroutine to finish")
	}

	// Verify the subscriber was removed
	agent.mu.Lock()
	subscriberCount = len(agent.subscribers)
	agent.mu.Unlock()

	if subscriberCount != 0 {
		t.Errorf("Expected 0 subscribers after Close(), got %d", subscriberCount)
	}
}

// TestIteratorContextCancel tests that an iterator stops properly when its context is cancelled
func TestIteratorContextCancel(t *testing.T) {
	// Create an agent
	agent := &Agent{
		subscribers: []chan *AgentMessage{},
	}

	// Create a context that we can cancel
	ctx, cancel := context.WithCancel(context.Background())
	it := agent.NewIterator(ctx, 0)
	defer it.Close()

	// Start a goroutine to call Next() which will block
	resultChan := make(chan *AgentMessage)
	go func() {
		resultChan <- it.Next()
	}()

	// Wait a minimal time, then cancel the context
	time.Sleep(10 * time.Millisecond)
	cancel()

	// Verify we get nil from the iterator
	select {
	case result := <-resultChan:
		if result != nil {
			t.Errorf("Expected nil result after context cancel, got %v", result)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Timed out waiting for iterator to return after context cancel")
	}

	// Verify the subscriber was removed due to context cancellation
	agent.mu.Lock()
	subscriberCount := len(agent.subscribers)
	agent.mu.Unlock()

	if subscriberCount != 0 {
		t.Errorf("Expected 0 subscribers after context cancel, got %d", subscriberCount)
	}
}
