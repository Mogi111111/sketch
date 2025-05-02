package loop

import (
	"context"
	"reflect"
	"sync"
	"testing"

	"sketch.dev/ant"
)

// MockConvo is a custom mock for ant.Convo interface
type MockConvo struct {
	mu sync.Mutex
	t  *testing.T

	// Maps method name to a list of calls with arguments and return values
	calls map[string][]*mockCall
	// Maps method name to expected calls
	expectations map[string][]*mockExpectation
}

type mockCall struct {
	args   []interface{}
	result []interface{}
}

type mockExpectation struct {
	until  chan any
	args   []interface{}
	result []interface{}
}

// Return sets up return values for an expectation
func (e *mockExpectation) Return(values ...interface{}) {
	e.result = values
}

// Return sets up return values for an expectation
func (e *mockExpectation) BlockAndReturn(until chan any, values ...interface{}) {
	e.until = until
	e.result = values
}

// NewMockConvo creates a new mock Convo
func NewMockConvo(t *testing.T) *MockConvo {
	return &MockConvo{
		t:            t,
		mu:           sync.Mutex{},
		calls:        make(map[string][]*mockCall),
		expectations: make(map[string][]*mockExpectation),
	}
}

// ExpectCall sets up an expectation for a method call
func (m *MockConvo) ExpectCall(method string, args ...interface{}) *mockExpectation {
	m.mu.Lock()
	defer m.mu.Unlock()
	expectation := &mockExpectation{args: args}
	if _, ok := m.expectations[method]; !ok {
		m.expectations[method] = []*mockExpectation{}
	}
	m.expectations[method] = append(m.expectations[method], expectation)
	return expectation
}

// findMatchingExpectation finds a matching expectation for a method call
func (m *MockConvo) findMatchingExpectation(method string, args ...interface{}) (*mockExpectation, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	expectations, ok := m.expectations[method]
	if !ok {
		return nil, false
	}

	for i, exp := range expectations {
		if matchArgs(exp.args, args) {
			if exp.until != nil {
				<-exp.until
			}
			// Remove the matched expectation
			m.expectations[method] = append(expectations[:i], expectations[i+1:]...)
			return exp, true
		}
	}
	return nil, false
}

// matchArgs checks if call arguments match expectation arguments
func matchArgs(expected, actual []interface{}) bool {
	if len(expected) != len(actual) {
		return false
	}

	for i, exp := range expected {
		// Special case: nil matches anything
		if exp == nil {
			continue
		}

		// Check for equality
		if !reflect.DeepEqual(exp, actual[i]) {
			return false
		}
	}
	return true
}

// recordCall records a method call
func (m *MockConvo) recordCall(method string, args ...interface{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.calls[method]; !ok {
		m.calls[method] = []*mockCall{}
	}
	m.calls[method] = append(m.calls[method], &mockCall{args: args})
}

func (m *MockConvo) SendMessage(message ant.Message) (*ant.MessageResponse, error) {
	m.recordCall("SendMessage", message)
	exp, ok := m.findMatchingExpectation("SendMessage", message)
	if !ok {
		m.t.Errorf("unexpected call to SendMessage: %+v", message)
		m.t.FailNow()
	}
	var retErr error
	m.mu.Lock()
	defer m.mu.Unlock()
	if err, ok := exp.result[1].(error); ok {
		retErr = err
	}
	return exp.result[0].(*ant.MessageResponse), retErr
}

func (m *MockConvo) SendUserTextMessage(message string, otherContents ...ant.Content) (*ant.MessageResponse, error) {
	m.recordCall("SendUserTextMessage", message, otherContents)
	exp, ok := m.findMatchingExpectation("SendUserTextMessage", message, otherContents)
	if !ok {
		m.t.Error("unexpected call to SendUserTextMessage")
		m.t.FailNow()
	}
	var retErr error
	m.mu.Lock()
	defer m.mu.Unlock()
	if err, ok := exp.result[1].(error); ok {
		retErr = err
	}
	return exp.result[0].(*ant.MessageResponse), retErr
}

func (m *MockConvo) ToolResultContents(ctx context.Context, resp *ant.MessageResponse) ([]ant.Content, error) {
	m.recordCall("ToolResultContents", resp)
	exp, ok := m.findMatchingExpectation("ToolResultContents", resp)
	if !ok {
		m.t.Error("unexpected call to ToolResultContents")
		m.t.FailNow()
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	var retErr error
	if err, ok := exp.result[1].(error); ok {
		retErr = err
	}

	return exp.result[0].([]ant.Content), retErr
}

func (m *MockConvo) ToolResultCancelContents(resp *ant.MessageResponse) ([]ant.Content, error) {
	m.recordCall("ToolResultCancelContents", resp)
	exp, ok := m.findMatchingExpectation("ToolResultCancelContents", resp)
	if !ok {
		m.t.Error("unexpected call to ToolResultCancelContents")
		m.t.FailNow()
	}
	var retErr error
	m.mu.Lock()
	defer m.mu.Unlock()
	if err, ok := exp.result[1].(error); ok {
		retErr = err
	}

	return exp.result[0].([]ant.Content), retErr
}

func (m *MockConvo) CumulativeUsage() ant.CumulativeUsage {
	m.recordCall("CumulativeUsage")
	return ant.CumulativeUsage{}
}

func (m *MockConvo) OverBudget() error {
	m.recordCall("OverBudget")
	return nil
}

func (m *MockConvo) GetID() string {
	m.recordCall("GetID")
	return "mock-conversation-id"
}

func (m *MockConvo) SubConvoWithHistory() *ant.Convo {
	m.recordCall("SubConvoWithHistory")
	return nil
}

func (m *MockConvo) ResetBudget(_ ant.Budget) {
	m.recordCall("ResetBudget")
}

// AssertExpectations checks that all expectations were met
func (m *MockConvo) AssertExpectations(t *testing.T) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for method, expectations := range m.expectations {
		if len(expectations) > 0 {
			t.Errorf("not all expectations were met for method %s:", method)
		}
	}
}
