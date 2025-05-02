package ant

import (
	"cmp"
	"context"
	"math"
	"net/http"
	"os"
	"strings"
	"testing"

	"sketch.dev/httprr"
)

func TestBasicConvo(t *testing.T) {
	ctx := context.Background()
	rr, err := httprr.Open("testdata/basic_convo.httprr", http.DefaultTransport)
	if err != nil {
		t.Fatal(err)
	}
	rr.ScrubReq(func(req *http.Request) error {
		req.Header.Del("x-api-key")
		return nil
	})

	apiKey := cmp.Or(os.Getenv("OUTER_SKETCH_ANTHROPIC_API_KEY"), os.Getenv("ANTHROPIC_API_KEY"))
	convo := NewConvo(ctx, apiKey)
	convo.HTTPC = rr.Client()

	const name = "Cornelius"
	res, err := convo.SendUserTextMessage("Hi, my name is " + name)
	if err != nil {
		t.Fatal(err)
	}
	for _, part := range res.Content {
		t.Logf("%s", part.Text)
	}
	res, err = convo.SendUserTextMessage("What is my name?")
	if err != nil {
		t.Fatal(err)
	}
	got := ""
	for _, part := range res.Content {
		got += part.Text
	}
	if !strings.Contains(got, name) {
		t.Errorf("model does not know the given name %s: %q", name, got)
	}
}

// TestCalculateCostFromTokens tests the calculateCostFromTokens function
func TestCalculateCostFromTokens(t *testing.T) {
	tests := []struct {
		name                     string
		model                    string
		inputTokens              uint64
		outputTokens             uint64
		cacheReadInputTokens     uint64
		cacheCreationInputTokens uint64
		want                     float64
	}{
		{
			name:                     "Zero tokens",
			model:                    Claude37Sonnet,
			inputTokens:              0,
			outputTokens:             0,
			cacheReadInputTokens:     0,
			cacheCreationInputTokens: 0,
			want:                     0,
		},
		{
			name:                     "1000 input tokens, 500 output tokens",
			model:                    Claude37Sonnet,
			inputTokens:              1000,
			outputTokens:             500,
			cacheReadInputTokens:     0,
			cacheCreationInputTokens: 0,
			want:                     0.0105,
		},
		{
			name:                     "10000 input tokens, 5000 output tokens",
			model:                    Claude37Sonnet,
			inputTokens:              10000,
			outputTokens:             5000,
			cacheReadInputTokens:     0,
			cacheCreationInputTokens: 0,
			want:                     0.105,
		},
		{
			name:                     "With cache read tokens",
			model:                    Claude37Sonnet,
			inputTokens:              1000,
			outputTokens:             500,
			cacheReadInputTokens:     2000,
			cacheCreationInputTokens: 0,
			want:                     0.0111,
		},
		{
			name:                     "With cache creation tokens",
			model:                    Claude37Sonnet,
			inputTokens:              1000,
			outputTokens:             500,
			cacheReadInputTokens:     0,
			cacheCreationInputTokens: 1500,
			want:                     0.016125,
		},
		{
			name:                     "With all token types",
			model:                    Claude37Sonnet,
			inputTokens:              1000,
			outputTokens:             500,
			cacheReadInputTokens:     2000,
			cacheCreationInputTokens: 1500,
			want:                     0.016725,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := Usage{
				InputTokens:              tt.inputTokens,
				OutputTokens:             tt.outputTokens,
				CacheReadInputTokens:     tt.cacheReadInputTokens,
				CacheCreationInputTokens: tt.cacheCreationInputTokens,
			}
			mr := MessageResponse{
				Model: tt.model,
				Usage: usage,
			}
			totalCost := mr.TotalDollars()
			if math.Abs(totalCost-tt.want) > 0.0001 {
				t.Errorf("totalCost = %v, want %v", totalCost, tt.want)
			}
		})
	}
}

// TestCancelToolUse tests the CancelToolUse function of the Convo struct
func TestCancelToolUse(t *testing.T) {
	tests := []struct {
		name         string
		setupToolUse bool
		toolUseID    string
		cancelErr    error
		expectError  bool
		expectCancel bool
	}{
		{
			name:         "Cancel existing tool use",
			setupToolUse: true,
			toolUseID:    "tool123",
			cancelErr:    nil,
			expectError:  false,
			expectCancel: true,
		},
		{
			name:         "Cancel existing tool use with error",
			setupToolUse: true,
			toolUseID:    "tool456",
			cancelErr:    context.Canceled,
			expectError:  false,
			expectCancel: true,
		},
		{
			name:         "Cancel non-existent tool use",
			setupToolUse: false,
			toolUseID:    "tool789",
			cancelErr:    nil,
			expectError:  true,
			expectCancel: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			convo := NewConvo(context.Background(), "")

			var cancelCalled bool
			var cancelledWithErr error

			if tt.setupToolUse {
				// Setup a mock cancel function to track calls
				mockCancel := func(err error) {
					cancelCalled = true
					cancelledWithErr = err
				}

				convo.muToolUseCancel.Lock()
				convo.toolUseCancel[tt.toolUseID] = mockCancel
				convo.muToolUseCancel.Unlock()
			}

			err := convo.CancelToolUse(tt.toolUseID, tt.cancelErr)

			// Check if we got the expected error state
			if (err != nil) != tt.expectError {
				t.Errorf("CancelToolUse() error = %v, expectError %v", err, tt.expectError)
			}

			// Check if the cancel function was called as expected
			if cancelCalled != tt.expectCancel {
				t.Errorf("Cancel function called = %v, expectCancel %v", cancelCalled, tt.expectCancel)
			}

			// If we expected the cancel to be called, verify it was called with the right error
			if tt.expectCancel && cancelledWithErr != tt.cancelErr {
				t.Errorf("Cancel function called with error = %v, expected %v", cancelledWithErr, tt.cancelErr)
			}

			// Verify the toolUseID was removed from the map if it was initially added
			if tt.setupToolUse {
				convo.muToolUseCancel.Lock()
				_, exists := convo.toolUseCancel[tt.toolUseID]
				convo.muToolUseCancel.Unlock()

				if exists {
					t.Errorf("toolUseID %s still exists in the map after cancellation", tt.toolUseID)
				}
			}
		})
	}
}
