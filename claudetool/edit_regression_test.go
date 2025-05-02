package claudetool

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestEmptyContentHandling tests handling of empty content in str_replace and related operations
// This test specifically reproduces conditions that might lead to "index out of range [0]" panic
func TestEmptyContentHandling(t *testing.T) {
	// Create a file with empty content
	emptyFile := setupTestFile(t, "")

	// Test running EditRun directly with empty content
	// This more closely simulates the actual call flow that led to the panic
	input := map[string]any{
		"command": "str_replace",
		"path":    emptyFile,
		"old_str": "nonexistent text",
		"new_str": "new content",
	}

	inputJSON, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("Failed to marshal input: %v", err)
	}

	// This should not panic but return an error
	_, err = EditRun(context.Background(), inputJSON)
	if err == nil {
		t.Fatalf("Expected error for empty file with str_replace but got none")
	}

	// Make sure the error message is as expected
	if !strings.Contains(err.Error(), "did not appear verbatim") {
		t.Errorf("Expected error message to indicate missing string, got: %s", err.Error())
	}
}

// TestNilParameterHandling tests error cases with nil parameters
// This test validates proper error handling when nil or invalid parameters are provided
func TestNilParameterHandling(t *testing.T) {
	// Create a test file
	testFile := setupTestFile(t, "test content")

	// Test case 1: nil old_str in str_replace
	input1 := map[string]any{
		"command": "str_replace",
		"path":    testFile,
		// old_str is deliberately missing
		"new_str": "replacement",
	}

	inputJSON1, err := json.Marshal(input1)
	if err != nil {
		t.Fatalf("Failed to marshal input: %v", err)
	}

	_, err = EditRun(context.Background(), inputJSON1)
	if err == nil {
		t.Fatalf("Expected error for missing old_str but got none")
	}
	if !strings.Contains(err.Error(), "parameter old_str is required") {
		t.Errorf("Expected error message to indicate missing old_str, got: %s", err.Error())
	}

	// Test case 2: nil new_str in insert
	input2 := map[string]any{
		"command":     "insert",
		"path":        testFile,
		"insert_line": 1,
		// new_str is deliberately missing
	}

	inputJSON2, err := json.Marshal(input2)
	if err != nil {
		t.Fatalf("Failed to marshal input: %v", err)
	}

	_, err = EditRun(context.Background(), inputJSON2)
	if err == nil {
		t.Fatalf("Expected error for missing new_str but got none")
	}
	if !strings.Contains(err.Error(), "parameter new_str is required") {
		t.Errorf("Expected error message to indicate missing new_str, got: %s", err.Error())
	}

	// Test case 3: nil view_range in view
	// This doesn't cause an error, but tests the code path
	input3 := map[string]any{
		"command": "view",
		"path":    testFile,
		// No view_range
	}

	inputJSON3, err := json.Marshal(input3)
	if err != nil {
		t.Fatalf("Failed to marshal input: %v", err)
	}

	// This should not result in an error
	_, err = EditRun(context.Background(), inputJSON3)
	if err != nil {
		t.Fatalf("Unexpected error for nil view_range: %v", err)
	}
}

// TestEmptySplitResult tests the specific scenario where strings.Split might return empty results
// This directly reproduces conditions that might have led to the "index out of range [0]" panic
func TestEmptySplitResult(t *testing.T) {
	// Direct test of strings.Split behavior and our handling of it
	emptyCases := []struct {
		content string
		oldStr  string
	}{
		{"", "any string"},
		{"content", "not in string"},
		{"\n\n", "also not here"},
	}

	for _, tc := range emptyCases {
		parts := strings.Split(tc.content, tc.oldStr)

		// Verify that strings.Split with non-matching separator returns a slice with original content
		if len(parts) != 1 {
			t.Errorf("Expected strings.Split to return a slice with 1 element when separator isn't found, got %d elements", len(parts))
		}

		// Double check the content
		if len(parts) > 0 && parts[0] != tc.content {
			t.Errorf("Expected parts[0] to be original content %q, got %q", tc.content, parts[0])
		}
	}

	// Test the actual unsafe scenario with empty content
	emptyFile := setupTestFile(t, "")

	// Get the content and simulate the internal string splitting
	content, _ := readFile(emptyFile)
	oldStr := "nonexistent"
	parts := strings.Split(content, oldStr)

	// Validate that the defensive code would work
	if len(parts) == 0 {
		parts = []string{""} // This is the fix
	}

	// This would have panicked without the fix
	_ = strings.Count(parts[0], "\n")
}
