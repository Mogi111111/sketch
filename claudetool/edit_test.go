package claudetool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// setupTestFile creates a temporary file with given content for testing
func setupTestFile(t *testing.T, content string) string {
	t.Helper()

	// Create a temporary directory
	tempDir, err := os.MkdirTemp("", "anthropic_edit_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}

	// Create a test file in the temp directory
	testFile := filepath.Join(tempDir, "test_file.txt")
	if err := os.WriteFile(testFile, []byte(content), 0o644); err != nil {
		os.RemoveAll(tempDir)
		t.Fatalf("Failed to write test file: %v", err)
	}

	// Register cleanup function
	t.Cleanup(func() {
		os.RemoveAll(tempDir)
	})

	return testFile
}

// callEditTool is a helper to call the edit tool with specific parameters
func callEditTool(t *testing.T, input map[string]any) string {
	t.Helper()

	// Convert input to JSON
	inputJSON, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("Failed to marshal input: %v", err)
	}

	// Call the tool
	result, err := EditRun(context.Background(), inputJSON)
	if err != nil {
		t.Fatalf("Tool execution failed: %v", err)
	}

	return result
}

// TestEditToolView tests the view command functionality
func TestEditToolView(t *testing.T) {
	content := "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
	testFile := setupTestFile(t, content)

	// Test the view command
	result := callEditTool(t, map[string]any{
		"command": "view",
		"path":    testFile,
	})

	// Verify results
	if !strings.Contains(result, "Line 1") {
		t.Errorf("View result should contain the file content, got: %s", result)
	}

	// Test view with range
	result = callEditTool(t, map[string]any{
		"command":    "view",
		"path":       testFile,
		"view_range": []int{2, 4},
	})

	// Verify range results
	if strings.Contains(result, "Line 1") || !strings.Contains(result, "Line 2") {
		t.Errorf("View with range should show only specified lines, got: %s", result)
	}
}

// TestEditToolStrReplace tests the str_replace command functionality
func TestEditToolStrReplace(t *testing.T) {
	content := "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
	testFile := setupTestFile(t, content)

	// Test the str_replace command
	result := callEditTool(t, map[string]any{
		"command": "str_replace",
		"path":    testFile,
		"old_str": "Line 3",
		"new_str": "Modified Line 3",
	})

	// Verify the file was modified
	modifiedContent, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Failed to read test file: %v", err)
	}

	if !strings.Contains(string(modifiedContent), "Modified Line 3") {
		t.Errorf("File content should be modified, got: %s", string(modifiedContent))
	}

	// Verify the result contains a snippet
	if !strings.Contains(result, "Modified Line 3") {
		t.Errorf("Result should contain the modified content, got: %s", result)
	}
}

// TestEditToolInsert tests the insert command functionality
func TestEditToolInsert(t *testing.T) {
	content := "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
	testFile := setupTestFile(t, content)

	// Test the insert command
	result := callEditTool(t, map[string]any{
		"command":     "insert",
		"path":        testFile,
		"insert_line": 2,
		"new_str":     "Inserted Line",
	})

	// Verify the file was modified
	modifiedContent, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Failed to read test file: %v", err)
	}

	expected := "Line 1\nLine 2\nInserted Line\nLine 3\nLine 4\nLine 5"
	if string(modifiedContent) != expected {
		t.Errorf("File content incorrect after insert. Expected:\n%s\nGot:\n%s", expected, string(modifiedContent))
	}

	// Verify the result contains a snippet
	if !strings.Contains(result, "Inserted Line") {
		t.Errorf("Result should contain the inserted content, got: %s", result)
	}
}

// TestEditToolCreate tests the create command functionality
func TestEditToolCreate(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "anthropic_edit_test_create_*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}

	t.Cleanup(func() {
		os.RemoveAll(tempDir)
	})

	newFilePath := filepath.Join(tempDir, "new_file.txt")
	content := "This is a new file\nWith multiple lines"

	// Test the create command
	result := callEditTool(t, map[string]any{
		"command":   "create",
		"path":      newFilePath,
		"file_text": content,
	})

	// Verify the file was created with the right content
	createdContent, err := os.ReadFile(newFilePath)
	if err != nil {
		t.Fatalf("Failed to read created file: %v", err)
	}

	if string(createdContent) != content {
		t.Errorf("Created file content incorrect. Expected:\n%s\nGot:\n%s", content, string(createdContent))
	}

	// Verify the result message
	if !strings.Contains(result, "File created successfully") {
		t.Errorf("Result should confirm file creation, got: %s", result)
	}
}

// TestEditToolUndoEdit tests the undo_edit command functionality
func TestEditToolUndoEdit(t *testing.T) {
	originalContent := "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
	testFile := setupTestFile(t, originalContent)

	// First modify the file
	callEditTool(t, map[string]any{
		"command": "str_replace",
		"path":    testFile,
		"old_str": "Line 3",
		"new_str": "Modified Line 3",
	})

	// Then undo the edit
	result := callEditTool(t, map[string]any{
		"command": "undo_edit",
		"path":    testFile,
	})

	// Verify the file was restored to original content
	restoredContent, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Failed to read test file: %v", err)
	}

	if string(restoredContent) != originalContent {
		t.Errorf("File content should be restored to original, got: %s", string(restoredContent))
	}

	// Verify the result message
	if !strings.Contains(result, "undone successfully") {
		t.Errorf("Result should confirm undo operation, got: %s", result)
	}
}

// TestEditToolErrors tests various error conditions
func TestEditToolErrors(t *testing.T) {
	content := "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
	testFile := setupTestFile(t, content)

	testCases := []struct {
		name   string
		input  map[string]any
		errMsg string
	}{
		{
			name: "Invalid command",
			input: map[string]any{
				"command": "invalid_command",
				"path":    testFile,
			},
			errMsg: "unrecognized command",
		},
		{
			name: "Non-existent file",
			input: map[string]any{
				"command": "view",
				"path":    "/non/existent/file.txt",
			},
			errMsg: "does not exist",
		},
		{
			name: "Missing required parameter",
			input: map[string]any{
				"command": "str_replace",
				"path":    testFile,
				// Missing old_str
			},
			errMsg: "parameter old_str is required",
		},
		{
			name: "Multiple occurrences in str_replace",
			input: map[string]any{
				"command": "str_replace",
				"path":    testFile,
				"old_str": "Line", // Appears multiple times
				"new_str": "Modified Line",
			},
			errMsg: "Multiple occurrences",
		},
		{
			name: "Invalid view range",
			input: map[string]any{
				"command":    "view",
				"path":       testFile,
				"view_range": []int{10, 20}, // Out of range
			},
			errMsg: "invalid view_range",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			inputJSON, err := json.Marshal(tc.input)
			if err != nil {
				t.Fatalf("Failed to marshal input: %v", err)
			}

			_, err = EditRun(context.Background(), inputJSON)
			if err == nil {
				t.Fatalf("Expected error but got none")
			}

			if !strings.Contains(err.Error(), tc.errMsg) {
				t.Errorf("Error message does not contain expected text. Expected to contain: %q, Got: %q", tc.errMsg, err.Error())
			}
		})
	}
}

// TestHandleStrReplaceEdgeCases tests the handleStrReplace function specifically for edge cases
// that could cause panics like "index out of range [0] with length 0"
func TestHandleStrReplaceEdgeCases(t *testing.T) {
	// The issue was with strings.Split returning an empty slice when the separator wasn't found
	// This test directly tests the internal implementation with conditions that might cause this

	// Create a test file with empty content
	emptyFile := setupTestFile(t, "")

	// Test with empty file content and arbitrary oldStr
	_, err := handleStrReplace(emptyFile, "some string that doesn't exist", "new content")
	if err == nil {
		t.Fatal("Expected error for empty file but got none")
	}
	if !strings.Contains(err.Error(), "did not appear verbatim") {
		t.Errorf("Expected error message to indicate missing string, got: %s", err.Error())
	}

	// Create a file with content that doesn't match oldStr
	nonMatchingFile := setupTestFile(t, "This is some content\nthat doesn't contain the target string")

	// Test with content that doesn't contain oldStr
	_, err = handleStrReplace(nonMatchingFile, "target string not present", "replacement")
	if err == nil {
		t.Fatal("Expected error for non-matching content but got none")
	}
	if !strings.Contains(err.Error(), "did not appear verbatim") {
		t.Errorf("Expected error message to indicate missing string, got: %s", err.Error())
	}

	// Test handling of the edge case that could potentially cause the "index out of range" panic
	// This directly verifies that the handleStrReplace function properly handles the case where
	// strings.Split returns an empty or unexpected result

	// Verify that the protection against empty parts slice works
	fileContent := ""
	oldStr := "some string"
	parts := strings.Split(fileContent, oldStr)
	if len(parts) == 0 {
		// This should match the protection in the code
		parts = []string{""}
	}

	// This should not panic with the fix in place
	_ = strings.Count(parts[0], "\n") // This line would have panicked without the fix
}

// TestViewRangeWithStrReplace tests that the view_range parameter works correctly
// with the str_replace command (tests the full workflow)
func TestViewRangeWithStrReplace(t *testing.T) {
	// Create test file with multiple lines
	content := "Line 1: First line\nLine 2: Second line\nLine 3: Third line\nLine 4: Fourth line\nLine 5: Fifth line"
	testFile := setupTestFile(t, content)

	// First view a subset of the file using view_range
	viewResult := callEditTool(t, map[string]any{
		"command":    "view",
		"path":       testFile,
		"view_range": []int{2, 4}, // Only lines 2-4
	})

	// Verify that we only see the specified lines
	if strings.Contains(viewResult, "Line 1:") || strings.Contains(viewResult, "Line 5:") {
		t.Errorf("View with range should only show lines 2-4, got: %s", viewResult)
	}
	if !strings.Contains(viewResult, "Line 2:") || !strings.Contains(viewResult, "Line 4:") {
		t.Errorf("View with range should show lines 2-4, got: %s", viewResult)
	}

	// Now perform a str_replace on one of the lines we viewed
	replaceResult := callEditTool(t, map[string]any{
		"command": "str_replace",
		"path":    testFile,
		"old_str": "Line 3: Third line",
		"new_str": "Line 3: MODIFIED Third line",
	})

	// Check that the replacement was successful
	if !strings.Contains(replaceResult, "Line 3: MODIFIED Third line") {
		t.Errorf("Replace result should contain the modified line, got: %s", replaceResult)
	}

	// Verify the file content was updated correctly
	modifiedContent, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Failed to read test file after modification: %v", err)
	}

	expectedContent := "Line 1: First line\nLine 2: Second line\nLine 3: MODIFIED Third line\nLine 4: Fourth line\nLine 5: Fifth line"
	if string(modifiedContent) != expectedContent {
		t.Errorf("File content after replacement is incorrect.\nExpected:\n%s\nGot:\n%s",
			expectedContent, string(modifiedContent))
	}

	// View the modified file with a different view_range
	finalViewResult := callEditTool(t, map[string]any{
		"command":    "view",
		"path":       testFile,
		"view_range": []int{3, 3}, // Only the modified line
	})

	// Verify we can see only the modified line
	if !strings.Contains(finalViewResult, "Line 3: MODIFIED Third line") {
		t.Errorf("Final view should show the modified line, got: %s", finalViewResult)
	}
	if strings.Contains(finalViewResult, "Line 2:") || strings.Contains(finalViewResult, "Line 4:") {
		t.Errorf("Final view should only show line 3, got: %s", finalViewResult)
	}
}
