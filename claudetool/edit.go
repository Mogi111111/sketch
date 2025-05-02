package claudetool

/*

Note: sketch wrote this based on translating https://raw.githubusercontent.com/anthropics/anthropic-quickstarts/refs/heads/main/computer-use-demo/computer_use_demo/tools/edit.py

## Implementation Notes
This tool is based on Anthropic's Python implementation of the `text_editor_20250124` tool. It maintains a history of file edits to support the undo functionality, and verifies text uniqueness for the str_replace operation to ensure safe edits.

*/

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"sketch.dev/ant"
)

// Constants for the AnthropicEditTool
const (
	editName = "str_replace_editor"
)

// Constants used by the tool
const (
	snippetLines     = 4
	maxResponseLen   = 16000
	truncatedMessage = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
)

// Command represents the type of operation to perform
type editCommand string

const (
	viewCommand       editCommand = "view"
	createCommand     editCommand = "create"
	strReplaceCommand editCommand = "str_replace"
	insertCommand     editCommand = "insert"
	undoEditCommand   editCommand = "undo_edit"
)

// editInput represents the expected input format for the edit tool
type editInput struct {
	Command    string  `json:"command"`
	Path       string  `json:"path"`
	FileText   *string `json:"file_text,omitempty"`
	ViewRange  []int   `json:"view_range,omitempty"`
	OldStr     *string `json:"old_str,omitempty"`
	NewStr     *string `json:"new_str,omitempty"`
	InsertLine *int    `json:"insert_line,omitempty"`
}

// fileHistory maintains a history of edits for each file to support undo functionality
var fileHistory = make(map[string][]string)

// AnthropicEditTool is a tool for viewing, creating, and editing files
var AnthropicEditTool = &ant.Tool{
	// Note that Type is model-dependent, and would be different for Claude 3.5, for example.
	Type: "text_editor_20250124",
	Name: editName,
	Run:  EditRun,
}

// EditRun is the implementation of the edit tool
func EditRun(ctx context.Context, input json.RawMessage) (string, error) {
	var editRequest editInput
	if err := json.Unmarshal(input, &editRequest); err != nil {
		return "", fmt.Errorf("failed to parse edit input: %v", err)
	}

	// Validate the command
	cmd := editCommand(editRequest.Command)
	if !isValidCommand(cmd) {
		return "", fmt.Errorf("unrecognized command %s. The allowed commands are: view, create, str_replace, insert, undo_edit", cmd)
	}

	path := editRequest.Path

	// Validate the path
	if err := validatePath(cmd, path); err != nil {
		return "", err
	}

	// Execute the appropriate command
	switch cmd {
	case viewCommand:
		return handleView(ctx, path, editRequest.ViewRange)
	case createCommand:
		if editRequest.FileText == nil {
			return "", fmt.Errorf("parameter file_text is required for command: create")
		}
		return handleCreate(path, *editRequest.FileText)
	case strReplaceCommand:
		if editRequest.OldStr == nil {
			return "", fmt.Errorf("parameter old_str is required for command: str_replace")
		}
		newStr := ""
		if editRequest.NewStr != nil {
			newStr = *editRequest.NewStr
		}
		return handleStrReplace(path, *editRequest.OldStr, newStr)
	case insertCommand:
		if editRequest.InsertLine == nil {
			return "", fmt.Errorf("parameter insert_line is required for command: insert")
		}
		if editRequest.NewStr == nil {
			return "", fmt.Errorf("parameter new_str is required for command: insert")
		}
		return handleInsert(path, *editRequest.InsertLine, *editRequest.NewStr)
	case undoEditCommand:
		return handleUndoEdit(path)
	default:
		return "", fmt.Errorf("command %s is not implemented", cmd)
	}
}

// Utility function to check if a command is valid
func isValidCommand(cmd editCommand) bool {
	switch cmd {
	case viewCommand, createCommand, strReplaceCommand, insertCommand, undoEditCommand:
		return true
	default:
		return false
	}
}

// validatePath checks if the path/command combination is valid
func validatePath(cmd editCommand, path string) error {
	// Check if it's an absolute path
	if !filepath.IsAbs(path) {
		suggestedPath := "/" + path
		return fmt.Errorf("the path %s is not an absolute path, it should start with '/'. Maybe you meant %s?", path, suggestedPath)
	}

	// Get file info
	info, err := os.Stat(path)

	// Check if path exists (except for create command)
	if err != nil {
		if os.IsNotExist(err) && cmd != createCommand {
			return fmt.Errorf("the path %s does not exist. Please provide a valid path", path)
		} else if !os.IsNotExist(err) {
			return fmt.Errorf("error accessing path %s: %v", path, err)
		}
	} else {
		// Path exists, check if it's a directory
		if info.IsDir() && cmd != viewCommand {
			return fmt.Errorf("the path %s is a directory and only the 'view' command can be used on directories", path)
		}

		// For create command, check if file already exists
		if cmd == createCommand {
			return fmt.Errorf("file already exists at: %s. Cannot overwrite files using command 'create'", path)
		}
	}

	return nil
}

// handleView implements the view command
func handleView(ctx context.Context, path string, viewRange []int) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("error accessing path %s: %v", path, err)
	}

	// Handle directory view
	if info.IsDir() {
		if viewRange != nil {
			return "", fmt.Errorf("the view_range parameter is not allowed when path points to a directory")
		}

		// List files in the directory (up to 2 levels deep)
		return listDirectory(ctx, path)
	}

	// Handle file view
	fileContent, err := readFile(path)
	if err != nil {
		return "", err
	}

	initLine := 1
	if viewRange != nil {
		if len(viewRange) != 2 {
			return "", fmt.Errorf("invalid view_range. It should be a list of two integers")
		}

		fileLines := strings.Split(fileContent, "\n")
		nLinesFile := len(fileLines)
		initLine, finalLine := viewRange[0], viewRange[1]

		if initLine < 1 || initLine > nLinesFile {
			return "", fmt.Errorf("invalid view_range: %v. Its first element %d should be within the range of lines of the file: [1, %d]",
				viewRange, initLine, nLinesFile)
		}

		if finalLine != -1 && finalLine < initLine {
			return "", fmt.Errorf("invalid view_range: %v. Its second element %d should be larger or equal than its first %d",
				viewRange, finalLine, initLine)
		}

		if finalLine > nLinesFile {
			return "", fmt.Errorf("invalid view_range: %v. Its second element %d should be smaller than the number of lines in the file: %d",
				viewRange, finalLine, nLinesFile)
		}

		if finalLine == -1 {
			fileContent = strings.Join(fileLines[initLine-1:], "\n")
		} else {
			fileContent = strings.Join(fileLines[initLine-1:finalLine], "\n")
		}
	}

	return makeOutput(fileContent, path, initLine), nil
}

// handleCreate implements the create command
func handleCreate(path string, fileText string) (string, error) {
	// Ensure the directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("failed to create directory %s: %v", dir, err)
	}

	// Write the file
	if err := writeFile(path, fileText); err != nil {
		return "", err
	}

	// Save to history
	fileHistory[path] = append(fileHistory[path], fileText)

	return fmt.Sprintf("File created successfully at: %s", path), nil
}

// handleStrReplace implements the str_replace command
func handleStrReplace(path, oldStr, newStr string) (string, error) {
	// Read the file content
	fileContent, err := readFile(path)
	if err != nil {
		return "", err
	}

	// Replace tabs with spaces
	fileContent = maybeExpandTabs(path, fileContent)
	oldStr = maybeExpandTabs(path, oldStr)
	newStr = maybeExpandTabs(path, newStr)

	// Check if oldStr is unique in the file
	occurrences := strings.Count(fileContent, oldStr)
	if occurrences == 0 {
		return "", fmt.Errorf("no replacement was performed, old_str %q did not appear verbatim in %s", oldStr, path)
	} else if occurrences > 1 {
		// Find line numbers where oldStr appears
		fileContentLines := strings.Split(fileContent, "\n")
		var lines []int
		for idx, line := range fileContentLines {
			if strings.Contains(line, oldStr) {
				lines = append(lines, idx+1)
			}
		}
		return "", fmt.Errorf("no replacement was performed. Multiple occurrences of old_str %q in lines %v. Please ensure it is unique", oldStr, lines)
	}

	// Save the current content to history
	fileHistory[path] = append(fileHistory[path], fileContent)

	// Replace oldStr with newStr
	newFileContent := strings.Replace(fileContent, oldStr, newStr, 1)

	// Write the new content to the file
	if err := writeFile(path, newFileContent); err != nil {
		return "", err
	}

	// Create a snippet of the edited section
	parts := strings.Split(fileContent, oldStr)
	if len(parts) == 0 {
		// This should never happen due to the earlier check, but let's be safe
		parts = []string{""}
	}
	replacementLine := strings.Count(parts[0], "\n")
	startLine := max(0, replacementLine-snippetLines)
	endLine := replacementLine + snippetLines + strings.Count(newStr, "\n")
	fileLines := strings.Split(newFileContent, "\n")
	if len(fileLines) == 0 {
		fileLines = []string{""}
	}
	endLine = min(endLine+1, len(fileLines))
	snippet := strings.Join(fileLines[startLine:endLine], "\n")

	// Prepare the success message
	successMsg := fmt.Sprintf("The file %s has been edited. ", path)
	successMsg += makeOutput(snippet, fmt.Sprintf("a snippet of %s", path), startLine+1)
	successMsg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

	return successMsg, nil
}

// handleInsert implements the insert command
func handleInsert(path string, insertLine int, newStr string) (string, error) {
	// Read the file content
	fileContent, err := readFile(path)
	if err != nil {
		return "", err
	}

	// Replace tabs with spaces
	fileContent = maybeExpandTabs(path, fileContent)
	newStr = maybeExpandTabs(path, newStr)

	// Split the file content into lines
	fileTextLines := strings.Split(fileContent, "\n")
	nLinesFile := len(fileTextLines)

	// Validate insert line
	if insertLine < 0 || insertLine > nLinesFile {
		return "", fmt.Errorf("invalid insert_line parameter: %d. It should be within the range of lines of the file: [0, %d]",
			insertLine, nLinesFile)
	}

	// Save the current content to history
	fileHistory[path] = append(fileHistory[path], fileContent)

	// Split the new string into lines
	newStrLines := strings.Split(newStr, "\n")

	// Create new content by inserting the new lines
	newFileTextLines := make([]string, 0, nLinesFile+len(newStrLines))
	newFileTextLines = append(newFileTextLines, fileTextLines[:insertLine]...)
	newFileTextLines = append(newFileTextLines, newStrLines...)
	newFileTextLines = append(newFileTextLines, fileTextLines[insertLine:]...)

	// Create a snippet of the edited section
	snippetStart := max(0, insertLine-snippetLines)
	snippetEnd := min(insertLine+snippetLines, nLinesFile)

	snippetLines := make([]string, 0)
	snippetLines = append(snippetLines, fileTextLines[snippetStart:insertLine]...)
	snippetLines = append(snippetLines, newStrLines...)
	snippetLines = append(snippetLines, fileTextLines[insertLine:snippetEnd]...)
	snippet := strings.Join(snippetLines, "\n")

	// Write the new content to the file
	newFileText := strings.Join(newFileTextLines, "\n")
	if err := writeFile(path, newFileText); err != nil {
		return "", err
	}

	// Prepare the success message
	successMsg := fmt.Sprintf("The file %s has been edited. ", path)
	successMsg += makeOutput(snippet, "a snippet of the edited file", max(1, insertLine-4+1))
	successMsg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

	return successMsg, nil
}

// handleUndoEdit implements the undo_edit command
func handleUndoEdit(path string) (string, error) {
	history, exists := fileHistory[path]
	if !exists || len(history) == 0 {
		return "", fmt.Errorf("no edit history found for %s", path)
	}

	// Get the last edit and remove it from history
	lastIdx := len(history) - 1
	oldText := history[lastIdx]
	fileHistory[path] = history[:lastIdx]

	// Write the old content back to the file
	if err := writeFile(path, oldText); err != nil {
		return "", err
	}

	return fmt.Sprintf("Last edit to %s undone successfully. %s", path, makeOutput(oldText, path, 1)), nil
}

// listDirectory lists files and directories up to 2 levels deep
func listDirectory(ctx context.Context, path string) (string, error) {
	cmd := fmt.Sprintf("find %s -maxdepth 2 -not -path '*/\\.*'", path)
	output, err := executeCommand(ctx, cmd)
	if err != nil {
		return "", fmt.Errorf("failed to list directory: %v", err)
	}

	return fmt.Sprintf("Here's the files and directories up to 2 levels deep in %s, excluding hidden items:\n%s\n", path, output), nil
}

// executeCommand executes a shell command and returns its output
func executeCommand(ctx context.Context, cmd string) (string, error) {
	// This is a simplified version without timeouts for now
	bash := exec.CommandContext(ctx, "bash", "-c", cmd)
	bash.Dir = WorkingDir(ctx)
	output, err := bash.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("command execution failed: %v: %s", err, string(output))
	}
	return maybetruncate(string(output)), nil
}

// readFile reads the content of a file
func readFile(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file %s: %v", path, err)
	}
	return string(content), nil
}

// writeFile writes content to a file
func writeFile(path, content string) error {
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return fmt.Errorf("failed to write to file %s: %v", path, err)
	}
	return nil
}

// makeOutput generates a formatted output for the CLI
func makeOutput(fileContent, fileDescriptor string, initLine int) string {
	fileContent = maybetruncate(fileContent)
	fileContent = maybeExpandTabs(fileDescriptor, fileContent)

	var lines []string
	for i, line := range strings.Split(fileContent, "\n") {
		lines = append(lines, fmt.Sprintf("%6d\t%s", i+initLine, line))
	}

	return fmt.Sprintf("Here's the result of running `cat -n` on %s:\n%s\n", fileDescriptor, strings.Join(lines, "\n"))
}

// maybetruncate truncates content and appends a notice if content exceeds the specified length
func maybetruncate(content string) string {
	if len(content) <= maxResponseLen {
		return content
	}
	return content[:maxResponseLen] + truncatedMessage
}

// maybeExpandTabs is currently a no-op. The python
// implementation replaces tabs with spaces, but this strikes
// me as unwise for our tool.
func maybeExpandTabs(path, s string) string {
	// return strings.ReplaceAll(s, "\t", "    ")
	return s
}
