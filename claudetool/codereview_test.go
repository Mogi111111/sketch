package claudetool

import (
	"bytes"
	"cmp"
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"golang.org/x/tools/txtar"
)

// updateTests is set to true when the -update flag is used.
// This will update the expected test results instead of failing tests.
var updateTests = flag.Bool("update", false, "update expected test results instead of failing tests")

// TestCodereviewDifferential runs all the end-to-end tests for the codereview and differential packages.
// Each test is defined in a .txtar file in the testdata directory.
func TestCodereviewDifferential(t *testing.T) {
	entries, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatalf("failed to read testdata directory: %v", err)
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".txtar") {
			continue
		}
		testPath := filepath.Join("testdata", entry.Name())
		testName := strings.TrimSuffix(entry.Name(), ".txtar")
		t.Run(testName, func(t *testing.T) {
			t.Parallel()
			runE2ETest(t, testPath, *updateTests)
		})
	}
}

// runE2ETest executes a single end-to-end test from a .txtar file.
func runE2ETest(t *testing.T, testPath string, update bool) {
	orig, err := os.ReadFile(testPath)
	if err != nil {
		t.Fatalf("failed to read test file %s: %v", testPath, err)
	}
	archive, err := txtar.ParseFile(testPath)
	if err != nil {
		t.Fatalf("failed to parse txtar file %s: %v", testPath, err)
	}

	tmpDir := t.TempDir()
	// resolve temp dir path so that we can canonicalize/normalize it later
	tmpDir = resolveRealPath(tmpDir)

	if err := initGoModule(tmpDir); err != nil {
		t.Fatalf("failed to initialize Go module: %v", err)
	}
	if err := initGitRepo(tmpDir); err != nil {
		t.Fatalf("failed to initialize git repository: %v", err)
	}
	if err := processTestFiles(t, archive, tmpDir, update); err != nil {
		t.Fatalf("error processing test files: %v", err)
	}

	// If we're updating, write back the modified archive to the file
	if update {
		updatedContent := txtar.Format(archive)
		// only write back changes, avoids git status churn
		if !bytes.Equal(orig, updatedContent) {
			if err := os.WriteFile(testPath, updatedContent, 0o644); err != nil {
				t.Errorf("Failed to update test file %s: %v", testPath, err)
			}
		}
	}
}

func gitCommitEnv() []string {
	return append(os.Environ(),
		"GIT_AUTHOR_NAME=Test Author",
		"GIT_AUTHOR_EMAIL=test@example.com",
		"GIT_COMMITTER_NAME=Test Committer",
		"GIT_COMMITTER_EMAIL=test@example.com",
		"GIT_AUTHOR_DATE=2025-01-01T00:00:00Z",
		"GIT_COMMITTER_DATE=2025-01-01T00:00:00Z",
	)
}

// initGitRepo initializes a new git repository in the specified directory.
func initGitRepo(dir string) error {
	cmd := exec.Command("git", "init")
	cmd.Dir = dir
	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("error initializing git repository: %w", err)
	}
	// create a single commit out of everything there now
	cmd = exec.Command("git", "add", ".")
	cmd.Dir = dir
	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("error staging files: %w", err)
	}
	cmd = exec.Command("git", "commit", "-m", "create repo")
	cmd.Dir = dir
	cmd.Env = gitCommitEnv()
	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("error making initial commit: %w", err)
	}
	return nil
}

// processTestFiles processes the files in the txtar archive in sequence.
func processTestFiles(t *testing.T, archive *txtar.Archive, dir string, update bool) error {
	var initialCommit string
	filesForNextCommit := make(map[string]bool)

	for i, file := range archive.Files {
		switch file.Name {
		case ".commit":
			commit, err := makeGitCommit(dir, string(file.Data), filesForNextCommit)
			if err != nil {
				return fmt.Errorf("error making git commit: %w", err)
			}
			clear(filesForNextCommit)
			initialCommit = cmp.Or(initialCommit, commit)
			// fmt.Println("initial commit:", initialCommit)
			// cmd := exec.Command("git", "log")
			// cmd.Dir = dir
			// cmd.Stdout = os.Stdout
			// cmd.Run()

		case ".run_test":
			got, err := runDifferentialTest(dir, initialCommit)
			if err != nil {
				return fmt.Errorf("error running differential test: %w", err)
			}
			want := string(file.Data)

			commitCleaner := strings.NewReplacer(initialCommit, "INITIAL_COMMIT_HASH")
			got = commitCleaner.Replace(got)
			want = commitCleaner.Replace(want)

			if update {
				archive.Files[i].Data = []byte(got)
				break
			}
			if strings.TrimSpace(want) != strings.TrimSpace(got) {
				t.Errorf("Results don't match.\nExpected:\n%s\n\nActual:\n%s", want, got)
			}

		case ".run_autoformat":
			if initialCommit == "" {
				return fmt.Errorf("initial commit not set, cannot run autoformat")
			}

			got, err := runAutoformat(dir, initialCommit)
			if err != nil {
				return fmt.Errorf("error running autoformat: %w", err)
			}
			slices.Sort(got)

			if update {
				correct := strings.Join(got, "\n") + "\n"
				archive.Files[i].Data = []byte(correct)
				break
			}

			want := strings.Split(strings.TrimSpace(string(file.Data)), "\n")
			if !slices.Equal(want, got) {
				t.Errorf("Formatted files don't match.\nExpected:\n%s\n\nActual:\n%s", want, got)
			}

		default:
			filePath := filepath.Join(dir, file.Name)
			if err := os.MkdirAll(filepath.Dir(filePath), 0o700); err != nil {
				return fmt.Errorf("error creating directory for %s: %w", file.Name, err)
			}
			data := file.Data
			// Remove second trailing newline if present.
			// An annoyance of the txtar format, messes with gofmt.
			if bytes.HasSuffix(data, []byte("\n\n")) {
				data = bytes.TrimSuffix(data, []byte("\n"))
			}
			if err := os.WriteFile(filePath, file.Data, 0o600); err != nil {
				return fmt.Errorf("error writing file %s: %w", file.Name, err)
			}
			filesForNextCommit[file.Name] = true
		}
	}

	return nil
}

// makeGitCommit commits the specified files with the given message.
// Returns the commit hash.
func makeGitCommit(dir, message string, files map[string]bool) (string, error) {
	for file := range files {
		cmd := exec.Command("git", "add", file)
		cmd.Dir = dir
		if err := cmd.Run(); err != nil {
			return "", fmt.Errorf("error staging file %s: %w", file, err)
		}
	}
	message = cmp.Or(message, "Test commit")

	// Make the commit with fixed author, committer, and date for stable hashes
	cmd := exec.Command("git", "commit", "--allow-empty", "-m", message)
	cmd.Dir = dir
	cmd.Env = gitCommitEnv()
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("error making commit: %w", err)
	}

	// Get the commit hash
	cmd = exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("error getting commit hash: %w", err)
	}

	return strings.TrimSpace(string(out)), nil
}

// runDifferentialTest runs the code review tool on the repository and returns the result.
func runDifferentialTest(dir, initialCommit string) (string, error) {
	if initialCommit == "" {
		return "", fmt.Errorf("initial commit not set, cannot run differential test")
	}

	// Create a code reviewer for the repository
	ctx := context.Background()
	reviewer, err := NewCodeReviewer(ctx, dir, initialCommit)
	if err != nil {
		return "", fmt.Errorf("error creating code reviewer: %w", err)
	}

	// Run the code review
	result, err := reviewer.Run(ctx, nil)
	if err != nil {
		return "", fmt.Errorf("error running code review: %w", err)
	}

	// Normalize paths in the result
	normalized := normalizePaths(result, dir)
	return normalized, nil
}

// normalizePaths replaces the temp directory paths with a standard placeholder
func normalizePaths(result string, tempDir string) string {
	return strings.ReplaceAll(result, tempDir, "/PATH/TO/REPO")
}

// initGoModule initializes a Go module in the specified directory.
func initGoModule(dir string) error {
	cmd := exec.Command("go", "mod", "init", "sketch.dev")
	cmd.Dir = dir
	return cmd.Run()
}

// runAutoformat runs the autoformat function on the repository and returns the list of formatted files.
func runAutoformat(dir, initialCommit string) ([]string, error) {
	if initialCommit == "" {
		return nil, fmt.Errorf("initial commit not set, cannot run autoformat")
	}
	ctx := context.Background()
	reviewer, err := NewCodeReviewer(ctx, dir, initialCommit)
	if err != nil {
		return nil, fmt.Errorf("error creating code reviewer: %w", err)
	}
	formattedFiles := reviewer.Autoformat(ctx)
	normalizedFiles := make([]string, len(formattedFiles))
	for i, file := range formattedFiles {
		normalizedFiles[i] = normalizePaths(file, dir)
	}
	slices.Sort(normalizedFiles)
	return normalizedFiles, nil
}

// resolveRealPath follows symlinks and returns the real path
// This handles platform-specific behaviors like macOS's /private prefix
func resolveRealPath(path string) string {
	// Follow symlinks to get the real path
	realPath, err := filepath.EvalSymlinks(path)
	if err != nil {
		// If we can't resolve symlinks, just return the original path
		return path
	}
	return realPath
}
