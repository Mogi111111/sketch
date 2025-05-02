package claudetool

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/packages"
	"sketch.dev/ant"
)

// This file does differential quality analysis of a commit relative to a base commit.

// Tool returns a tool spec for a CodeReview tool backed by r.
func (r *CodeReviewer) Tool() *ant.Tool {
	spec := &ant.Tool{
		Name:        "codereview",
		Description: `Run an automated code review.`,
		// If you modify this, update the termui template for prettier rendering.
		InputSchema: ant.MustSchema(`{"type": "object", "properties": {}}`),
		Run:         r.Run,
	}
	return spec
}

func (r *CodeReviewer) Run(ctx context.Context, m json.RawMessage) (string, error) {
	// NOTE: If you add or modify error messages here, update the corresponding UI parsing in:
	// webui/src/web-components/sketch-tool-card.ts (SketchToolCardCodeReview.getStatusIcon)
	if err := r.RequireNormalGitState(ctx); err != nil {
		slog.DebugContext(ctx, "CodeReviewer.Run: failed to check for normal git state", "err", err)
		return "", err
	}
	if err := r.RequireNoUncommittedChanges(ctx); err != nil {
		slog.DebugContext(ctx, "CodeReviewer.Run: failed to check for uncommitted changes", "err", err)
		return "", err
	}

	// Check that the current commit is not the initial commit
	currentCommit, err := r.CurrentCommit(ctx)
	if err != nil {
		slog.DebugContext(ctx, "CodeReviewer.Run: failed to get current commit", "err", err)
		return "", err
	}
	if r.IsInitialCommit(currentCommit) {
		slog.DebugContext(ctx, "CodeReviewer.Run: current commit is initial commit, nothing to review")
		return "", fmt.Errorf("no new commits have been added, nothing to review")
	}

	// No matter what failures happen from here out, we will declare this to have been reviewed.
	// This should help avoid the model getting blocked by a broken code review tool.
	r.reviewed = append(r.reviewed, currentCommit)

	changedFiles, err := r.changedFiles(ctx, r.initialCommit, currentCommit)
	if err != nil {
		slog.DebugContext(ctx, "CodeReviewer.Run: failed to get changed files", "err", err)
		return "", err
	}

	// Prepare to analyze before/after for the impacted files.
	// We use the current commit to determine what packages exist and are impacted.
	// The packages in the initial commit may be different.
	// Good enough for now.
	// TODO: do better
	allPkgs, err := r.packagesForFiles(ctx, changedFiles)
	if err != nil {
		// TODO: log and skip to stuff that doesn't require packages
		slog.DebugContext(ctx, "CodeReviewer.Run: failed to get packages for files", "err", err)
		return "", err
	}
	allPkgList := slices.Collect(maps.Keys(allPkgs))

	var errorMessages []string // problems we want the model to address
	var infoMessages []string  // info the model should consider

	// Find potentially related files that should also be considered
	// TODO: add some caching here, since this depends only on the initial commit and the changed files, not the details of the changes
	relatedFiles, err := r.findRelatedFiles(ctx, changedFiles)
	if err != nil {
		slog.DebugContext(ctx, "CodeReviewer.Run: failed to find related files", "err", err)
	} else {
		relatedMsg := r.formatRelatedFiles(relatedFiles)
		if relatedMsg != "" {
			infoMessages = append(infoMessages, relatedMsg)
		}
	}

	testMsg, err := r.checkTests(ctx, allPkgList)
	if err != nil {
		slog.DebugContext(ctx, "CodeReviewer.Run: failed to check tests", "err", err)
		return "", err
	}
	if testMsg != "" {
		errorMessages = append(errorMessages, testMsg)
	}

	goplsMsg, err := r.checkGopls(ctx, changedFiles) // includes vet checks
	if err != nil {
		slog.DebugContext(ctx, "CodeReviewer.Run: failed to check gopls", "err", err)
		return "", err
	}
	if goplsMsg != "" {
		errorMessages = append(errorMessages, goplsMsg)
	}

	// NOTE: If you change this output format, update the corresponding UI parsing in:
	// webui/src/web-components/sketch-tool-card.ts (SketchToolCardCodeReview.getStatusIcon)
	buf := new(strings.Builder)
	if len(infoMessages) > 0 {
		buf.WriteString("# Info\n\n")
		buf.WriteString(strings.Join(infoMessages, "\n\n"))
		buf.WriteString("\n\n")
	}
	if len(errorMessages) > 0 {
		buf.WriteString("# Errors\n\n")
		buf.WriteString(strings.Join(errorMessages, "\n\n"))
		buf.WriteString("\n\nPlease fix before proceeding.\n")
	}
	if buf.Len() == 0 {
		buf.WriteString("OK")
	}
	return buf.String(), nil
}

func (r *CodeReviewer) initializeInitialCommitWorktree(ctx context.Context) error {
	if r.initialWorktree != "" {
		return nil
	}
	tmpDir, err := os.MkdirTemp("", "sketch-codereview-worktree")
	if err != nil {
		return err
	}
	worktreeCmd := exec.CommandContext(ctx, "git", "worktree", "add", "--detach", tmpDir, r.initialCommit)
	worktreeCmd.Dir = r.repoRoot
	out, err := worktreeCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("unable to create worktree for initial commit: %w\n%s", err, out)
	}
	r.initialWorktree = tmpDir
	return nil
}

func (r *CodeReviewer) checkTests(ctx context.Context, pkgList []string) (string, error) {
	goTestArgs := []string{"test", "-json", "-v"}
	goTestArgs = append(goTestArgs, pkgList...)

	afterTestCmd := exec.CommandContext(ctx, "go", goTestArgs...)
	afterTestCmd.Dir = r.repoRoot
	afterTestOut, _ := afterTestCmd.Output()
	// unfortunately, we can't short-circuit here even if all tests pass,
	// because we need to check for skipped tests.

	err := r.initializeInitialCommitWorktree(ctx)
	if err != nil {
		return "", err
	}

	beforeTestCmd := exec.CommandContext(ctx, "go", goTestArgs...)
	beforeTestCmd.Dir = r.initialWorktree
	beforeTestOut, _ := beforeTestCmd.Output() // ignore error, interesting info is in the output

	// Parse the jsonl test results
	beforeResults, beforeParseErr := parseTestResults(beforeTestOut)
	if beforeParseErr != nil {
		return "", fmt.Errorf("unable to parse test results for initial commit: %w\n%s", beforeParseErr, beforeTestOut)
	}
	afterResults, afterParseErr := parseTestResults(afterTestOut)
	if afterParseErr != nil {
		return "", fmt.Errorf("unable to parse test results for current commit: %w\n%s", afterParseErr, afterTestOut)
	}
	testRegressions, err := r.compareTestResults(beforeResults, afterResults)
	if err != nil {
		return "", fmt.Errorf("failed to compare test results: %w", err)
	}
	// TODO: better output formatting?
	res := r.formatTestRegressions(testRegressions)
	return res, nil
}

// GoplsIssue represents a single issue reported by gopls check
type GoplsIssue struct {
	Position string // File position in format "file:line:col-range"
	Message  string // Description of the issue
}

// checkGopls runs gopls check on the provided files in both the current and initial state,
// compares the results, and reports any new issues introduced in the current state.
func (r *CodeReviewer) checkGopls(ctx context.Context, changedFiles []string) (string, error) {
	if len(changedFiles) == 0 {
		return "", nil // no files to check
	}

	// Filter out non-Go files as gopls only works on Go files
	// and verify they still exist (not deleted)
	var goFiles []string
	for _, file := range changedFiles {
		if !strings.HasSuffix(file, ".go") {
			continue // not a Go file
		}

		// Check if the file still exists (not deleted)
		if _, err := os.Stat(file); os.IsNotExist(err) {
			continue // file doesn't exist anymore (deleted)
		}

		goFiles = append(goFiles, file)
	}

	if len(goFiles) == 0 {
		return "", nil // no Go files to check
	}

	// Run gopls check on the current state
	goplsArgs := append([]string{"check"}, goFiles...)

	afterGoplsCmd := exec.CommandContext(ctx, "gopls", goplsArgs...)
	afterGoplsCmd.Dir = r.repoRoot
	afterGoplsOut, err := afterGoplsCmd.CombinedOutput() // gopls returns non-zero if it finds issues
	if err != nil {
		// Check if the output looks like real gopls issues or if it's just error output
		if !looksLikeGoplsIssues(afterGoplsOut) {
			slog.WarnContext(ctx, "CodeReviewer.checkGopls: gopls check failed to run properly", "err", err, "output", string(afterGoplsOut))
			return "", nil // Skip rather than failing the entire code review
		}
		// Otherwise, proceed with parsing - it's likely just the non-zero exit code due to found issues
	}

	// Parse the output
	afterIssues := parseGoplsOutput(r.repoRoot, afterGoplsOut)

	// If no issues were found, we're done
	if len(afterIssues) == 0 {
		return "", nil
	}

	// Gopls detected issues in the current state, check if they existed in the initial state
	initErr := r.initializeInitialCommitWorktree(ctx)
	if initErr != nil {
		return "", err
	}

	// For each file that exists in the initial commit, run gopls check
	var initialFilesToCheck []string
	for _, file := range goFiles {
		// Get relative path for git operations
		relFile, err := filepath.Rel(r.repoRoot, file)
		if err != nil {
			slog.WarnContext(ctx, "CodeReviewer.checkGopls: failed to get relative path", "repo_root", r.repoRoot, "file", file, "err", err)
			continue
		}

		// Check if the file exists in the initial commit
		checkCmd := exec.CommandContext(ctx, "git", "cat-file", "-e", fmt.Sprintf("%s:%s", r.initialCommit, relFile))
		checkCmd.Dir = r.repoRoot
		if err := checkCmd.Run(); err == nil {
			// File exists in initial commit
			initialFilePath := filepath.Join(r.initialWorktree, relFile)
			initialFilesToCheck = append(initialFilesToCheck, initialFilePath)
		}
	}

	// Run gopls check on the files that existed in the initial commit
	var beforeIssues []GoplsIssue
	if len(initialFilesToCheck) > 0 {
		beforeGoplsArgs := append([]string{"check"}, initialFilesToCheck...)
		beforeGoplsCmd := exec.CommandContext(ctx, "gopls", beforeGoplsArgs...)
		beforeGoplsCmd.Dir = r.initialWorktree
		beforeGoplsOut, beforeCmdErr := beforeGoplsCmd.CombinedOutput()
		if beforeCmdErr != nil && !looksLikeGoplsIssues(beforeGoplsOut) {
			// If gopls fails to run properly on the initial commit, log a warning and continue
			// with empty before issues - this will be conservative and report more issues
			slog.WarnContext(ctx, "CodeReviewer.checkGopls: gopls check failed on initial commit", "err", err, "output", string(beforeGoplsOut))
		} else {
			beforeIssues = parseGoplsOutput(r.initialWorktree, beforeGoplsOut)
		}
	}

	// Find new issues that weren't present in the initial state
	goplsRegressions := findGoplsRegressions(beforeIssues, afterIssues)
	if len(goplsRegressions) == 0 {
		return "", nil // no new issues
	}

	// Format the results
	return r.formatGoplsRegressions(goplsRegressions), nil
}

// parseGoplsOutput parses the text output from gopls check
// Each line has the format: '/path/to/file.go:448:22-26: unused parameter: path'
func parseGoplsOutput(root string, output []byte) []GoplsIssue {
	var issues []GoplsIssue
	lines := strings.Split(string(output), "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Skip lines that look like error messages rather than gopls issues
		if strings.HasPrefix(line, "Error:") ||
			strings.HasPrefix(line, "Failed:") ||
			strings.HasPrefix(line, "Warning:") ||
			strings.HasPrefix(line, "gopls:") {
			continue
		}

		// Find the first colon that separates the file path from the line number
		firstColonIdx := strings.Index(line, ":")
		if firstColonIdx < 0 {
			continue // Invalid format
		}

		// Verify the part before the first colon looks like a file path
		potentialPath := line[:firstColonIdx]
		if !strings.HasSuffix(potentialPath, ".go") {
			continue // Not a Go file path
		}

		// Find the position of the first message separator ': '
		// This separates the position info from the message
		messageStart := strings.Index(line, ": ")
		if messageStart < 0 || messageStart <= firstColonIdx {
			continue // Invalid format
		}

		// Extract position and message
		position := line[:messageStart]
		rel, err := filepath.Rel(root, position)
		if err == nil {
			position = rel
		}
		message := line[messageStart+2:] // Skip the ': ' separator

		// Verify position has the expected format (at least 2 colons for line:col)
		colonCount := strings.Count(position, ":")
		if colonCount < 2 {
			continue // Not enough position information
		}

		issues = append(issues, GoplsIssue{
			Position: position,
			Message:  message,
		})
	}

	return issues
}

// looksLikeGoplsIssues checks if the output appears to be actual gopls issues
// rather than error messages about gopls itself failing
func looksLikeGoplsIssues(output []byte) bool {
	// If output is empty, it's not valid issues
	if len(output) == 0 {
		return false
	}

	// Check if output has at least one line that looks like a gopls issue
	// A gopls issue looks like: '/path/to/file.go:123:45-67: message'
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// A gopls issue has at least two colons (file path, line number, column)
		// and contains a colon followed by a space (separating position from message)
		colonCount := strings.Count(line, ":")
		hasSeparator := strings.Contains(line, ": ")

		if colonCount >= 2 && hasSeparator {
			// Check if it starts with a likely file path (ending in .go)
			parts := strings.SplitN(line, ":", 2)
			if strings.HasSuffix(parts[0], ".go") {
				return true
			}
		}
	}
	return false
}

// normalizeGoplsPosition extracts just the file path from a position string
func normalizeGoplsPosition(position string) string {
	// Extract just the file path by taking everything before the first colon
	parts := strings.Split(position, ":")
	if len(parts) < 1 {
		return position
	}
	return parts[0]
}

// findGoplsRegressions identifies gopls issues that are new in the after state
func findGoplsRegressions(before, after []GoplsIssue) []GoplsIssue {
	var regressions []GoplsIssue

	// Build map of before issues for easier lookup
	beforeIssueMap := make(map[string]map[string]bool) // file -> message -> exists
	for _, issue := range before {
		file := normalizeGoplsPosition(issue.Position)
		if _, exists := beforeIssueMap[file]; !exists {
			beforeIssueMap[file] = make(map[string]bool)
		}
		// Store both the exact message and the general issue type for fuzzy matching
		beforeIssueMap[file][issue.Message] = true

		// Extract the general issue type (everything before the first ':' in the message)
		generalIssue := issue.Message
		if colonIdx := strings.Index(issue.Message, ":"); colonIdx > 0 {
			generalIssue = issue.Message[:colonIdx]
		}
		beforeIssueMap[file][generalIssue] = true
	}

	// Check each after issue to see if it's new
	for _, afterIssue := range after {
		file := normalizeGoplsPosition(afterIssue.Position)
		isNew := true

		if fileIssues, fileExists := beforeIssueMap[file]; fileExists {
			// Check for exact message match
			if fileIssues[afterIssue.Message] {
				isNew = false
			} else {
				// Check for general issue type match
				generalIssue := afterIssue.Message
				if colonIdx := strings.Index(afterIssue.Message, ":"); colonIdx > 0 {
					generalIssue = afterIssue.Message[:colonIdx]
				}
				if fileIssues[generalIssue] {
					isNew = false
				}
			}
		}

		if isNew {
			regressions = append(regressions, afterIssue)
		}
	}

	// Sort regressions for deterministic output
	slices.SortFunc(regressions, func(a, b GoplsIssue) int {
		return strings.Compare(a.Position, b.Position)
	})

	return regressions
}

// formatGoplsRegressions generates a human-readable summary of gopls check regressions
func (r *CodeReviewer) formatGoplsRegressions(regressions []GoplsIssue) string {
	if len(regressions) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("Gopls check issues detected:\n\n")

	// Format each issue
	for i, issue := range regressions {
		sb.WriteString(fmt.Sprintf("%d. %s: %s\n", i+1, filepath.Join(r.repoRoot, issue.Position), issue.Message))
	}

	sb.WriteString("\nIMPORTANT: Only fix new gopls check issues in parts of the code that you have already edited.")
	sb.WriteString(" Do not change existing code that was not part of your current edits.\n")
	return sb.String()
}

func (r *CodeReviewer) HasReviewed(commit string) bool {
	return slices.Contains(r.reviewed, commit)
}

func (r *CodeReviewer) IsInitialCommit(commit string) bool {
	return commit == r.initialCommit
}

// packagesForFiles returns maps of packages related to the given files:
// 1. directPkgs: packages that directly contain the changed files
// 2. allPkgs: all packages that might be affected, including downstream packages that depend on the direct packages
// It may include false positives.
// Files must be absolute paths!
func (r *CodeReviewer) packagesForFiles(ctx context.Context, files []string) (allPkgs map[string]*packages.Package, err error) {
	for _, f := range files {
		if !filepath.IsAbs(f) {
			return nil, fmt.Errorf("path %q is not absolute", f)
		}
	}
	cfg := &packages.Config{
		Mode:    packages.LoadImports | packages.NeedEmbedFiles,
		Context: ctx,
		// Logf: func(msg string, args ...any) {
		// 	slog.DebugContext(ctx, "loading go packages", "msg", fmt.Sprintf(msg, args...))
		// },
		// TODO: in theory, go.mod might not be in the repo root, and there might be multiple go.mod files.
		// We can cross that bridge when we get there.
		Dir:   r.repoRoot,
		Tests: true,
	}
	universe, err := packages.Load(cfg, "./...")
	if err != nil {
		return nil, err
	}
	// Identify packages that directly contain the changed files
	directPkgs := make(map[string]*packages.Package) // import path -> package
	for _, pkg := range universe {
		pkgFiles := allFiles(pkg)
		for _, file := range files {
			if pkgFiles[file] {
				// prefer test packages, as they contain strictly more files (right?)
				prev := directPkgs[pkg.PkgPath]
				if prev == nil || prev.ForTest == "" {
					directPkgs[pkg.PkgPath] = pkg
				}
			}
		}
	}

	allPkgs = maps.Clone(directPkgs)

	// Add packages that depend on the direct packages
	addDependentPackages(universe, allPkgs)
	return allPkgs, nil
}

// allFiles returns all files that might be referenced by the package.
// It may contain false positives.
func allFiles(p *packages.Package) map[string]bool {
	files := make(map[string]bool)
	// Add files from package info
	add := [][]string{p.GoFiles, p.CompiledGoFiles, p.OtherFiles, p.EmbedFiles, p.IgnoredFiles}
	for _, extra := range add {
		for _, file := range extra {
			files[file] = true
		}
	}
	// Add files from testdata directory
	testdataDir := filepath.Join(p.Dir, "testdata")
	if _, err := os.Stat(testdataDir); err == nil {
		fsys := os.DirFS(p.Dir)
		fs.WalkDir(fsys, "testdata", func(path string, d fs.DirEntry, err error) error {
			if err == nil && !d.IsDir() {
				files[filepath.Join(p.Dir, path)] = true
			}
			return nil
		})
	}
	return files
}

// addDependentPackages adds to pkgs all packages from universe
// that directly or indirectly depend on any package already in pkgs.
func addDependentPackages(universe []*packages.Package, pkgs map[string]*packages.Package) {
	for {
		changed := false
		for _, p := range universe {
			if strings.HasSuffix(p.PkgPath, ".test") { // ick, but I don't see another way
				// skip test packages
				continue
			}
			if _, ok := pkgs[p.PkgPath]; ok {
				// already in pkgs
				continue
			}
			for importPath := range p.Imports {
				if _, ok := pkgs[importPath]; ok {
					// imports a package dependent on pkgs, add it
					pkgs[p.PkgPath] = p
					changed = true
					break
				}
			}
		}
		if !changed {
			break
		}
	}
}

// testJSON is a union of BuildEvent and TestEvent
type testJSON struct {
	// TestEvent only:
	// The Time field holds the time the event happened. It is conventionally omitted
	// for cached test results.
	Time time.Time `json:"Time"`
	// BuildEvent only:
	// The ImportPath field gives the package ID of the package being built.
	// This matches the Package.ImportPath field of go list -json and the
	// TestEvent.FailedBuild field of go test -json. Note that it does not
	// match TestEvent.Package.
	ImportPath string `json:"ImportPath"` // BuildEvent only
	// TestEvent only:
	// The Package field, if present, specifies the package being tested. When the
	// go command runs parallel tests in -json mode, events from different tests are
	// interlaced; the Package field allows readers to separate them.
	Package string `json:"Package"`
	// Action is used in both BuildEvent and TestEvent.
	// It is the key to distinguishing between them.
	// BuildEvent:
	// build-output or build-fail
	// TestEvent:
	// start, run, pause, cont, pass, bench, fail, output, skip
	Action string `json:"Action"`
	// TestEvent only:
	// The Test field, if present, specifies the test, example, or benchmark function
	// that caused the event. Events for the overall package test do not set Test.
	Test string `json:"Test"`
	// TestEvent only:
	// The Elapsed field is set for "pass" and "fail" events. It gives the time elapsed in seconds
	// for the specific test or the overall package test that passed or failed.
	Elapsed float64
	// TestEvent:
	// The Output field is set for Action == "output" and is a portion of the
	// test's output (standard output and standard error merged together). The
	// output is unmodified except that invalid UTF-8 output from a test is coerced
	// into valid UTF-8 by use of replacement characters. With that one exception,
	// the concatenation of the Output fields of all output events is the exact output
	// of the test execution.
	// BuildEvent:
	// The Output field is set for Action == "build-output" and is a portion of
	// the build's output. The concatenation of the Output fields of all output
	// events is the exact output of the build. A single event may contain one
	// or more lines of output and there may be more than one output event for
	// a given ImportPath. This matches the definition of the TestEvent.Output
	// field produced by go test -json.
	Output string `json:"Output"`
	// TestEvent only:
	// The FailedBuild field is set for Action == "fail" if the test failure was caused
	// by a build failure. It contains the package ID of the package that failed to
	// build. This matches the ImportPath field of the "go list" output, as well as the
	// BuildEvent.ImportPath field as emitted by "go build -json".
	FailedBuild string `json:"FailedBuild"`
}

// parseTestResults converts test output in JSONL format into a slice of testJSON objects
func parseTestResults(testOutput []byte) ([]testJSON, error) {
	var results []testJSON
	dec := json.NewDecoder(bytes.NewReader(testOutput))
	for {
		var event testJSON
		if err := dec.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		results = append(results, event)
	}
	return results, nil
}

// testStatus represents the status of a test in a given commit
type testStatus int

//go:generate go tool stringer -type=testStatus -trimprefix=testStatus
const (
	testStatusUnknown testStatus = iota
	testStatusPass
	testStatusFail
	testStatusBuildFail
	testStatusSkip
	testStatusNoTests // no tests exist for this package
)

// testRegression represents a test that regressed between commits
type testRegression struct {
	Package      string
	Test         string // empty for package tests
	BeforeStatus testStatus
	AfterStatus  testStatus
	Output       string // failure output in the after state
}

func (r *testRegression) Source() string {
	if r.Test == "" {
		return r.Package
	}
	return fmt.Sprintf("%s.%s", r.Package, r.Test)
}

type packageResult struct {
	Status     testStatus            // overall status for the package
	TestStatus map[string]testStatus // name -> status
	TestOutput map[string][]string   // name -> output parts
}

// collectTestStatuses processes a slice of test events and returns rich status information
func collectTestStatuses(results []testJSON) map[string]*packageResult {
	m := make(map[string]*packageResult)

	for _, event := range results {
		pkg := event.Package
		p, ok := m[pkg]
		if !ok {
			p = new(packageResult)
			p.TestStatus = make(map[string]testStatus)
			p.TestOutput = make(map[string][]string)
			m[pkg] = p
		}

		switch event.Action {
		case "output":
			p.TestOutput[event.Test] = append(p.TestOutput[event.Test], event.Output)
			continue
		case "pass":
			if event.Test == "" {
				p.Status = testStatusPass
			} else {
				p.TestStatus[event.Test] = testStatusPass
			}
		case "fail":
			if event.Test == "" {
				if event.FailedBuild != "" {
					p.Status = testStatusBuildFail
				} else {
					p.Status = testStatusFail
				}
			} else {
				p.TestStatus[event.Test] = testStatusFail
			}
		case "skip":
			if event.Test == "" {
				p.Status = testStatusNoTests
			} else {
				p.TestStatus[event.Test] = testStatusSkip
			}
		}
	}

	return m
}

// compareTestResults identifies tests that have regressed between commits
func (r *CodeReviewer) compareTestResults(beforeResults, afterResults []testJSON) ([]testRegression, error) {
	before := collectTestStatuses(beforeResults)
	after := collectTestStatuses(afterResults)
	var testLevelRegressions []testRegression
	var packageLevelRegressions []testRegression

	afterPkgs := slices.Sorted(maps.Keys(after))
	for _, pkg := range afterPkgs {
		afterResult := after[pkg]
		afterStatus := afterResult.Status
		// Can't short-circuit here when tests are passing, because we need to check for skipped tests.
		beforeResult, ok := before[pkg]
		beforeStatus := testStatusNoTests
		if ok {
			beforeStatus = beforeResult.Status
		}
		// If things no longer build, stop at the package level.
		// Otherwise, proceed to the test level, so we have more precise information.
		if afterStatus == testStatusBuildFail && beforeStatus != testStatusBuildFail {
			packageLevelRegressions = append(packageLevelRegressions, testRegression{
				Package:      pkg,
				BeforeStatus: beforeStatus,
				AfterStatus:  afterStatus,
			})
			continue
		}
		tests := slices.Sorted(maps.Keys(afterResult.TestStatus))
		for _, test := range tests {
			afterStatus := afterResult.TestStatus[test]
			switch afterStatus {
			case testStatusPass:
				continue
			case testStatusUnknown:
				slog.WarnContext(context.Background(), "unknown test status", "package", pkg, "test", test)
				continue
			}
			beforeStatus := beforeResult.TestStatus[test]
			if isRegression(beforeStatus, afterStatus) {
				testLevelRegressions = append(testLevelRegressions, testRegression{
					Package:      pkg,
					Test:         test,
					BeforeStatus: beforeStatus,
					AfterStatus:  afterStatus,
					Output:       strings.Join(afterResult.TestOutput[test], ""),
				})
			}
		}
	}

	// If we have test-level regressions, report only those
	// Otherwise, report package-level regressions
	var regressions []testRegression
	if len(testLevelRegressions) > 0 {
		regressions = testLevelRegressions
	} else {
		regressions = packageLevelRegressions
	}

	// Sort regressions for consistent output
	slices.SortFunc(regressions, func(a, b testRegression) int {
		// First by package
		if c := strings.Compare(a.Package, b.Package); c != 0 {
			return c
		}
		// Then by test name
		return strings.Compare(a.Test, b.Test)
	})

	return regressions, nil
}

// badnessLevels maps test status to a badness level
// Higher values indicate worse status (more severe issues)
var badnessLevels = map[testStatus]int{
	testStatusBuildFail: 5, // Worst
	testStatusFail:      4,
	testStatusSkip:      3,
	testStatusNoTests:   2,
	testStatusPass:      1,
	testStatusUnknown:   0, // Least bad - avoids false positives
}

// regressionFormatter defines a mapping of before/after state pairs to descriptive messages
type regressionKey struct {
	before, after testStatus
}

var regressionMessages = map[regressionKey]string{
	{testStatusUnknown, testStatusBuildFail}: "New test has build/vet errors",
	{testStatusUnknown, testStatusFail}:      "New test is failing",
	{testStatusUnknown, testStatusSkip}:      "New test is skipped",
	{testStatusPass, testStatusBuildFail}:    "Was passing, now has build/vet errors",
	{testStatusPass, testStatusFail}:         "Was passing, now failing",
	{testStatusPass, testStatusSkip}:         "Was passing, now skipped",
	{testStatusNoTests, testStatusBuildFail}: "Previously had no tests, now has build/vet errors",
	{testStatusNoTests, testStatusFail}:      "Previously had no tests, now has failing tests",
	{testStatusNoTests, testStatusSkip}:      "Previously had no tests, now has skipped tests",
	{testStatusSkip, testStatusBuildFail}:    "Was skipped, now has build/vet errors",
	{testStatusSkip, testStatusFail}:         "Was skipped, now failing",
	{testStatusFail, testStatusBuildFail}:    "Was failing, now has build/vet errors",
}

// isRegression determines if a test has regressed based on before and after status
// A regression is defined as an increase in badness level
func isRegression(before, after testStatus) bool {
	// Higher badness level means worse status
	return badnessLevels[after] > badnessLevels[before]
}

// formatTestRegressions generates a human-readable summary of test regressions
func (r *CodeReviewer) formatTestRegressions(regressions []testRegression) string {
	if len(regressions) == 0 {
		return ""
	}

	buf := new(strings.Builder)
	fmt.Fprintf(buf, "Test regressions detected between initial commit (%s) and HEAD:\n\n", r.initialCommit)

	for i, reg := range regressions {
		fmt.Fprintf(buf, "%d: %v: ", i+1, reg.Source())
		key := regressionKey{reg.BeforeStatus, reg.AfterStatus}
		message, exists := regressionMessages[key]
		if !exists {
			message = "Regression detected"
		}
		fmt.Fprintf(buf, "%s\n", message)
	}

	return buf.String()
}

// RelatedFile represents a file historically related to the changed files
type RelatedFile struct {
	Path        string  // Path to the file
	Correlation float64 // Correlation score (0.0-1.0)
}

// findRelatedFiles identifies files that are historically related to the changed files
// by analyzing git commit history for co-occurrences.
func (r *CodeReviewer) findRelatedFiles(ctx context.Context, changedFiles []string) ([]RelatedFile, error) {
	commits, err := r.getCommitsTouchingFiles(ctx, changedFiles)
	if err != nil {
		return nil, fmt.Errorf("failed to get commits touching files: %w", err)
	}
	if len(commits) == 0 {
		return nil, nil
	}

	relChanged := make(map[string]bool, len(changedFiles))
	for _, file := range changedFiles {
		rel, err := filepath.Rel(r.repoRoot, file)
		if err != nil {
			return nil, fmt.Errorf("failed to get relative path for %s: %w", file, err)
		}
		relChanged[rel] = true
	}

	historyFiles := make(map[string]int)
	var historyMu sync.Mutex

	maxWorkers := runtime.GOMAXPROCS(0)
	semaphore := make(chan bool, maxWorkers)
	var wg sync.WaitGroup

	for _, commit := range commits {
		wg.Add(1)
		semaphore <- true // acquire

		go func(commit string) {
			defer wg.Done()
			defer func() { <-semaphore }() // release
			commitFiles, err := r.getFilesInCommit(ctx, commit)
			if err != nil {
				slog.WarnContext(ctx, "Failed to get files in commit", "commit", commit, "err", err)
				return
			}
			incr := 0
			for _, file := range commitFiles {
				if relChanged[file] {
					incr++
				}
			}
			if incr == 0 {
				return
			}
			historyMu.Lock()
			defer historyMu.Unlock()
			for _, file := range commitFiles {
				historyFiles[file] += incr
			}
		}(commit)
	}
	wg.Wait()

	// normalize
	maxCount := 0
	for _, count := range historyFiles {
		maxCount = max(maxCount, count)
	}
	if maxCount == 0 {
		return nil, nil
	}

	var relatedFiles []RelatedFile
	for file, count := range historyFiles {
		if relChanged[file] {
			// Don't include inputs in the output.
			continue
		}
		correlation := float64(count) / float64(maxCount)
		// Require min correlation to avoid noise
		if correlation >= 0.1 {
			relatedFiles = append(relatedFiles, RelatedFile{Path: file, Correlation: correlation})
		}
	}

	// Highest correlation first
	slices.SortFunc(relatedFiles, func(a, b RelatedFile) int {
		return -1 * cmp.Compare(a.Correlation, b.Correlation)
	})

	// Limit to 1 correlated file per input file.
	// (Arbitrary limit, to be adjusted.)
	maxFiles := len(changedFiles)
	if len(relatedFiles) > maxFiles {
		relatedFiles = relatedFiles[:maxFiles]
	}

	// TODO: add an LLM in the mix here (like the keyword search tool) to do a filtering pass,
	// and then increase the strength of the wording in the relatedFiles message.

	return relatedFiles, nil
}

// getCommitsTouchingFiles returns all commits that touch any of the specified files
func (r *CodeReviewer) getCommitsTouchingFiles(ctx context.Context, files []string) ([]string, error) {
	if len(files) == 0 {
		return nil, nil
	}
	fileArgs := append([]string{"rev-list", "--all", "--date-order", "--max-count=100", "--"}, files...)
	cmd := exec.CommandContext(ctx, "git", fileArgs...)
	cmd.Dir = r.repoRoot
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed to get commits: %w\n%s", err, out)
	}
	return nonEmptyTrimmedLines(out), nil
}

// getFilesInCommit returns all files changed in a specific commit
func (r *CodeReviewer) getFilesInCommit(ctx context.Context, commit string) ([]string, error) {
	cmd := exec.CommandContext(ctx, "git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
	cmd.Dir = r.repoRoot
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed to get files in commit: %w\n%s", err, out)
	}
	return nonEmptyTrimmedLines(out), nil
}

func nonEmptyTrimmedLines(b []byte) []string {
	var lines []string
	for line := range strings.Lines(string(b)) {
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

// formatRelatedFiles formats the related files list into a human-readable message
func (r *CodeReviewer) formatRelatedFiles(files []RelatedFile) string {
	if len(files) == 0 {
		return ""
	}

	buf := new(strings.Builder)

	fmt.Fprintf(buf, "Potentially related files:\n\n")

	for _, file := range files {
		fmt.Fprintf(buf, "- %s (%0.0f%%)\n", file.Path, 100*file.Correlation)
	}

	fmt.Fprintf(buf, "\nThese files have historically changed with the files you have modified. Consider whether they require updates as well.\n")
	return buf.String()
}
