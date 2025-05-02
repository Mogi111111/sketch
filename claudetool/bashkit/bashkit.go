package bashkit

import (
	"fmt"
	"strings"

	"mvdan.cc/sh/v3/syntax"
)

var checks = []func(*syntax.CallExpr) error{
	noGitConfigUsernameEmailChanges,
}

// Check inspects bashScript and returns an error if it ought not be executed.
// Check DOES NOT PROVIDE SECURITY against malicious actors.
// It is intended to catch straightforward mistakes in which a model
// does things despite having been instructed not to do them.
func Check(bashScript string) error {
	r := strings.NewReader(bashScript)
	parser := syntax.NewParser()
	file, err := parser.Parse(r, "")
	if err != nil {
		// Execution will fail, but we'll get a better error message from bash.
		// Note that if this were security load bearing, this would be a terrible idea:
		// You could smuggle stuff past Check by exploiting differences in what is considered syntactically valid.
		// But it is not.
		return nil
	}

	syntax.Walk(file, func(node syntax.Node) bool {
		if err != nil {
			return false
		}
		callExpr, ok := node.(*syntax.CallExpr)
		if !ok {
			return true
		}
		for _, check := range checks {
			err = check(callExpr)
			if err != nil {
				return false
			}
		}
		return true
	})

	return err
}

// noGitConfigUsernameEmailChanges checks for git config username/email changes.
// It uses simple heuristics, and has both false positives and false negatives.
func noGitConfigUsernameEmailChanges(cmd *syntax.CallExpr) error {
	if hasGitConfigUsernameEmailChanges(cmd) {
		return fmt.Errorf("permission denied: changing git config username/email is not allowed, use env vars instead")
	}
	return nil
}

func hasGitConfigUsernameEmailChanges(cmd *syntax.CallExpr) bool {
	if len(cmd.Args) < 3 {
		return false
	}
	if cmd.Args[0].Lit() != "git" {
		return false
	}

	configIndex := -1
	for i, arg := range cmd.Args {
		if arg.Lit() == "config" {
			configIndex = i
			break
		}
	}

	if configIndex < 0 || configIndex == len(cmd.Args)-1 {
		return false
	}

	// check for user.name or user.email
	keyIndex := -1
	for i, arg := range cmd.Args {
		if i < configIndex {
			continue
		}
		if arg.Lit() == "user.name" || arg.Lit() == "user.email" {
			keyIndex = i
			break
		}
	}

	if keyIndex < 0 || keyIndex == len(cmd.Args)-1 {
		return false
	}

	// user.name/user.email is followed by a value
	return true
}

// WillRunGitCommit checks if the provided bash script will run 'git commit'.
// It returns true if any command in the script is a git commit command.
func WillRunGitCommit(bashScript string) (bool, error) {
	r := strings.NewReader(bashScript)
	parser := syntax.NewParser()
	file, err := parser.Parse(r, "")
	if err != nil {
		// Parsing failed, but let's not consider this an error for the same reasons as in Check
		return false, nil
	}

	willCommit := false

	syntax.Walk(file, func(node syntax.Node) bool {
		callExpr, ok := node.(*syntax.CallExpr)
		if !ok {
			return true
		}
		if isGitCommitCommand(callExpr) {
			willCommit = true
			return false
		}
		return true
	})

	return willCommit, nil
}

// isGitCommitCommand checks if a command is 'git commit'.
func isGitCommitCommand(cmd *syntax.CallExpr) bool {
	if len(cmd.Args) < 2 {
		return false
	}

	// First argument must be 'git'
	if cmd.Args[0].Lit() != "git" {
		return false
	}

	// Look for 'commit' in any position after 'git'
	for i := 1; i < len(cmd.Args); i++ {
		if cmd.Args[i].Lit() == "commit" {
			return true
		}
	}

	return false
}
