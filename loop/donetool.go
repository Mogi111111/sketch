package loop

import (
	"context"
	"encoding/json"
	"fmt"

	"sketch.dev/ant"
	"sketch.dev/claudetool"
)

// makeDoneTool creates a tool that provides a checklist to the agent. There
// are some duplicative instructions here and in the system prompt, and it's
// not as reliable as it could be. Historically, we've found that Claude ignores
// the tool results here, so we don't tell the tool to say "hey, really check this"
// at the moment, though we've tried.
func makeDoneTool(codereview *claudetool.CodeReviewer, gitUsername, gitEmail string) *ant.Tool {
	return &ant.Tool{
		Name:        "done",
		Description: `Use this tool when you have achieved the user's goal. The parameters form a checklist which you should evaluate.`,
		InputSchema: json.RawMessage(doneChecklistJSONSchema(gitUsername, gitEmail)),
		Run: func(ctx context.Context, input json.RawMessage) (string, error) {
			// Cannot be done with a messy git.
			if err := codereview.RequireNormalGitState(ctx); err != nil {
				return "", err
			}
			if err := codereview.RequireNoUncommittedChanges(ctx); err != nil {
				return "", err
			}
			// Ensure that the current commit has been reviewed.
			head, err := codereview.CurrentCommit(ctx)
			if err == nil {
				needsReview := !codereview.IsInitialCommit(head) && !codereview.HasReviewed(head)
				if needsReview {
					return "", fmt.Errorf("codereview tool has not been run for commit %v", head)
				}
			}
			return `Please ask the user to review your work. Be concise - users are more likely to read shorter comments.`, nil
		},
	}
}

func doneChecklistJSONSchema(gitUsername, gitEmail string) string {
	gitCommitDescription := fmt.Sprintf(`Create git commits for any code changes you made. Match the style of recent commit messages. Include 'Co-Authored-By: sketch <hello@sketch.dev>' and the original user prompt. Use GIT_AUTHOR_NAME="%s" GIT_AUTHOR_EMAIL="%s" (not git config).`,
		gitUsername, gitEmail)
	desc, err := json.Marshal(gitCommitDescription)
	if err != nil {
		panic(err)
	}
	return doneChecklistJSONSchemaPrefix + string(desc) + doneChecklistJSONSchemaSuffix
}

// TODO: this is ugly, maybe JSON-encode a deeply nested map[string]any instead? also ugly.
const (
	doneChecklistJSONSchemaPrefix = `{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Checklist",
  "description": "A schema for tracking checklist items with status and comments",
  "type": "object",
  "required": ["checklist_items"],
  "properties": {
    "checklist_items": {
      "type": "object",
      "description": "Collection of checklist items",
      "properties": {
        "wrote_tests": {
          "$ref": "#/definitions/checklistItem",
          "description": "If code was changed, tests were written or updated."
        },
        "passes_tests": {
          "$ref": "#/definitions/checklistItem",
          "description": "If any commits were made, tests pass."
        },
        "code_reviewed": {
          "$ref": "#/definitions/checklistItem",
          "description": "If any commits were made, the codereview tool was run and its output was addressed."
        },
        "git_commit": {
          "$ref": "#/definitions/checklistItem",
          "description": `

	doneChecklistJSONSchemaSuffix = `
		}
      },
      "additionalProperties": {
        "$ref": "#/definitions/checklistItem"
      }
    }
  },
  "definitions": {
    "checklistItem": {
      "type": "object",
      "required": ["status"],
      "properties": {
        "status": {
          "type": "string",
          "description": "Current status of the checklist item",
          "enum": ["yes", "no", "not applicable", "other"]
        },
        "description": {
          "type": "string",
          "description": "Description of what this checklist item verifies"
        },
        "comments": {
          "type": "string",
          "description": "Additional comments or context for this checklist item"
        }
      }
    }
  }
}`
)
