package claudetool

import (
	"context"
	"encoding/json"

	"sketch.dev/ant"
)

// The Think tool provides space to think.
var Think = &ant.Tool{
	Name:        thinkName,
	Description: thinkDescription,
	InputSchema: ant.MustSchema(thinkInputSchema),
	Run:         thinkRun,
}

const (
	thinkName        = "think"
	thinkDescription = `Think out loud, take notes, form plans. Has no external effects.`

	// If you modify this, update the termui template for prettier rendering.
	thinkInputSchema = `
{
  "type": "object",
  "required": ["thoughts"],
  "properties": {
    "thoughts": {
      "type": "string",
      "description": "The thoughts, notes, or plans to record"
    }
  }
}
`
)

func thinkRun(ctx context.Context, m json.RawMessage) (string, error) {
	return "recorded", nil
}
