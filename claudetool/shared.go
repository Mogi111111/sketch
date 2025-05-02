// Package claudetool provides tools for Claude AI models.
//
// When adding, removing, or modifying tools in this package,
// remember to update the tool display template in termui/termui.go
// to ensure proper tool output formatting.
package claudetool

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"
)

type workingDirCtxKeyType string

const workingDirCtxKey workingDirCtxKeyType = "workingDir"

func WithWorkingDir(ctx context.Context, wd string) context.Context {
	return context.WithValue(ctx, workingDirCtxKey, wd)
}

func WorkingDir(ctx context.Context) string {
	// If cmd.Dir is empty, it uses the current working directory,
	// so we can use that as a fallback.
	wd, _ := ctx.Value(workingDirCtxKey).(string)
	return wd
}

// sendTelemetry posts debug data to an internal logging server.
// It is meant for use by people developing sketch and is disabled by default.
// This is a best-effort operation; errors are logged but not returned.
func sendTelemetry(ctx context.Context, typ string, data any) {
	telemetryEndpoint := os.Getenv("SKETCH_TELEMETRY_ENDPOINT")
	if telemetryEndpoint == "" {
		return
	}
	err := doPostTelemetry(ctx, telemetryEndpoint, typ, data)
	if err != nil {
		slog.DebugContext(ctx, "failed to send JSON to server", "type", typ, "error", err)
	}
}

func doPostTelemetry(ctx context.Context, telemetryEndpoint, typ string, data any) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal %#v as JSON: %w", data, err)
	}
	timestamp := time.Now().Unix()
	url := fmt.Sprintf(telemetryEndpoint+"/%s_%d.json", typ, timestamp)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request for %s: %w", typ, err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send %s JSON to server: %w", typ, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("server returned non-success status for %s: %d", typ, resp.StatusCode)
	}
	slog.DebugContext(ctx, "successfully sent JSON to server", "file_type", typ, "url", url)
	return nil
}
