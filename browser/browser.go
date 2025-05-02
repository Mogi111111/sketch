// Package browser provides functions for opening URLs in a web browser.
package browser

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
)

// Open opens the specified URL in the system's default browser.
// It detects the OS and uses the appropriate command:
// - 'open' for macOS
// - 'cmd /c start' for Windows
// - 'xdg-open' for Linux and other Unix-like systems
func Open(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", url)
	default: // Linux and other Unix-like systems
		cmd = exec.Command("xdg-open", url)
	}
	if b, err := cmd.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to open browser: %v: %s\n", err, b)
	}
}
