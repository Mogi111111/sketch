// Package gzhandler provides an HTTP file server implementation that serves pre-compressed files
// when available to clients that support gzip encoding.
package gzhandler

import (
	"io"
	"io/fs"
	"mime"
	"net/http"
	"path"
	"strings"
)

// Handler is an http.Handler that checks for pre-compressed files
// and serves them with appropriate headers when available.
type Handler struct {
	root http.FileSystem
}

// New creates a handler that serves HTTP requests
// with the contents of the file system rooted at root and uses pre-compressed
// .gz files when available, with appropriate headers.
func New(root fs.FS) http.Handler {
	return &Handler{root: http.FS(root)}
}

// ServeHTTP serves a file with special handling for pre-compressed .gz files
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Clean and prepare the URL path
	urlPath := r.URL.Path
	if !strings.HasPrefix(urlPath, "/") {
		urlPath = "/" + urlPath
	}
	urlPath = path.Clean(urlPath)

	// Check if client accepts gzip encoding
	acceptEncoding := r.Header.Get("Accept-Encoding")
	acceptsGzip := strings.Contains(acceptEncoding, "gzip")

	// Check if the file itself is not a gzip file (we don't want to double-compress)
	isCompressibleFile := !strings.HasSuffix(urlPath, ".gz")

	if acceptsGzip && isCompressibleFile {
		// Try to open the gzipped version of the file
		gzPath := urlPath + ".gz"
		gzFile, err := h.root.Open(gzPath)

		if err == nil {
			defer gzFile.Close()

			// Get file info to check if it's a regular file
			gzStat, err := gzFile.Stat()
			if err != nil || gzStat.IsDir() {
				// Not a valid file, fall back to normal serving
				http.FileServer(h.root).ServeHTTP(w, r)
				return
			}

			// Determine the content type based on the original file (not the .gz)
			contentType := mime.TypeByExtension(path.Ext(urlPath))
			if contentType == "" {
				contentType = "application/octet-stream"
			}

			// Set the appropriate headers for serving gzipped content
			w.Header().Set("Content-Type", contentType)
			w.Header().Set("Content-Encoding", "gzip")
			w.Header().Set("Vary", "Accept-Encoding")

			// Read the gzipped file into memory to avoid 'seeker can't seek' error
			gzippedData, err := io.ReadAll(gzFile)
			if err != nil {
				http.Error(w, "Error reading gzipped content", http.StatusInternalServerError)
				return
			}

			// Write the headers and gzipped content
			w.WriteHeader(http.StatusOK)
			w.Write(gzippedData)
			return
		}
	}

	// Fall back to standard file serving if gzipped version not found or not applicable
	http.FileServer(h.root).ServeHTTP(w, r)
}
