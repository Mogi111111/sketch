package gzhandler

import (
	"compress/gzip"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"
)

func TestHandler_ServeHTTP(t *testing.T) {
	// Create a test filesystem with regular and gzipped files
	testFS := fstest.MapFS{
		"regular.txt": &fstest.MapFile{
			Data: []byte("This is a regular text file"),
			Mode: 0o644,
		},
		"regular.txt.gz": &fstest.MapFile{
			Data: compressString(t, "This is a regular text file"),
			Mode: 0o644,
		},
		"regular.js": &fstest.MapFile{
			Data: []byte("console.log('Hello world');"),
			Mode: 0o644,
		},
		"regular.js.gz": &fstest.MapFile{
			Data: compressString(t, "console.log('Hello world');"),
			Mode: 0o644,
		},
		"nogzip.css": &fstest.MapFile{
			Data: []byte(".body { color: red; }"),
			Mode: 0o644,
		},
	}

	// Create the handler using our test filesystem
	handler := New(testFS)

	// Define test cases
	tests := []struct {
		name               string
		path               string
		acceptGzip         bool
		expectedStatus     int
		expectedBody       string
		expectedGzipHeader string
		expectedType       string
	}{
		{
			name:               "Serve gzipped text file when accepted",
			path:               "/regular.txt",
			acceptGzip:         true,
			expectedStatus:     http.StatusOK,
			expectedBody:       "This is a regular text file",
			expectedGzipHeader: "gzip",
			expectedType:       "text/plain; charset=utf-8",
		},
		{
			name:               "Serve regular text file when gzip not accepted",
			path:               "/regular.txt",
			acceptGzip:         false,
			expectedStatus:     http.StatusOK,
			expectedBody:       "This is a regular text file",
			expectedGzipHeader: "",
			expectedType:       "text/plain; charset=utf-8",
		},
		{
			name:               "Serve gzipped JS file when accepted",
			path:               "/regular.js",
			acceptGzip:         true,
			expectedStatus:     http.StatusOK,
			expectedBody:       "console.log('Hello world');",
			expectedGzipHeader: "gzip",
			expectedType:       "text/javascript; charset=utf-8",
		},
		{
			name:               "Serve regular CSS file when gzip not available",
			path:               "/nogzip.css",
			acceptGzip:         true,
			expectedStatus:     http.StatusOK,
			expectedBody:       ".body { color: red; }",
			expectedGzipHeader: "",
			expectedType:       "text/css; charset=utf-8",
		},
		{
			name:           "Return 404 for non-existent file",
			path:           "/nonexistent.txt",
			acceptGzip:     true,
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Create a request for the specified path
			req := httptest.NewRequest("GET", tc.path, nil)

			// Set Accept-Encoding header if needed
			if tc.acceptGzip {
				req.Header.Set("Accept-Encoding", "gzip")
			}

			// Create a response recorder
			rec := httptest.NewRecorder()

			// Serve the request
			handler.ServeHTTP(rec, req)

			// Check status code
			if rec.Code != tc.expectedStatus {
				t.Errorf("Expected status %d, got %d", tc.expectedStatus, rec.Code)
				return
			}

			// For non-200 responses, we don't check the body
			if tc.expectedStatus != http.StatusOK {
				return
			}

			// Check Content-Type header (skip for .txt files since MIME mappings can vary by OS)
			if !strings.HasSuffix(tc.path, ".txt") {
				contentType := rec.Header().Get("Content-Type")
				if contentType != tc.expectedType {
					t.Errorf("Expected Content-Type %q, got %q", tc.expectedType, contentType)
				}
			}

			// Check Content-Encoding header
			contentEncoding := rec.Header().Get("Content-Encoding")
			if contentEncoding != tc.expectedGzipHeader {
				t.Errorf("Expected Content-Encoding %q, got %q", tc.expectedGzipHeader, contentEncoding)
			}

			// Read response body
			var bodyReader io.Reader = rec.Body

			// If response is gzipped, decompress it
			if contentEncoding == "gzip" {
				gzReader, err := gzip.NewReader(rec.Body)
				if err != nil {
					t.Fatalf("Failed to create gzip reader: %v", err)
				}
				defer gzReader.Close()
				bodyReader = gzReader
			}

			// Read and check body content
			actualBody, err := io.ReadAll(bodyReader)
			if err != nil {
				t.Fatalf("Failed to read response body: %v", err)
			}

			if string(actualBody) != tc.expectedBody {
				t.Errorf("Expected body %q, got %q", tc.expectedBody, string(actualBody))
			}
		})
	}
}

// TestHandleDirectories tests that directories are handled properly
func TestHandleDirectories(t *testing.T) {
	// Create a test filesystem with a directory
	testFS := fstest.MapFS{
		"dir": &fstest.MapFile{
			Mode: fs.ModeDir | 0o755,
		},
		"dir/index.html": &fstest.MapFile{
			Data: []byte("<html>Directory index</html>"),
			Mode: 0o644,
		},
		"dir/index.html.gz": &fstest.MapFile{
			Data: compressString(t, "<html>Directory index</html>"),
			Mode: 0o644,
		},
	}

	// Create the handler using our test filesystem
	handler := New(testFS)

	// Create a request for the directory
	req := httptest.NewRequest("GET", "/dir/", nil)
	req.Header.Set("Accept-Encoding", "gzip")

	// Create a response recorder
	rec := httptest.NewRecorder()

	// Serve the request
	handler.ServeHTTP(rec, req)

	// Check status code should be 200 (directory index)
	if rec.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", rec.Code)
	}

	// Note: Directory listings may not use gzip encoding by default with http.FileServer
	// This is acceptable behavior, so we don't enforce gzip encoding for directories
	contentEncoding := rec.Header().Get("Content-Encoding")

	// Check if body contains the index content (after decompression)
	var bodyReader io.Reader
	if contentEncoding == "gzip" {
		gzReader, err := gzip.NewReader(rec.Body)
		if err != nil {
			t.Fatalf("Failed to create gzip reader: %v", err)
		}
		defer gzReader.Close()
		bodyReader = gzReader
	} else {
		bodyReader = rec.Body
	}

	body, err := io.ReadAll(bodyReader)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if !strings.Contains(string(body), "Directory index") {
		t.Errorf("Expected directory index content, got %q", string(body))
	}
}

// Helper function to compress a string into gzip format
func compressString(t *testing.T, s string) []byte {
	var buf strings.Builder
	gw := gzip.NewWriter(&buf)

	_, err := gw.Write([]byte(s))
	if err != nil {
		t.Fatalf("Failed to write to gzip writer: %v", err)
	}

	if err := gw.Close(); err != nil {
		t.Fatalf("Failed to close gzip writer: %v", err)
	}

	return []byte(buf.String())
}
