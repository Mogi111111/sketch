package dockerimg

import (
	"cmp"
	"context"
	"flag"
	"io/fs"
	"net/http"
	"os"
	"strings"
	"testing"
	"testing/fstest"

	gcmp "github.com/google/go-cmp/cmp"
	"sketch.dev/httprr"
)

var flagRewriteWant = flag.Bool("rewritewant", false, "rewrite the dockerfiles we want from the model")

func TestCreateDockerfile(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name string
		fsys fs.FS
	}{
		{
			name: "Basic repo with README",
			fsys: fstest.MapFS{
				"README.md": &fstest.MapFile{Data: []byte("# Test Project\nA Go project for testing.")},
			},
		},
		{
			// TODO: this looks bogus.
			name: "Repo with README and workflow",
			fsys: fstest.MapFS{
				"README.md": &fstest.MapFile{Data: []byte("# Test Project\nA Go project for testing.")},
				".github/workflows/test.yml": &fstest.MapFile{Data: []byte(`name: Test
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install and activate corepack
        run: |
          npm install -g corepack
          corepack enable
      - run: go test ./...`)},
			},
		},
		{
			name: "mention a devtool in the readme",
			fsys: fstest.MapFS{
				"readme.md": &fstest.MapFile{Data: []byte("# Test Project\nYou must install `dot` to run the tests.")},
			},
		},
		{
			name: "empty repo",
			fsys: fstest.MapFS{
				"main.go": &fstest.MapFile{Data: []byte("package main\n\nfunc main() {}")},
			},
		},
		{
			name: "python misery",
			fsys: fstest.MapFS{
				"README.md": &fstest.MapFile{Data: []byte("# Our amazing repo\n\nTo use this project you need python 3.11 and the dvc tool")},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			basePath := "testdata/" + strings.ToLower(strings.Replace(t.Name(), "/", "_", -1))
			rrPath := basePath + ".httprr"
			rr, err := httprr.Open(rrPath, http.DefaultTransport)
			if err != nil && !os.IsNotExist(err) {
				t.Fatal(err)
			}
			rr.ScrubReq(func(req *http.Request) error {
				req.Header.Del("x-api-key")
				return nil
			})
			initFiles, err := readInitFiles(tt.fsys)
			if err != nil {
				t.Fatal(err)
			}
			apiKey := cmp.Or(os.Getenv("OUTER_SKETCH_ANTHROPIC_API_KEY"), os.Getenv("ANTHROPIC_API_KEY"))
			result, err := createDockerfile(ctx, rr.Client(), "", apiKey, initFiles, "")
			if err != nil {
				t.Fatal(err)
			}

			wantPath := basePath + ".dockerfile"

			if *flagRewriteWant {
				if err := os.WriteFile(wantPath, []byte(result), 0o666); err != nil {
					t.Fatal(err)
				}
				return
			}

			wantBytes, err := os.ReadFile(wantPath)
			if err != nil {
				t.Fatal(err)
			}
			want := string(wantBytes)
			if diff := gcmp.Diff(want, result); diff != "" {
				t.Errorf("dockerfile does not match. got:\n----\n%s\n----\n\ndiff: %s", result, diff)
			}
		})
	}
}

func TestReadInitFiles(t *testing.T) {
	testFS := fstest.MapFS{
		"README.md":                  &fstest.MapFile{Data: []byte("# Test Repo")},
		".github/workflows/test.yml": &fstest.MapFile{Data: []byte("name: Test Workflow")},
		"main.go":                    &fstest.MapFile{Data: []byte("package main")},
		".git/HEAD":                  &fstest.MapFile{Data: []byte("ref: refs/heads/main")},
		"random/README.md":           &fstest.MapFile{Data: []byte("ignore me")},
	}

	files, err := readInitFiles(testFS)
	if err != nil {
		t.Fatalf("readInitFiles failed: %v", err)
	}

	// Should have 2 files: README.md and .github/workflows/test.yml
	if len(files) != 2 {
		t.Errorf("Expected 2 files, got %d", len(files))
	}

	if content, ok := files["README.md"]; !ok {
		t.Error("README.md not found")
	} else if content != "# Test Repo" {
		t.Errorf("README.md has incorrect content: %q", content)
	}

	if content, ok := files[".github/workflows/test.yml"]; !ok {
		t.Error(".github/workflows/test.yml not found")
	} else if content != "name: Test Workflow" {
		t.Errorf("Workflow file has incorrect content: %q", content)
	}

	if _, ok := files["main.go"]; ok {
		t.Error("main.go should not be included")
	}

	if _, ok := files[".git/HEAD"]; ok {
		t.Error(".git/HEAD should not be included")
	}
}

func TestReadInitFilesWithSubdir(t *testing.T) {
	// Create a file system with files in a subdirectory
	testFS := fstest.MapFS{
		"subdir/README.md":                  &fstest.MapFile{Data: []byte("# Test Repo")},
		"subdir/.github/workflows/test.yml": &fstest.MapFile{Data: []byte("name: Test Workflow")},
		"subdir/main.go":                    &fstest.MapFile{Data: []byte("package main")},
	}

	// Use fs.Sub to get a sub-filesystem
	subFS, err := fs.Sub(testFS, "subdir")
	if err != nil {
		t.Fatalf("fs.Sub failed: %v", err)
	}

	files, err := readInitFiles(subFS)
	if err != nil {
		t.Fatalf("readInitFiles failed: %v", err)
	}

	// Should have 2 files: README.md and .github/workflows/test.yml
	if len(files) != 2 {
		t.Errorf("Expected 2 files, got %d", len(files))
	}

	// Verify README.md was found
	if content, ok := files["README.md"]; !ok {
		t.Error("README.md not found")
	} else if content != "# Test Repo" {
		t.Errorf("README.md has incorrect content: %q", content)
	}

	// Verify workflow file was found
	if content, ok := files[".github/workflows/test.yml"]; !ok {
		t.Error(".github/workflows/test.yml not found")
	} else if content != "name: Test Workflow" {
		t.Errorf("Workflow file has incorrect content: %q", content)
	}
}

// TestDockerHashIsPushed ensures that any changes made to the
// dockerfile template have been pushed to the default image.
func TestDockerHashIsPushed(t *testing.T) {
	name, _, tag := DefaultImage()

	if err := checkTagExists(tag); err != nil {
		if strings.Contains(err.Error(), "not found") {
			t.Fatalf(`Currently released docker image %s does not match dockerfileCustomTmpl.

Inspecting the docker image shows the current hash of dockerfileBase is %s,
but it is not published in the GitHub container registry.

This means the template constants in createdockerfile.go have been
edited (e.g. dockerfileBase changed), but a new version
of the public default docker image has not been built and pushed.

To do so:

	go run ./dockerimg/pushdockerimg.go

`, name, tag)
		} else {
			t.Fatalf("checkTagExists: %v", err)
		}
	}
}
