package webui

import (
	"bytes"
	"fmt"
	"io/fs"
	"time"
)

// memFS implements fs.FS in-memory.
type memFS struct {
	m map[string][]byte
}

func (m memFS) Open(name string) (fs.File, error) {
	b, found := m.m[name]
	if !found {
		return nil, fmt.Errorf("esbuild.memFS(%q): %w", name, fs.ErrNotExist)
	}
	return &memFile{name: name, Reader: *bytes.NewReader(b)}, nil
}

func (m memFS) ReadFile(name string) ([]byte, error) {
	b, found := m.m[name]
	if !found {
		return nil, fmt.Errorf("esbuild.memFS.ReadFile(%q): %w", name, fs.ErrNotExist)
	}
	return append(make([]byte, 0, len(b)), b...), nil
}

// memFile implements fs.File in-memory.
type memFile struct {
	// embedding is very important here because need more than
	// Read, we need Seek to make http.ServeContent happy.
	bytes.Reader
	name string
}

func (f *memFile) Stat() (fs.FileInfo, error) { return &memFileInfo{f: f}, nil }
func (f *memFile) Close() error               { return nil }

var start = time.Now()

type memFileInfo struct {
	f *memFile
}

func (i memFileInfo) Name() string       { return i.f.name }
func (i memFileInfo) Size() int64        { return i.f.Reader.Size() }
func (i memFileInfo) Mode() fs.FileMode  { return 0o444 }
func (i memFileInfo) ModTime() time.Time { return start }
func (i memFileInfo) IsDir() bool        { return false }
func (i memFileInfo) Sys() any           { return nil }
