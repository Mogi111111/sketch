// Package skribe defines sketch-wide logging types and functions.
//
// Logging happens via slog.
package skribe

import (
	"context"
	"io"
	"log/slog"
	"slices"
	"strings"
)

type attrsKey struct{}

func Redact(arr []string) []string {
	ret := []string{}
	for _, s := range arr {
		if strings.HasPrefix(s, "ANTHROPIC_API_KEY=") {
			ret = append(ret, "ANTHROPIC_API_KEY=[REDACTED]")
		} else {
			ret = append(ret, s)
		}
	}
	return ret
}

func ContextWithAttr(ctx context.Context, add ...slog.Attr) context.Context {
	attrs := slices.Clone(Attrs(ctx))
	attrs = append(attrs, add...)
	return context.WithValue(ctx, attrsKey{}, attrs)
}

func Attrs(ctx context.Context) []slog.Attr {
	attrs, _ := ctx.Value(attrsKey{}).([]slog.Attr)
	return attrs
}

func AttrsWrap(h slog.Handler) slog.Handler {
	return &augmentHandler{Handler: h}
}

type augmentHandler struct {
	slog.Handler
}

func (h *augmentHandler) Handle(ctx context.Context, r slog.Record) error {
	attrs := Attrs(ctx)
	r.AddAttrs(attrs...)
	return h.Handler.Handle(ctx, r)
}

type multiHandler struct {
	AllHandler slog.Handler
}

// Enabled implements slog.Handler. Ignores slog.Level - if there's a logger, this returns true.
func (mh *multiHandler) Enabled(ctx context.Context, l slog.Level) bool {
	_, ok := ctx.Value(skribeCtxHandlerKey).(slog.Handler)
	return ok
}

// WithAttrs implements slog.Handler.
func (mh *multiHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	panic("unimplemented")
}

// WithGroup implements slog.Handler.
func (mh *multiHandler) WithGroup(name string) slog.Handler {
	panic("unimplemented")
}

func NewMultiHandler() *multiHandler {
	return &multiHandler{}
}

type scribeCtxKeyType string

const skribeCtxHandlerKey scribeCtxKeyType = "skribe-handlerKey"

func (mh *multiHandler) NewSlogHandlerCtx(ctx context.Context, logFile io.Writer) context.Context {
	h := slog.NewJSONHandler(logFile, &slog.HandlerOptions{Level: slog.LevelDebug})
	w := AttrsWrap(h)
	return context.WithValue(ctx, skribeCtxHandlerKey, w)
}

func (mh *multiHandler) Handle(ctx context.Context, r slog.Record) error {
	if mh.AllHandler != nil {
		if err := mh.AllHandler.Handle(ctx, r); err != nil {
			return err
		}
	}
	attrs := Attrs(ctx)
	r.AddAttrs(attrs...)
	handler, ok := ctx.Value(skribeCtxHandlerKey).(slog.Handler)
	if !ok {
		panic("no skribeCtxHandlerKey value in ctx")
	}
	return handler.Handle(ctx, r)
}
