// Package aster converts Vega and Vega-Lite visualization specs to SVG and PNG.
// It embeds Vega/Vega-Lite inside QuickJS (via WASM) for a pure-Go,
// CGO-free solution.
//
// Basic usage:
//
//	c, err := aster.New()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer c.Close()
//
//	svg, err := c.VegaLiteToSVG(specJSON)
//	png, err := c.VegaLiteToPNG(specJSON)
package aster

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"

	"github.com/mgilbir/aster/internal/resvg"
	"github.com/mgilbir/aster/internal/runtime"
	"github.com/mgilbir/aster/internal/textmeasure"
)

// Converter renders Vega/Vega-Lite specs to SVG and PNG.
type Converter struct {
	rt       *runtime.Runtime
	measurer *textmeasure.Measurer
	fonts    fontPlan // shared by text measurement and PNG rasterization
	loader   Loader   // stashed for Close()
	closed   bool     // set by Close; every entry point checks it

	pngOnce     sync.Once
	pngRenderer *resvg.Renderer
	pngErr      error
}

// errConverterClosed is returned by every rendering method after Close.
var errConverterClosed = errors.New("aster: converter is closed")

// New creates a new Converter with the given options.
//
// New warms the QuickJS engine cold: it evaluates the vendored Vega/Vega-Lite
// bundle (~145ms) on every call. To build many identically-configured
// Converters (e.g. a converter pool) at a fraction of that cost, use a Factory,
// which warms once and restores a memory snapshot for every later Converter.
func New(opts ...Option) (*Converter, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(cfg)
	}

	// The QuickJS WASM runtime has no timezone database; only UTC (via a Date
	// polyfill) is implemented. Failing here beats silently rendering with an
	// unexpected timezone.
	if cfg.timezone != "" && cfg.timezone != "UTC" {
		return nil, fmt.Errorf("aster: unsupported timezone %q (only \"UTC\" is supported)", cfg.timezone)
	}

	// Both pipelines are configured from a single font plan so SVG layout and
	// PNG rasterization can't disagree about fonts or generic-family mappings.
	plan := newFontPlan(cfg)

	measurer, tm, err := buildMeasurer(cfg, plan)
	if err != nil {
		return nil, err
	}

	rt, err := runtime.New(runtimeConfig(cfg, tm))
	if err != nil {
		// runtime.New already namespaces its errors ("aster/runtime: ...");
		// don't double-prefix.
		return nil, err
	}

	return assembleConverter(rt, plan, measurer, cfg.loader), nil
}

// buildMeasurer builds the text measurer (and its runtime adapter) from the
// font plan, or returns nils when text measurement is disabled. A fresh
// measurer is built per Converter so Converters never share mutable
// measurement state.
func buildMeasurer(cfg *config, plan fontPlan) (*textmeasure.Measurer, runtime.TextMeasurer, error) {
	if !cfg.textMeasure {
		return nil, nil, nil
	}
	measurer, err := textmeasure.New(plan.measurerOptions()...)
	if err != nil {
		return nil, nil, fmt.Errorf("aster: initializing text measurer: %w", err)
	}
	return measurer, measurer, nil
}

// runtimeConfig maps the Converter config to the runtime config.
func runtimeConfig(cfg *config, tm runtime.TextMeasurer) runtime.Config {
	return runtime.Config{
		Loader:       cfg.loader,
		TextMeasurer: tm,
		Theme:        cfg.theme,
		MemoryLimit:  cfg.memoryLimit,
		Timeout:      cfg.timeout,
		Version:      cfg.vegaLiteVersion,
		Timezone:     cfg.timezone,
	}
}

// assembleConverter wires a ready runtime, font plan, and measurer into a
// Converter.
func assembleConverter(rt *runtime.Runtime, plan fontPlan, measurer *textmeasure.Measurer, loader Loader) *Converter {
	return &Converter{
		rt:       rt,
		measurer: measurer,
		fonts:    plan,
		loader:   loader,
	}
}

// VersionInfo describes an available Vega-Lite version set.
type VersionInfo struct {
	Key             string // internal key accepted by the runtime, e.g. "vl6_4"
	VegaVersion     string // resolved Vega runtime version, e.g. "6.2.0"
	VegaLiteVersion string // Vega-Lite version, e.g. "6.4.0"
}

// AvailableVersions reports the Vega-Lite version sets bundled in this build,
// sorted by key. Pass a VegaLiteVersion (e.g. "6.4") to WithVegaLiteVersion.
func AvailableVersions() ([]VersionInfo, error) {
	m, err := runtime.AvailableVersions()
	if err != nil {
		return nil, err
	}
	out := make([]VersionInfo, 0, len(m))
	for k, v := range m {
		out = append(out, VersionInfo{Key: k, VegaVersion: v.VegaVersion, VegaLiteVersion: v.VegaLiteVersion})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out, nil
}

// Close releases all resources held by the Converter. It is safe to call
// multiple times; after Close every rendering method returns an error.
func (c *Converter) Close() error {
	if c.closed {
		return nil
	}
	c.closed = true
	var firstErr error
	if c.pngRenderer != nil {
		if err := c.pngRenderer.Close(context.Background()); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if c.rt != nil {
		if err := c.rt.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if closer, ok := c.loader.(io.Closer); ok {
		if err := closer.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// VegaToSVG renders a Vega spec (JSON) to an SVG string.
func (c *Converter) VegaToSVG(spec []byte) (string, error) {
	if c.closed {
		return "", errConverterClosed
	}
	return c.rt.VegaToSVG(string(spec))
}

// VegaLiteToSVG renders a Vega-Lite spec (JSON) to an SVG string.
func (c *Converter) VegaLiteToSVG(spec []byte) (string, error) {
	if c.closed {
		return "", errConverterClosed
	}
	return c.rt.VegaLiteToSVG(string(spec))
}

// VegaLiteToVega compiles a Vega-Lite spec (JSON) to a full Vega spec (JSON).
func (c *Converter) VegaLiteToVega(spec []byte) ([]byte, error) {
	if c.closed {
		return nil, errConverterClosed
	}
	result, err := c.rt.VegaLiteToVega(string(spec))
	if err != nil {
		return nil, err
	}
	return []byte(result), nil
}

// VegaToPNG renders a Vega spec (JSON) to a PNG image.
func (c *Converter) VegaToPNG(spec []byte, opts ...PNGOption) ([]byte, error) {
	svg, err := c.VegaToSVG(spec)
	if err != nil {
		return nil, err
	}
	return c.SVGToPNG(svg, opts...)
}

// VegaLiteToPNG renders a Vega-Lite spec (JSON) to a PNG image.
func (c *Converter) VegaLiteToPNG(spec []byte, opts ...PNGOption) ([]byte, error) {
	svg, err := c.VegaLiteToSVG(spec)
	if err != nil {
		return nil, err
	}
	return c.SVGToPNG(svg, opts...)
}

// SVGToPNG converts an SVG string to a PNG image using resvg.
func (c *Converter) SVGToPNG(svg string, opts ...PNGOption) ([]byte, error) {
	if c.closed {
		// Without this guard a post-Close call would lazily instantiate a
		// fresh PNG renderer that nothing would ever release.
		return nil, errConverterClosed
	}
	cfg := defaultPNGConfig()
	for _, opt := range opts {
		opt(cfg)
	}

	if !(cfg.scale > 0) || math.IsInf(cfg.scale, 1) {
		return nil, fmt.Errorf("aster: invalid PNG scale %v (must be a positive, finite number)", cfg.scale)
	}

	r, err := c.pngRendererInit()
	if err != nil {
		return nil, err
	}

	out, err := r.Render(context.Background(), []byte(svg), cfg.scale)
	if err != nil {
		return nil, err
	}
	switch {
	case cfg.quantizeColors > 0:
		out = quantizeOrRecodePNG(out, cfg.quantizeColors)
	case cfg.recode:
		out = recodePNG(out)
	}
	return out, nil
}

// pngRendererInit lazily initializes the PNG renderer on first use. It draws
// its fonts and generic-family mapping from the same fontPlan that configured
// text measurement, so PNG glyphs are rasterized with the font the SVG layout
// was measured against.
func (c *Converter) pngRendererInit() (*resvg.Renderer, error) {
	c.pngOnce.Do(func() {
		fonts, families := c.fonts.resvgFonts()
		c.pngRenderer, c.pngErr = resvg.New(context.Background(), fonts, families)
		if c.pngErr != nil {
			c.pngErr = fmt.Errorf("aster: initializing PNG renderer: %w", c.pngErr)
		}
	})
	return c.pngRenderer, c.pngErr
}
