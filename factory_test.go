package aster_test

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/mgilbir/aster"
)

// factorySpecs returns the bar-chart fixture plus every vl-convert spec that
// renders without a loader or a version-specific known failure — the broadest
// set of inputs that a warm-restored and a cold Converter can both drive
// identically with the default config.
func factorySpecs(t *testing.T) map[string][]byte {
	t.Helper()
	specs := map[string][]byte{}

	bar, err := os.ReadFile("testdata/bar-chart.vl.json")
	if err != nil {
		t.Fatalf("reading bar-chart spec: %v", err)
	}
	specs["bar-chart"] = bar

	// vl-convert specs that need remote/relative data or geo features the
	// default (loader-less, VL 6.4) config cannot render.
	skip := map[string]bool{
		"circle_binned_base_url": true, // needs a base URL loader
		"lookup_urls":            true, // needs a data loader
		"remote_images":          true, // needs a remote image loader
		"maptile_background":     true, // needs a tile loader
		"maptile_background_2":   true, // geoScale + loader
		"seattle-weather":        true, // needs a data loader
		"geoScale":               true, // geoScale unavailable
		"custom_projection":      true, // geo projection data
	}
	paths, err := filepath.Glob("testdata/vl-convert/*.vl.json")
	if err != nil {
		t.Fatalf("globbing vl-convert specs: %v", err)
	}
	for _, p := range paths {
		name := strings.TrimSuffix(filepath.Base(p), ".vl.json")
		if skip[name] {
			continue
		}
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("reading %s: %v", p, err)
		}
		specs[name] = b
	}
	return specs
}

// renderAll renders every spec through c and returns name -> {svg, vega, png}.
type rendered struct {
	svg  string
	vega []byte
	png  []byte
}

func renderAll(t *testing.T, c *aster.Converter, specs map[string][]byte) map[string]rendered {
	t.Helper()
	out := make(map[string]rendered, len(specs))
	for name, spec := range specs {
		svg, err := c.VegaLiteToSVG(spec)
		if err != nil {
			t.Fatalf("%s: VegaLiteToSVG: %v", name, err)
		}
		vega, err := c.VegaLiteToVega(spec)
		if err != nil {
			t.Fatalf("%s: VegaLiteToVega: %v", name, err)
		}
		png, err := c.VegaLiteToPNG(spec)
		if err != nil {
			t.Fatalf("%s: VegaLiteToPNG: %v", name, err)
		}
		out[name] = rendered{svg: svg, vega: vega, png: png}
	}
	return out
}

// TestFactoryByteIdenticalOutput is the acceptance bar: a Converter restored
// from a warm-start snapshot must produce EXACTLY the same SVG, Vega, and PNG
// bytes as a cold Converter, for every spec. The Factory's first Converter
// warms cold; a second and third are restored, so this also proves the restore
// path (not just the warm one) matches.
func TestFactoryByteIdenticalOutput(t *testing.T) {
	specs := factorySpecs(t)

	cold, err := aster.New()
	if err != nil {
		t.Fatalf("cold New: %v", err)
	}
	defer func() { _ = cold.Close() }()
	want := renderAll(t, cold, specs)

	f := aster.NewFactory()

	// First Factory Converter warms cold and captures the snapshot.
	warm, err := f.New()
	if err != nil {
		t.Fatalf("factory warm New: %v", err)
	}
	defer func() { _ = warm.Close() }()
	assertRendersEqual(t, "warm", want, renderAll(t, warm, specs))

	// Every later Converter is restored from the snapshot.
	for i := 0; i < 3; i++ {
		restored, err := f.New()
		if err != nil {
			t.Fatalf("factory restore New #%d: %v", i, err)
		}
		assertRendersEqual(t, "restore", want, renderAll(t, restored, specs))
		_ = restored.Close()
	}
}

func assertRendersEqual(t *testing.T, label string, want, got map[string]rendered) {
	t.Helper()
	for name, w := range want {
		g, ok := got[name]
		if !ok {
			t.Errorf("%s: missing render for %s", label, name)
			continue
		}
		if g.svg != w.svg {
			t.Errorf("%s/%s: SVG differs from cold (%d vs %d bytes)", label, name, len(g.svg), len(w.svg))
		}
		if !bytes.Equal(g.vega, w.vega) {
			t.Errorf("%s/%s: Vega differs from cold (%d vs %d bytes)", label, name, len(g.vega), len(w.vega))
		}
		if !bytes.Equal(g.png, w.png) {
			t.Errorf("%s/%s: PNG differs from cold (%d vs %d bytes)", label, name, len(g.png), len(w.png))
		}
	}
}

// TestFactoryConcurrentIsolation runs many restored Converters concurrently,
// each rendering a different spec repeatedly, and asserts every output matches
// the single-threaded cold reference. This proves restored instances do not
// interfere: distinct outputs, no cross-contamination between the private
// per-instance memory images.
func TestFactoryConcurrentIsolation(t *testing.T) {
	specs := factorySpecs(t)

	cold, err := aster.New()
	if err != nil {
		t.Fatalf("cold New: %v", err)
	}
	want := renderAll(t, cold, specs)
	_ = cold.Close()

	f := aster.NewFactory()
	// Prime the snapshot so all goroutines take the restore path.
	primer, err := f.New()
	if err != nil {
		t.Fatalf("primer New: %v", err)
	}
	_ = primer.Close()

	var wg sync.WaitGroup
	errCh := make(chan string, 256)
	for name, spec := range specs {
		wg.Add(1)
		go func(name string, spec []byte) {
			defer wg.Done()
			c, err := f.New()
			if err != nil {
				errCh <- name + ": restore New: " + err.Error()
				return
			}
			defer func() { _ = c.Close() }()
			for i := 0; i < 5; i++ {
				svg, err := c.VegaLiteToSVG(spec)
				if err != nil {
					errCh <- name + ": VegaLiteToSVG: " + err.Error()
					return
				}
				if svg != want[name].svg {
					errCh <- name + ": SVG differs from cold under concurrency"
					return
				}
			}
		}(name, spec)
	}
	wg.Wait()
	close(errCh)
	for msg := range errCh {
		t.Error(msg)
	}
}

// TestFactoryVersionKeying confirms that Factories with different Vega-Lite
// versions keep separate snapshots and each produce output matching a cold
// Converter of the same version — a snapshot from one version is never restored
// into another.
func TestFactoryVersionKeying(t *testing.T) {
	spec, err := os.ReadFile("testdata/bar-chart.vl.json")
	if err != nil {
		t.Fatalf("reading spec: %v", err)
	}

	for _, ver := range []string{"5.8", "6.4"} {
		cold, err := aster.New(aster.WithVegaLiteVersion(ver))
		if err != nil {
			t.Fatalf("cold New %s: %v", ver, err)
		}
		wantSVG, err := cold.VegaLiteToSVG(spec)
		if err != nil {
			t.Fatalf("cold render %s: %v", ver, err)
		}
		_ = cold.Close()

		f := aster.NewFactory(aster.WithVegaLiteVersion(ver))
		if _, err := f.New(); err != nil { // warm
			t.Fatalf("factory warm %s: %v", ver, err)
		}
		restored, err := f.New() // restore
		if err != nil {
			t.Fatalf("factory restore %s: %v", ver, err)
		}
		gotSVG, err := restored.VegaLiteToSVG(spec)
		if err != nil {
			t.Fatalf("restore render %s: %v", ver, err)
		}
		_ = restored.Close()

		if gotSVG != wantSVG {
			t.Errorf("version %s: restored SVG differs from cold (%d vs %d bytes)", ver, len(gotSVG), len(wantSVG))
		}
	}
}

// BenchmarkFactoryNew measures restore cost: the snapshot is warmed once
// outside the loop, so each iteration pays only the memory copy + export
// re-resolution. Compare against BenchmarkNew (cold) to see the warm-start win.
func BenchmarkFactoryNew(b *testing.B) {
	f := aster.NewFactory()
	// Warm + capture the snapshot before the timed loop.
	first, err := f.New()
	if err != nil {
		b.Fatalf("factory warm New: %v", err)
	}
	_ = first.Close()

	b.ReportAllocs()
	for b.Loop() {
		c, err := f.New()
		if err != nil {
			b.Fatalf("factory New: %v", err)
		}
		if err := c.Close(); err != nil {
			b.Fatalf("Close: %v", err)
		}
	}
}
