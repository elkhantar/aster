package aster

import (
	"sync"

	"github.com/mgilbir/aster/internal/runtime"
)

// Factory builds many Converters that share one warm-start snapshot.
//
// The first Converter a Factory builds warms the QuickJS engine cold —
// instantiate the WASM module and evaluate the Vega/Vega-Lite bundle (~168ms) —
// and captures a snapshot of the warmed WASM linear memory. Every later
// Converter is restored from that snapshot: a private per-instance copy of the
// warmed memory plus re-resolution of the module's exports, skipping the ~145ms
// bundle evaluation. It is the recommended way to construct a pool of
// identically-configured Converters, and the first-request latency it removes
// matters even when a pool amortizes construction.
//
// All Converters from one Factory share its options, hence one snapshot. The
// warmed heap depends on the Vega-Lite version and timezone, so for a different
// version or timezone use a separate Factory; a restore under a mismatched key
// is refused rather than silently wrong. Options that do not affect the warmed
// heap (loader, fonts, theme, memory limit, timeout) may be varied only by
// using distinct Factories — a single Factory applies its fixed options to
// every Converter.
//
// Converters built by a Factory are independent and isolated: each has its own
// private linear memory copied from the immutable snapshot, so one Converter
// can never observe or corrupt another's state. A Factory is safe for
// concurrent use; its Converters are not (like any Converter, use one per
// goroutine or pool them).
type Factory struct {
	cfg *config

	mu   sync.Mutex
	snap *runtime.Snapshot
}

// NewFactory creates a Factory whose Converters are configured with opts.
func NewFactory(opts ...Option) *Factory {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(cfg)
	}
	return &Factory{cfg: cfg}
}

// New builds a Converter. The first successful call warms the engine cold and
// captures a snapshot; every later call restores it. The returned Converter is
// indistinguishable from one built by New with the Factory's options.
func (f *Factory) New() (*Converter, error) {
	// Both pipelines are configured from a single font plan (see New).
	plan := newFontPlan(f.cfg)
	measurer, tm, err := buildMeasurer(f.cfg, plan)
	if err != nil {
		return nil, err
	}
	rtCfg := runtimeConfig(f.cfg, tm)

	f.mu.Lock()
	if f.snap == nil {
		// First Converter: warm cold, capture the snapshot, and hand back the
		// warm runtime itself (its heap is exactly what later restores clone).
		// The lock serializes only the one-time warm; concurrent first-callers
		// wait rather than each warming their own.
		rt, err := runtime.NewWarm(rtCfg)
		if err != nil {
			f.mu.Unlock()
			return nil, err // runtime errors are already namespaced.
		}
		snap, err := rt.Snapshot()
		if err != nil {
			// Snapshotting failed: return a working (cold) Converter and leave
			// the snapshot uncached so a later call retries the warm. The
			// feature degrades to plain cold construction, never to incorrect
			// output.
			f.mu.Unlock()
			return assembleConverter(rt, plan, measurer, f.cfg.loader), nil
		}
		f.snap = snap
		f.mu.Unlock()
		return assembleConverter(rt, plan, measurer, f.cfg.loader), nil
	}
	snap := f.snap
	f.mu.Unlock()

	rt, err := runtime.Restore(rtCfg, snap)
	if err != nil {
		return nil, err // runtime errors are already namespaced.
	}
	return assembleConverter(rt, plan, measurer, f.cfg.loader), nil
}
