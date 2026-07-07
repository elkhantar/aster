package runtime

import (
	"fmt"

	"github.com/mgilbir/aster/internal/quickjs"
)

// Snapshot is a warmed runtime image plus the configuration key it is valid
// for. It captures the QuickJS heap after polyfills and Vega/Vega-Lite module
// loading, so a Restore skips that ~145ms of JavaScript evaluation.
//
// The warmed heap depends only on the Vega-Lite version set and the timezone
// (the two inputs that mutate the heap during warm-up); the theme is a
// per-render argument and the loader/text-measurer are host-side bridge
// callbacks, none of which are baked into the image. Restore therefore rejects
// a config whose version or timezone differs from the captured one.
type Snapshot struct {
	qjs      *quickjs.Snapshot
	version  string
	timezone string
}

// Bytes reports the captured linear-memory size, for diagnostics.
func (s *Snapshot) Bytes() int { return s.qjs.Bytes() }

// normalizeTimezone applies the same "" -> "UTC" default installPolyfills uses,
// so an empty and an explicit "UTC" timezone key to the same snapshot.
func normalizeTimezone(tz string) string {
	if tz == "" {
		return "UTC"
	}
	return tz
}

// NewWarm creates a Runtime exactly like New but backed by a snapshot-capable
// QuickJS instance, so Snapshot can later capture its warmed heap. Its output
// is identical to New's.
func NewWarm(cfg Config) (*Runtime, error) {
	return newRuntime(cfg, quickjs.NewWarm)
}

// Snapshot captures the warmed heap of a runtime created via NewWarm, keyed to
// the runtime's version and timezone. Call it on a freshly warmed runtime,
// before any render, while the engine is quiescent.
func (r *Runtime) Snapshot() (*Snapshot, error) {
	qs, err := r.rt.Snapshot()
	if err != nil {
		return nil, fmt.Errorf("aster/runtime: snapshot: %w", err)
	}
	return &Snapshot{
		qjs:      qs,
		version:  r.config.Version,
		timezone: normalizeTimezone(r.config.Timezone),
	}, nil
}

// Restore builds a Runtime from a snapshot, skipping polyfill installation and
// module loading. cfg must resolve to the same version and timezone the
// snapshot was captured with; other config (memory limit, timeout, loader, text
// measurer, theme) may differ and is applied to the restored instance.
func Restore(cfg Config, snap *Snapshot) (*Runtime, error) {
	if snap == nil {
		return nil, fmt.Errorf("aster/runtime: nil snapshot")
	}

	cfg, err := resolveVersion(cfg)
	if err != nil {
		return nil, err
	}
	if cfg.Version != snap.version || normalizeTimezone(cfg.Timezone) != snap.timezone {
		return nil, fmt.Errorf("aster/runtime: snapshot config mismatch (snapshot version=%q timezone=%q; requested version=%q timezone=%q)",
			snap.version, snap.timezone, cfg.Version, normalizeTimezone(cfg.Timezone))
	}

	qrt, err := quickjs.Restore(qjsConfig(cfg), snap.qjs)
	if err != nil {
		return nil, fmt.Errorf("aster/runtime: restore: %w", err)
	}
	return &Runtime{rt: qrt, config: cfg}, nil
}
