// Package runtime manages the QuickJS engine lifecycle, loads vendored
// Vega/Vega-Lite modules, and bridges Go callbacks into JavaScript.
package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"sort"
	"strings"
	"time"

	asterjs "github.com/mgilbir/aster/internal/js"
	"github.com/mgilbir/aster/internal/quickjs"
)

// Loader is the subset of the aster.Loader interface needed by the runtime.
type Loader interface {
	Load(ctx context.Context, uri string) ([]byte, error)
	Sanitize(ctx context.Context, uri string) (string, error)
}

// TextMeasurer measures text width given a string and CSS font descriptor.
type TextMeasurer interface {
	MeasureText(text, cssFont string) float64
}

// Config holds runtime configuration.
type Config struct {
	Loader       Loader
	TextMeasurer TextMeasurer
	Theme        string
	MemoryLimit  uint64
	Timeout      time.Duration
	Version      string // version set key, e.g. "vl6_4" (default)
	Timezone     string // IANA timezone name or "UTC" (default: "UTC")
}

// Runtime wraps a QuickJS engine with Vega/Vega-Lite loaded.
type Runtime struct {
	rt      *quickjs.Runtime
	config  Config
	crashed bool // set after a WASM panic; further calls return errors
}

// versionIndex matches the top-level versions.json from the vendoring tool.
type versionIndex struct {
	Default  string                    `json:"default"`
	Versions map[string]versionIndexDef `json:"versions"`
}

type versionIndexDef struct {
	VegaVersion     string `json:"vegaVersion"`
	VegaLiteVersion string `json:"vegaLiteVersion"`
}

// manifest matches the JSON structure from the vendoring tool.
type manifest struct {
	VegaVersion     string           `json:"vegaVersion"`
	VegaLiteVersion string           `json:"vegaLiteVersion"`
	Modules         []manifestModule `json:"modules"`
}

type manifestModule struct {
	Name     string `json:"name"`
	Version  string `json:"version"`
	SHA256   string `json:"sha256"`
	Filename string `json:"filename"`
}

// New creates a new Runtime, loading all vendored JS modules and registering
// Go bridge functions.
func New(cfg Config) (*Runtime, error) {
	return newRuntime(cfg, quickjs.New)
}

// qjsConfig maps the runtime configuration to the QuickJS engine config.
func qjsConfig(cfg Config) quickjs.Config {
	return quickjs.Config{
		MemoryLimit: cfg.MemoryLimit,
		Timeout:     cfg.Timeout,
		Bridge:      bridgeConfig(cfg),
	}
}

// resolveVersion fills in the default version and validates it against the
// available version sets, returning the resolved cfg.
func resolveVersion(cfg Config) (Config, error) {
	idx, err := readVersionIndex()
	if err != nil {
		return cfg, err
	}
	if cfg.Version == "" {
		cfg.Version = idx.Default
	} else if _, ok := idx.Versions[cfg.Version]; !ok {
		return cfg, fmt.Errorf("aster/runtime: unknown Vega-Lite version set %q; available: %s", cfg.Version, idx.describe())
	}
	return cfg, nil
}

// newRuntime builds a Runtime using mk to construct the underlying QuickJS
// engine (quickjs.New for a cold runtime, quickjs.NewWarm for a
// snapshot-capable one), then installs polyfills and loads the vendored
// Vega/Vega-Lite modules.
func newRuntime(cfg Config, mk func(quickjs.Config) (*quickjs.Runtime, error)) (*Runtime, error) {
	rt, err := mk(qjsConfig(cfg))
	if err != nil {
		return nil, fmt.Errorf("aster/runtime: creating QuickJS runtime: %w", err)
	}

	cfg, err = resolveVersion(cfg)
	if err != nil {
		_ = rt.Close()
		return nil, err
	}

	r := &Runtime{rt: rt, config: cfg}

	if err := r.installPolyfills(); err != nil {
		_ = rt.Close()
		return nil, err
	}

	if err := r.loadModules(); err != nil {
		_ = rt.Close()
		return nil, err
	}

	return r, nil
}

// Close releases the QuickJS runtime.
// If the WASM runtime has crashed, Close silently skips cleanup to avoid
// secondary panics.
func (r *Runtime) Close() (err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("aster/runtime: panic during close: %v", p)
		}
	}()

	if r.rt != nil && !r.crashed {
		_ = r.rt.Close()
		r.rt = nil
	}
	return nil
}

// bridgeConfig adapts the runtime configuration to the Go callbacks that
// bridge.js reaches through the __aster_* globals. Callbacks execute
// synchronously on the engine thread, mirroring the previous behavior.
func bridgeConfig(cfg Config) quickjs.Bridge {
	// Both loader callbacks get the render timeout as a context deadline.
	// This matters for Sanitize too: HTTPLoader.Sanitize can do live DNS
	// resolution (BlockPrivateNetworks), and the eval watchdog cannot preempt
	// an in-flight host call — without a deadline here, a hung resolver would
	// stall the render indefinitely.
	callCtx := func() (context.Context, context.CancelFunc) {
		if cfg.Timeout > 0 {
			return context.WithTimeout(context.Background(), cfg.Timeout)
		}
		return context.Background(), func() {}
	}
	var b quickjs.Bridge
	if cfg.Loader != nil {
		b.Load = func(url string) ([]byte, error) {
			ctx, cancel := callCtx()
			defer cancel()
			return cfg.Loader.Load(ctx, url)
		}
		b.Sanitize = func(uri string) (string, error) {
			ctx, cancel := callCtx()
			defer cancel()
			return cfg.Loader.Sanitize(ctx, uri)
		}
	}
	if cfg.TextMeasurer != nil {
		b.MeasureText = cfg.TextMeasurer.MeasureText
	}
	return b
}

// installPolyfills adds missing global APIs that QuickJS doesn't provide
// but that Vega/Vega-Lite expect to exist.
func (r *Runtime) installPolyfills() error {
	polyfills := `
		// structuredClone — Vega/Vega-Lite use this for deep cloning specs.
		// A JSON round-trip would drop undefined-valued properties (and turn
		// undefined array holes into null), which breaks specs carrying
		// explicit undefined projection params (e.g. geo_sphere,
		// geo_custom_projection). This recursive clone preserves undefined and
		// handles Date/RegExp/Map/Set/typed arrays and reference cycles.
		if (typeof globalThis.structuredClone === 'undefined') {
			globalThis.structuredClone = function(input) {
				var seen = new Map();
				function clone(v) {
					if (v === null || typeof v !== 'object') return v;
					if (seen.has(v)) return seen.get(v);
					if (v instanceof Date) return new Date(v.getTime());
					if (v instanceof RegExp) return new RegExp(v.source, v.flags);
					if (Array.isArray(v)) {
						var arr = new Array(v.length);
						seen.set(v, arr);
						for (var i = 0; i < v.length; i++) arr[i] = clone(v[i]);
						return arr;
					}
					if (v instanceof Map) {
						var m = new Map();
						seen.set(v, m);
						v.forEach(function(val, key) { m.set(clone(key), clone(val)); });
						return m;
					}
					if (v instanceof Set) {
						var s = new Set();
						seen.set(v, s);
						v.forEach(function(val) { s.add(clone(val)); });
						return s;
					}
					// DataView before the generic isView branch: its
					// constructor takes a buffer, not a view.
					if (v instanceof DataView) {
						return new DataView(v.buffer.slice(0), v.byteOffset, v.byteLength);
					}
					if (ArrayBuffer.isView(v)) return new v.constructor(v);
					if (v instanceof ArrayBuffer) return v.slice(0);
					var out = {};
					seen.set(v, out);
					var keys = Object.keys(v);
					for (var k = 0; k < keys.length; k++) out[keys[k]] = clone(v[keys[k]]);
					return out;
				}
				return clone(input);
			};
		}

		// setTimeout/clearTimeout — d3-timer and vega-scenegraph reference these.
		// Installed unconditionally: the engine's native timers (QuickJS-NG
		// provides them) would queue callbacks that our synchronous render
		// loop never services, leaving render promises pending forever.
		//
		// Callbacks run as microtasks rather than synchronously: patterns
		// like vega-scenegraph's ready(), which polls a counter via
		// setTimeout while the counter is decremented in a promise .then,
		// need timer callbacks to interleave with promise reactions.
		// Endless timer chains are bounded by the Go-side eval watchdog.
		//
		// The delay argument is ignored: callbacks fire in insertion order as
		// microtasks, not ordered by delay. Vega's static render path does not
		// depend on real timer ordering. A throwing callback is swallowed (as
		// in a browser, where one timer's exception does not abort the others)
		// so a single failing timer cannot wedge a render.
		//
		// setInterval is aliased to setTimeout and therefore fires exactly
		// once. A static render has no ongoing time in which to repeat;
		// anything that genuinely needs a repeating timer would hang the
		// synchronous render loop instead.
		{
			const _timers = new Map();
			let _nextId = 1;
			globalThis.setTimeout = function(fn, delay) {
				const id = _nextId++;
				_timers.set(id, true);
				Promise.resolve().then(() => {
					if (!_timers.delete(id)) return; // cleared meanwhile
					try { fn(); } catch(e) {}
				});
				return id;
			};
			globalThis.clearTimeout = function(id) {
				_timers.delete(id);
			};
			globalThis.setInterval = function(fn, delay) {
				return globalThis.setTimeout(fn, delay);
			};
			globalThis.clearInterval = function(id) {
				globalThis.clearTimeout(id);
			};
		}

		// requestAnimationFrame — vega-view may reference this. Same
		// microtask scheduling policy as setTimeout above.
		{
			globalThis.requestAnimationFrame = function(fn) {
				return globalThis.setTimeout(fn, 0);
			};
			globalThis.cancelAnimationFrame = function(id) {
				globalThis.clearTimeout(id);
			};
		}

		// performance.now — some modules may reference this.
		if (typeof globalThis.performance === 'undefined') {
			globalThis.performance = { now: function() { return Date.now(); } };
		}
	`

	if err := r.rt.EvalGlobal("__aster_polyfills__.js", polyfills); err != nil {
		return fmt.Errorf("aster/runtime: installing polyfills: %w", err)
	}

	// Force UTC timezone by redirecting local Date methods to UTC equivalents.
	// QuickJS in WASM has no timezone configuration, so we polyfill it.
	tz := r.config.Timezone
	if tz == "" {
		tz = "UTC"
	}
	if tz == "UTC" {
		utcPolyfill := `
			Date.prototype.getTimezoneOffset = function() { return 0; };
			Date.prototype.getFullYear = Date.prototype.getUTCFullYear;
			Date.prototype.getMonth = Date.prototype.getUTCMonth;
			Date.prototype.getDate = Date.prototype.getUTCDate;
			Date.prototype.getDay = Date.prototype.getUTCDay;
			Date.prototype.getHours = Date.prototype.getUTCHours;
			Date.prototype.getMinutes = Date.prototype.getUTCMinutes;
			Date.prototype.getSeconds = Date.prototype.getUTCSeconds;
			Date.prototype.getMilliseconds = Date.prototype.getUTCMilliseconds;
			Date.prototype.setFullYear = Date.prototype.setUTCFullYear;
			Date.prototype.setMonth = Date.prototype.setUTCMonth;
			Date.prototype.setDate = Date.prototype.setUTCDate;
			Date.prototype.setHours = Date.prototype.setUTCHours;
			Date.prototype.setMinutes = Date.prototype.setUTCMinutes;
			Date.prototype.setSeconds = Date.prototype.setUTCSeconds;
			Date.prototype.setMilliseconds = Date.prototype.setUTCMilliseconds;
		`
		if err := r.rt.EvalGlobal("__aster_tz__.js", utcPolyfill); err != nil {
			return fmt.Errorf("aster/runtime: installing UTC timezone polyfill: %w", err)
		}
	}

	return nil
}

// describe lists the available version sets as "vegaLiteVersion (key)" pairs,
// sorted, for use in error messages.
func (idx *versionIndex) describe() string {
	parts := make([]string, 0, len(idx.Versions))
	for k, v := range idx.Versions {
		parts = append(parts, fmt.Sprintf("%s (%s)", v.VegaLiteVersion, k))
	}
	sort.Strings(parts)
	return strings.Join(parts, ", ")
}

// readVersionIndex reads and parses the versions.json index.
func readVersionIndex() (*versionIndex, error) {
	data, err := fs.ReadFile(asterjs.Modules, "modules/versions.json")
	if err != nil {
		return nil, fmt.Errorf("aster/runtime: reading versions index: %w", err)
	}
	var idx versionIndex
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, fmt.Errorf("aster/runtime: parsing versions index: %w", err)
	}
	return &idx, nil
}

// AvailableVersions returns the available version set keys and their metadata.
func AvailableVersions() (map[string]struct{ VegaVersion, VegaLiteVersion string }, error) {
	idx, err := readVersionIndex()
	if err != nil {
		return nil, err
	}
	result := make(map[string]struct{ VegaVersion, VegaLiteVersion string }, len(idx.Versions))
	for k, v := range idx.Versions {
		result[k] = struct{ VegaVersion, VegaLiteVersion string }{v.VegaVersion, v.VegaLiteVersion}
	}
	return result, nil
}

// loadModules reads the manifest and loads all vendored JS modules in order.
func (r *Runtime) loadModules() error {
	// Read manifest from the versioned subdirectory.
	ver := r.config.Version
	manifestPath := "modules/" + ver + "/manifest.json"
	manifestData, err := fs.ReadFile(asterjs.Modules, manifestPath)
	if err != nil {
		return fmt.Errorf("aster/runtime: reading manifest for %s: %w", ver, err)
	}

	var m manifest
	if err := json.Unmarshal(manifestData, &m); err != nil {
		return fmt.Errorf("aster/runtime: parsing manifest: %w", err)
	}

	// Load each module in topological order.
	for _, mod := range m.Modules {
		src, err := fs.ReadFile(asterjs.Modules, "modules/"+ver+"/"+mod.Filename)
		if err != nil {
			return fmt.Errorf("aster/runtime: reading module %s: %w", mod.Name, err)
		}

		if err := r.rt.LoadModule(mod.Name, string(src)); err != nil {
			return fmt.Errorf("aster/runtime: loading module %s: %w", mod.Name, err)
		}
	}

	// Load the bridge module.
	if err := r.rt.LoadModule("bridge", asterjs.BridgeJS); err != nil {
		return fmt.Errorf("aster/runtime: loading bridge module: %w", err)
	}

	return nil
}

// VegaToSVG renders a Vega spec to SVG.
func (r *Runtime) VegaToSVG(specJSON string) (string, error) {
	spec, err := jsStringLiteral(specJSON)
	if err != nil {
		return "", err
	}
	theme, err := themeLiteral(r.config.Theme)
	if err != nil {
		return "", err
	}

	script := fmt.Sprintf(`
		import { vegaToSvg } from 'bridge';
		export default await vegaToSvg(%s, %s);
	`, spec, theme)

	return r.evalModule(script)
}

// VegaLiteToSVG renders a Vega-Lite spec to SVG.
func (r *Runtime) VegaLiteToSVG(specJSON string) (string, error) {
	spec, err := jsStringLiteral(specJSON)
	if err != nil {
		return "", err
	}
	theme, err := themeLiteral(r.config.Theme)
	if err != nil {
		return "", err
	}

	script := fmt.Sprintf(`
		import { vegaLiteToSvg } from 'bridge';
		export default await vegaLiteToSvg(%s, %s);
	`, spec, theme)

	return r.evalModule(script)
}

// VegaLiteToVega compiles a Vega-Lite spec to a Vega spec. The configured
// theme is passed to the Vega-Lite compiler, so the compiled spec matches
// what the render paths produce (compile-then-VegaToSVG ≡ VegaLiteToSVG).
func (r *Runtime) VegaLiteToVega(specJSON string) (string, error) {
	spec, err := jsStringLiteral(specJSON)
	if err != nil {
		return "", err
	}
	theme, err := themeLiteral(r.config.Theme)
	if err != nil {
		return "", err
	}
	script := fmt.Sprintf(`
		import { vegaLiteToVega } from 'bridge';
		export default vegaLiteToVega(%s, %s);
	`, spec, theme)

	return r.evalModule(script)
}

var (
	errRuntimeCrashed = errors.New("aster/runtime: WASM runtime is no longer usable (crashed or timed out); create a new Converter")
	errRuntimeClosed  = errors.New("aster/runtime: runtime is closed; create a new Converter")
)

// evalModule evaluates an inline ES module and returns its default export as a string.
// It recovers from panics in the WASM runtime and converts them to errors.
func (r *Runtime) evalModule(script string) (result string, err error) {
	if r.crashed {
		return "", errRuntimeCrashed
	}
	// Close nils r.rt; without this guard a post-Close call would nil-deref
	// into the panic recovery below and masquerade as a WASM crash.
	if r.rt == nil {
		return "", errRuntimeClosed
	}

	defer func() {
		if p := recover(); p != nil {
			r.crashed = true
			err = fmt.Errorf("aster/runtime: WASM panic: %v", p)
		}
	}()

	result, err = r.rt.EvalModule(script)
	if err != nil {
		// A render timeout closes the underlying module mid-eval, and every
		// later call would otherwise fail with an opaque low-level error.
		// Detect the closed/timed-out state, latch it, and return a clear
		// sentinel so callers know to build a new Converter.
		if r.rt.Closed() {
			r.crashed = true
			return "", errRuntimeCrashed
		}
		return "", fmt.Errorf("aster/runtime: eval: %w", err)
	}
	return result, nil
}

// jsStringLiteral encodes s as a JavaScript string literal (double-quoted).
// It is used to pass spec/theme text into generated module source as inert
// data rather than as code: encoding.json escapes quotes, backslashes,
// control characters, and the U+2028/U+2029 line separators, and HTML-escapes
// <, >, & — so no byte of s (backticks, ${...}, quotes, newlines) can break
// out of the literal or be interpreted as a template interpolation.
func jsStringLiteral(s string) (string, error) {
	b, err := json.Marshal(s)
	if err != nil {
		return "", fmt.Errorf("aster/runtime: encoding string literal: %w", err)
	}
	return string(b), nil
}

// themeLiteral encodes the configured theme JSON as a JS string literal, or
// returns the JS keyword "undefined" when no theme is set. bridge.js
// JSON.parses the resulting string.
func themeLiteral(theme string) (string, error) {
	if theme == "" {
		return "undefined", nil
	}
	return jsStringLiteral(theme)
}
