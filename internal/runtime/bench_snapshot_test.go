package runtime

import "testing"

// BenchmarkRuntimeNew measures cold runtime construction: quickjs engine +
// installPolyfills + loadModules (the ~145ms Vega/Vega-Lite bundle eval).
func BenchmarkRuntimeNew(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		r, err := New(Config{})
		if err != nil {
			b.Fatal(err)
		}
		_ = r.Close()
	}
}

// BenchmarkRuntimeRestore measures warm-start restore at the runtime layer:
// instantiate + private memory copy + export re-resolve + re-arm, skipping
// installPolyfills and loadModules entirely.
func BenchmarkRuntimeRestore(b *testing.B) {
	warm, err := NewWarm(Config{})
	if err != nil {
		b.Fatal(err)
	}
	defer warm.Close()
	snap, err := warm.Snapshot()
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		r, err := Restore(Config{}, snap)
		if err != nil {
			b.Fatal(err)
		}
		_ = r.Close()
	}
}
