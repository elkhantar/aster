package quickjs

import (
	"context"
	"crypto/sha256"
	"testing"

	wazero "github.com/mgilbir/andsifr"
	"github.com/mgilbir/andsifr/api"
	"github.com/mgilbir/andsifr/experimental"
	"github.com/mgilbir/andsifr/imports/wasi_snapshot_preview1"
)

// rawTestModule compiles the embedded QuickJS module in a throwaway runtime and
// returns helpers to instantiate it with a chosen memory allocator, running no
// start functions (so instantiation applies only the module's active data
// segments — the exact state the pre-applied-data skip is about).
func rawTestModule(t *testing.T) (context.Context, func(alloc experimental.MemoryAllocator) api.Module) {
	t.Helper()
	ctx := context.Background()
	wrt := newWazeroRuntime(ctx)
	t.Cleanup(func() { _ = wrt.Close(ctx) })
	if _, err := wasi_snapshot_preview1.Instantiate(ctx, wrt); err != nil {
		t.Fatalf("wasi instantiate: %v", err)
	}
	compiled, err := wrt.CompileModule(ctx, wasmBytes)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	inst := func(alloc experimental.MemoryAllocator) api.Module {
		cfg := wazero.NewModuleConfig().WithName("").WithStartFunctions()
		ic := ctx
		if alloc != nil {
			ic = experimental.WithMemoryAllocator(ctx, alloc)
		}
		mod, err := wrt.InstantiateModule(ic, compiled, cfg)
		if err != nil {
			t.Fatalf("instantiate: %v", err)
		}
		return mod
	}
	return ctx, inst
}

// capturedModuleID returns the andsifr module identity observed for the raw
// module via captureMemory.PreappliedDataFor.
func capturedModuleID(t *testing.T, ctx context.Context, inst func(experimental.MemoryAllocator) api.Module) [32]byte {
	t.Helper()
	alloc := &captureAllocator{}
	mod := inst(alloc)
	defer func() { _ = mod.Close(ctx) }()
	if alloc.mem == nil || !alloc.mem.gotID {
		t.Fatal("module identity was not captured during instantiation; the pre-applied-data skip could never engage")
	}
	return alloc.mem.moduleID
}

// TestSnapshotModuleIDMatchesAndsifr documents andsifr's module-identity formula
// for aster's runtime config. aster does NOT set WithCloseOnContextDone, so
// ensureTermination is false (a single 0 byte), and registers no function
// listeners, so the identity reduces to SHA-256(wasm ++ {0}). If andsifr ever
// changes the formula, the captured ID (used in production) stays correct while
// this cross-check flags the drift for review.
func TestSnapshotModuleIDMatchesAndsifr(t *testing.T) {
	ctx, inst := rawTestModule(t)
	got := capturedModuleID(t, ctx, inst)

	h := sha256.New()
	h.Write(wasmBytes)
	h.Write([]byte{0}) // ensureTermination = false, no listeners
	var want [32]byte
	h.Sum(want[:0])

	if got != want {
		t.Fatalf("captured module ID %x != SHA-256(wasm++{0}) %x; andsifr's identity formula or aster's runtime config changed — review before relying on the fast path", got, want)
	}
}

// TestSnapshotFastPathActive proves the data-segment copy-skip is actually
// taken (not silently disabled by a module-identity mismatch, which would fall
// back to a full copy — fail-safe but with zero speedup). It serves a
// deliberately corrupted image and asserts the corruption survives for the
// correct module ID (skip honored) but is overwritten for a wrong ID (negative
// control). Adapted from loom's TestMemoryImageFastPathActive.
func TestSnapshotFastPathActive(t *testing.T) {
	ctx, inst := rawTestModule(t)
	realID := capturedModuleID(t, ctx, inst)

	// A fresh instance's non-zero bytes come only from active data segments
	// (BSS is zero), so the first non-zero byte is guaranteed inside one.
	base := inst(nil)
	view, ok := base.Memory().Read(0, base.Memory().Size())
	if !ok {
		t.Fatal("read baseline memory")
	}
	baseline := append([]byte(nil), view...)
	_ = base.Close(ctx)

	off := -1
	for i, b := range baseline {
		if b != 0 {
			off = i
			break
		}
	}
	if off < 0 {
		t.Fatal("no non-zero byte in baseline memory; cannot probe a data segment")
	}

	probeWith := func(id [32]byte) byte {
		corrupted := append([]byte(nil), baseline...)
		corrupted[off] ^= 0xFF
		mod := inst(&restoreAllocator{image: corrupted, moduleID: id})
		defer func() { _ = mod.Close(ctx) }()
		got, ok := mod.Memory().Read(uint32(off), 1)
		if !ok {
			t.Fatal("read probe byte")
		}
		return got[0]
	}

	if got := probeWith(realID); got != baseline[off]^0xFF {
		t.Fatalf("fast path inactive: probe byte was re-copied (got %#x, corruption %#x); the pre-applied-data skip is not engaging", got, baseline[off]^0xFF)
	}

	var wrongID [32]byte
	if got := probeWith(wrongID); got != baseline[off] {
		t.Fatalf("negative control failed: wrong-ID image should have been copied over (got %#x, want baseline %#x)", got, baseline[off])
	}
}

// TestSnapshotRestorePristineAndIsolated verifies that restoring a real warmed
// snapshot yields memory bit-identical to the captured image, that a second
// restore is equally pristine (the host image is immutable), and that dirtying
// one restored instance's heap does not leak into a later restore (per-instance
// isolation).
func TestSnapshotRestorePristineAndIsolated(t *testing.T) {
	warm, err := NewWarm(Config{})
	if err != nil {
		t.Fatalf("NewWarm: %v", err)
	}
	defer func() { _ = warm.Close() }()

	snap, err := warm.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	imageHash := sha256.Sum256(snap.image)

	memHash := func(r *Runtime) [32]byte {
		mem := r.mod.Memory()
		pages, _ := mem.Grow(0)
		data, ok := mem.Read(0, pages*65536)
		if !ok {
			t.Fatal("read restored memory")
		}
		return sha256.Sum256(data)
	}

	// First restore must start bit-identical to the image.
	r1, err := Restore(Config{}, snap)
	if err != nil {
		t.Fatalf("Restore #1: %v", err)
	}
	if memHash(r1) != imageHash {
		t.Fatal("restored instance memory differs from snapshot image")
	}

	// Dirty r1 by running a render (grows/mutates the heap).
	if _, err := r1.EvalModule("export default 1 + 1;"); err != nil {
		t.Fatalf("r1 eval: %v", err)
	}
	_ = r1.Close()

	// A later restore must still be pristine: the host image was not mutated,
	// and r1's writes went only to its private copy.
	r2, err := Restore(Config{}, snap)
	if err != nil {
		t.Fatalf("Restore #2: %v", err)
	}
	defer func() { _ = r2.Close() }()
	if memHash(r2) != imageHash {
		t.Fatal("second restored instance not pristine; the snapshot image or a prior instance leaked state")
	}
	if sha256.Sum256(snap.image) != imageHash {
		t.Fatal("snapshot image bytes were mutated after restores")
	}
}

// BenchmarkQuickJSNew measures cold engine construction (instantiate + evaluate
// the vendored bundle) at the quickjs layer.
func BenchmarkQuickJSNew(b *testing.B) {
	// Warm loadModules is done inside New via the runtime layer, not here; this
	// bench only covers a bare quickjs engine (no Vega bundle), so it isolates
	// instantiate + initEngine.
	b.ReportAllocs()
	for b.Loop() {
		r, err := New(Config{})
		if err != nil {
			b.Fatal(err)
		}
		_ = r.Close()
	}
}

// BenchmarkQuickJSWarmVsRestore reports cold NewWarm (instantiate + init) vs
// Restore (instantiate + memory copy + re-resolve + re-arm) for a bare engine.
func BenchmarkQuickJSRestore(b *testing.B) {
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
