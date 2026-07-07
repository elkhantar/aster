package quickjs

import (
	"context"
	"errors"

	"github.com/mgilbir/andsifr/experimental"
)

// Snapshot is an immutable image of a warmed QuickJS runtime: a copy of its
// entire WASM linear memory plus the host-side handles (context/runtime
// pointers and the scratch buffer) needed to resume it in a fresh module
// instance.
//
// WASM linear memory is self-contained and offset-0 based: every QuickJS heap
// pointer, and the ctxPtr/rtPtr/scratch the Runtime holds, is an offset into
// that one memory. So restoring the image into a fresh instance and reusing the
// captured handles resumes the engine exactly where it was snapshotted, without
// re-evaluating the Vega/Vega-Lite bundle. The api.Function exports are
// per-instance and are re-resolved on Restore, never captured here.
//
// A Snapshot is read-only after capture and safe for concurrent use: each
// Restore copies the image into private per-instance memory and never aliases
// the Snapshot's bytes, so restored instances cannot observe or mutate one
// another (or the Snapshot).
type Snapshot struct {
	image    []byte   // full linear memory at capture; never mutated afterwards
	ctxPtr   uint32   // JSContext*
	rtPtr    uint32   // JSRuntime*
	scratch  uint32   // out-param scratch buffer
	moduleID [32]byte // andsifr module identity the image was captured under
}

// Pages reports the linear-memory page count captured in the snapshot.
func (s *Snapshot) Pages() int { return len(s.image) / 65536 }

// Bytes reports the linear-memory size captured in the snapshot.
func (s *Snapshot) Bytes() int { return len(s.image) }

// NewWarm creates a runtime exactly like New but backed by a snapshot-capable
// allocator, so Snapshot can later capture its warmed linear memory. Its
// output is byte-identical to a runtime from New; the allocator only mirrors
// the default Go-slice memory while recording the module identity Snapshot
// needs.
func NewWarm(cfg Config) (*Runtime, error) {
	ctx := context.Background()
	wrt := newWazeroRuntime(ctx)

	r := &Runtime{wrt: wrt, timeout: cfg.Timeout}
	alloc := &captureAllocator{}
	if err := r.instantiate(ctx, cfg, alloc, true); err != nil {
		_ = wrt.Close(ctx)
		return nil, err
	}
	r.capMem = alloc.mem
	if err := r.initEngine(ctx, cfg); err != nil {
		_ = wrt.Close(ctx)
		return nil, err
	}
	return r, nil
}

// Snapshot captures the current linear memory and engine handles of a runtime
// created via NewWarm. The runtime must be quiescent (no pending jobs, no
// in-flight calls); callers snapshot immediately after warm-up, before any
// render. The returned Snapshot is independent of the runtime, which remains
// usable.
func (r *Runtime) Snapshot() (*Snapshot, error) {
	if r.closed || r.mod == nil || r.mod.IsClosed() {
		return nil, errors.New("quickjs: cannot snapshot a closed runtime")
	}
	if r.capMem == nil || !r.capMem.gotID {
		// gotID is set when andsifr consults PreappliedDataFor during
		// instantiation. Without it we do not know the module identity a
		// Restore must match, so the fast path could never engage.
		return nil, errors.New("quickjs: runtime not created for snapshotting (module identity not captured); use NewWarm")
	}

	mem := r.mod.Memory()
	// Size() overflows to 0 at the 4 GiB maximum; Grow(0) yields the current
	// page count reliably.
	pages, _ := mem.Grow(0)
	size := pages * 65536
	view, ok := mem.Read(0, size)
	if !ok {
		return nil, errors.New("quickjs: reading linear memory for snapshot: out of bounds")
	}
	image := make([]byte, len(view))
	copy(image, view)

	return &Snapshot{
		image:    image,
		ctxPtr:   r.ctxPtr,
		rtPtr:    r.rtPtr,
		scratch:  r.scratch,
		moduleID: r.capMem.moduleID,
	}, nil
}

// Restore builds a runtime from a snapshot. It instantiates a fresh module
// instance whose linear memory is a private copy of the snapshot image, skips
// the guest initialization and Vega/Vega-Lite loading that produced the image,
// reuses the captured engine handles, and re-arms the per-instance stack guard
// and memory limit. cfg's Bridge (loader, text measurer) is wired freshly per
// instance and may differ from the warm runtime's — bridge callbacks fire at
// render time and carry no state in the snapshot image.
func Restore(cfg Config, snap *Snapshot) (*Runtime, error) {
	if snap == nil {
		return nil, errors.New("quickjs: nil snapshot")
	}
	ctx := context.Background()
	wrt := newWazeroRuntime(ctx)

	r := &Runtime{wrt: wrt, timeout: cfg.Timeout}
	alloc := &restoreAllocator{image: snap.image, moduleID: snap.moduleID}
	if err := r.instantiate(ctx, cfg, alloc, false); err != nil {
		_ = wrt.Close(ctx)
		return nil, err
	}
	// Resume the warmed engine: its context/runtime/scratch live at fixed
	// offsets inside the restored linear memory.
	r.ctxPtr = snap.ctxPtr
	r.rtPtr = snap.rtPtr
	r.scratch = snap.scratch
	if err := r.armEngine(ctx, cfg); err != nil {
		_ = wrt.Close(ctx)
		return nil, err
	}
	return r, nil
}

// captureMemory is the LinearMemory used during a warm (cold) instantiation. It
// behaves exactly like andsifr's default Go-slice allocation (initial length =
// the module minimum, capacity = the module's capacity hint, growth preserves
// contents) so a warmed runtime's output is byte-identical to a default one. It
// additionally records the andsifr module identity observed during data-segment
// application, and never claims pre-applied data (returns false) so the warm
// instance copies its data segments normally.
type captureMemory struct {
	buf      []byte
	max      uint64
	moduleID [32]byte
	gotID    bool
}

func (m *captureMemory) Reallocate(size uint64) []byte {
	if uint64(cap(m.buf)) >= size {
		m.buf = m.buf[:size]
		return m.buf
	}
	if m.max != 0 && size > m.max {
		return nil
	}
	nb := make([]byte, size)
	copy(nb, m.buf)
	m.buf = nb
	return m.buf
}

func (m *captureMemory) Free() {}

// PreappliedDataFor records the module identity andsifr asks about and always
// returns false: a warm instance must copy its data segments normally.
func (m *captureMemory) PreappliedDataFor(id [32]byte) bool {
	m.moduleID = id
	m.gotID = true
	return false
}

type captureAllocator struct{ mem *captureMemory }

func (a *captureAllocator) Allocate(capBytes, maxBytes uint64) experimental.LinearMemory {
	a.mem = &captureMemory{buf: make([]byte, 0, capBytes), max: maxBytes}
	return a.mem
}

// restoreMemory is a LinearMemory backed by a private per-instance copy of a
// snapshot image. Unlike a min-size (build-time pre-initialized) image, aster's
// snapshot is captured after the Vega/Vega-Lite heap has grown well beyond the
// module's declared minimum, so the first Reallocate (which andsifr issues at
// the module minimum) must return the entire image rather than truncating it to
// the minimum — otherwise the warmed heap above the minimum would be lost.
// Later Reallocate calls grow the buffer. It declares the image already
// contains the module's active data segments, but only for the exact module
// identity the snapshot was captured from.
type restoreMemory struct {
	buf      []byte
	max      uint64
	moduleID [32]byte
}

func (m *restoreMemory) Reallocate(size uint64) []byte {
	if uint64(len(m.buf)) >= size {
		// The initial request is the module minimum, always <= the image
		// length; return the full warmed image, not a truncated prefix.
		return m.buf
	}
	if m.max != 0 && size > m.max {
		return nil
	}
	nb := make([]byte, size)
	copy(nb, m.buf)
	m.buf = nb
	return m.buf
}

// Free may run from the timeout watcher while the guest still executes to its
// next termination check. Dropping this reference is safe because the module
// instance's own Buffer slice keeps the backing array alive for the GC until
// the instance is gone — there is no manual unmap to race (unlike a
// memfd/mmap-backed image), so no reference counting is needed.
func (m *restoreMemory) Free() { m.buf = nil }

// PreappliedDataFor reports that this instance's memory already holds the
// module's active data segments — but only for the exact module the snapshot
// was captured from. A mismatch merely makes andsifr fall back to copying the
// segments (fail-safe, never a wrong result).
func (m *restoreMemory) PreappliedDataFor(id [32]byte) bool { return id == m.moduleID }

type restoreAllocator struct {
	image    []byte
	moduleID [32]byte
}

func (a *restoreAllocator) Allocate(capBytes, maxBytes uint64) experimental.LinearMemory {
	// Each instance gets its OWN private copy of the image; two instances never
	// share a backing slice, so one instance can never observe or corrupt
	// another's memory (or the immutable source image).
	buf := make([]byte, len(a.image))
	copy(buf, a.image)
	return &restoreMemory{buf: buf, max: maxBytes, moduleID: a.moduleID}
}
