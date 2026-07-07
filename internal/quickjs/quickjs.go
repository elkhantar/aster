package quickjs

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	"github.com/mgilbir/andsifr"
	"github.com/mgilbir/andsifr/api"
	"github.com/mgilbir/andsifr/experimental"
	"github.com/mgilbir/andsifr/imports/wasi_snapshot_preview1"
)

// JSValue encoding on wasm32 (NaN-boxed): a uint64 with the tag in the high
// 32 bits and the payload (pointer or immediate) in the low 32 bits.
const (
	jsTagModule    = -3
	jsTagException = 6
	jsTagUndefined = 3
)

// JS_Eval flags.
const (
	jsEvalTypeGlobal      = 0
	jsEvalTypeModule      = 1
	jsEvalFlagCompileOnly = 1 << 5
)

// JS_PromiseState results.
const (
	jsPromisePending   = 0
	jsPromiseFulfilled = 1
	jsPromiseRejected  = 2
)

// jsStackSize is the JS engine stack budget; see initEngine for rationale.
const jsStackSize = 1 << 20

func valueTag(v uint64) int32     { return int32(v >> 32) }
func valuePtr(v uint64) uint32    { return uint32(v) }
func isException(v uint64) bool   { return valueTag(v) == jsTagException }
func isUndefined(v uint64) bool   { return valueTag(v) == jsTagUndefined }

// Config configures a Runtime.
type Config struct {
	// MemoryLimit caps the JS heap in bytes (0 = engine default). WASM linear
	// memory is 32-bit addressable, so values above 4 GiB are clamped.
	MemoryLimit uint64
	// Timeout bounds each evaluation; on expiry the underlying WASM module
	// is closed and the Runtime becomes unusable (0 = no limit). The
	// deadline takes effect at the next host-call boundary (data loading,
	// text measurement, clock reads); pure-JS loops that never leave the
	// engine are not preempted. wazero's WithCloseOnContextDone would close
	// that gap but costs 3-6x on compute-heavy renders, so it stays off.
	Timeout time.Duration
	// Bridge exposes Go callbacks to JavaScript.
	Bridge Bridge
	// Stdout and Stderr receive the guest's console output (default: discarded).
	Stdout, Stderr io.Writer
}

// compilationCache shares compiled machine code across Runtimes in this
// process. Each Runtime still compiles + instantiates within its own wazero
// runtime, so no compiled-module state ever crosses wazero runtimes.
var compilationCache = wazero.NewCompilationCache()

// Runtime is a QuickJS engine instance. It is not safe for concurrent use.
type Runtime struct {
	wrt wazero.Runtime
	mod api.Module

	fnMalloc              api.Function
	fnFree                api.Function
	fnQJSInitArgv         api.Function
	fnQJSGetContext       api.Function
	fnQJSDestroy          api.Function
	fnGetRuntime          api.Function
	fnSetMemoryLimit      api.Function
	fnSetMaxStackSize     api.Function
	fnUpdateStackTop      api.Function
	fnEval                api.Function
	fnEvalFunction        api.Function
	fnExecutePendingJob   api.Function
	fnPromiseState        api.Function
	fnPromiseResult       api.Function
	fnGetModuleNamespace  api.Function
	fnGetPropertyStr      api.Function
	fnToCStringLen2       api.Function
	fnFreeCString         api.Function
	fnFreeValue           api.Function
	fnGetException        api.Function

	ctxPtr  uint32 // JSContext*
	rtPtr   uint32 // JSRuntime*
	scratch uint32 // 8 bytes of guest memory for out-params

	// capMem is set only for runtimes created via NewWarm. It records the
	// andsifr module identity observed during instantiation, which Snapshot
	// needs so a later Restore can bind its image to the exact same module.
	capMem *captureMemory

	timeout time.Duration
	closed  bool
}

// New creates a QuickJS runtime, initializes the engine, and installs the
// __aster_* bridge globals for the configured callbacks.
func New(cfg Config) (*Runtime, error) {
	ctx := context.Background()
	wrt := newWazeroRuntime(ctx)

	r := &Runtime{wrt: wrt, timeout: cfg.Timeout}
	if err := r.instantiate(ctx, cfg, nil, true); err != nil {
		_ = wrt.Close(ctx)
		return nil, err
	}
	if err := r.initEngine(ctx, cfg); err != nil {
		_ = wrt.Close(ctx)
		return nil, err
	}
	return r, nil
}

// newWazeroRuntime builds a wazero runtime sharing the process-wide compilation
// cache. Every Runtime still compiles + instantiates within its own wazero
// runtime, so no compiled-module state ever crosses wazero runtimes.
func newWazeroRuntime(ctx context.Context) wazero.Runtime {
	rcfg := wazero.NewRuntimeConfig().
		WithCompilationCache(compilationCache)
	return wazero.NewRuntimeWithConfig(ctx, rcfg)
}

// instantiate compiles and instantiates the QuickJS WASM module and wires its
// exports. When alloc is non-nil it backs the instance's linear memory (used
// for snapshot capture and restore). runInit controls whether the WASI
// reactor's _initialize runs: true for a fresh engine, false when restoring a
// snapshot image whose guest initialization already ran (re-running it would
// corrupt the warmed heap).
func (r *Runtime) instantiate(ctx context.Context, cfg Config, alloc experimental.MemoryAllocator, runInit bool) error {
	if _, err := wasi_snapshot_preview1.Instantiate(ctx, r.wrt); err != nil {
		return fmt.Errorf("quickjs: instantiating WASI: %w", err)
	}

	compiled, err := r.wrt.CompileModule(ctx, wasmBytes)
	if err != nil {
		return fmt.Errorf("quickjs: compiling module: %w", err)
	}

	stdout := cfg.Stdout
	if stdout == nil {
		stdout = io.Discard
	}
	stderr := cfg.Stderr
	if stderr == nil {
		stderr = io.Discard
	}

	mcfg := wazero.NewModuleConfig().
		WithName("quickjs").
		WithFSConfig(wazero.NewFSConfig().WithFSMount(newBridgeFS(cfg.Bridge), "/aster")).
		WithStdout(stdout).
		WithStderr(stderr).
		WithSysWalltime().
		WithSysNanotime().
		WithSysNanosleep()
	if runInit {
		if _, ok := compiled.ExportedFunctions()["_initialize"]; ok {
			mcfg = mcfg.WithStartFunctions("_initialize")
		}
	} else {
		// Restoring a snapshot image: the guest is already initialized, so run
		// no start functions (clearing the default "_start" too).
		mcfg = mcfg.WithStartFunctions()
	}

	instCtx := ctx
	if alloc != nil {
		instCtx = experimental.WithMemoryAllocator(ctx, alloc)
	}

	mod, err := r.wrt.InstantiateModule(instCtx, compiled, mcfg)
	if err != nil {
		return fmt.Errorf("quickjs: instantiating module: %w", err)
	}
	r.mod = mod

	return r.wireExports()
}

// wireExports resolves and validates the required WASM exports.
func (r *Runtime) wireExports() error {
	exports := map[string]*api.Function{
		"malloc":                &r.fnMalloc,
		"free":                  &r.fnFree,
		"qjs_init_argv":         &r.fnQJSInitArgv,
		"qjs_get_context":       &r.fnQJSGetContext,
		"qjs_destroy":           &r.fnQJSDestroy,
		"JS_GetRuntime":         &r.fnGetRuntime,
		"JS_SetMemoryLimit":     &r.fnSetMemoryLimit,
		"JS_SetMaxStackSize":    &r.fnSetMaxStackSize,
		"JS_UpdateStackTop":     &r.fnUpdateStackTop,
		"JS_Eval":               &r.fnEval,
		"JS_EvalFunction":       &r.fnEvalFunction,
		"JS_ExecutePendingJob":  &r.fnExecutePendingJob,
		"JS_PromiseState":       &r.fnPromiseState,
		"JS_PromiseResult":      &r.fnPromiseResult,
		"JS_GetModuleNamespace": &r.fnGetModuleNamespace,
		"JS_GetPropertyStr":     &r.fnGetPropertyStr,
		"JS_ToCStringLen2":      &r.fnToCStringLen2,
		"JS_FreeCString":        &r.fnFreeCString,
		"JS_FreeValue":          &r.fnFreeValue,
		"JS_GetException":       &r.fnGetException,
	}
	for name, fn := range exports {
		*fn = r.mod.ExportedFunction(name)
		if *fn == nil {
			return fmt.Errorf("quickjs: missing WASM export: %s", name)
		}
	}
	return nil
}

func (r *Runtime) initEngine(ctx context.Context, cfg Config) error {
	// qjs_init_argv performs the standard qjs CLI setup: runtime, context,
	// std handlers, module loader, and (with --std) the std/os globals the
	// bridge bootstrap needs for std.loadFile.
	args := []string{"qjs", "--std"}
	argPtrs := make([]uint32, len(args))
	for i, a := range args {
		p, err := r.writeCString(ctx, a)
		if err != nil {
			return err
		}
		argPtrs[i] = p
	}
	argvPtr, err := r.malloc(ctx, uint32(4*len(args)))
	if err != nil {
		return err
	}
	for i, p := range argPtrs {
		if !r.mod.Memory().WriteUint32Le(argvPtr+uint32(4*i), p) {
			return errors.New("quickjs: writing argv: out of bounds")
		}
	}
	res, err := r.fnQJSInitArgv.Call(ctx, uint64(len(args)), uint64(argvPtr))
	r.free(ctx, argvPtr)
	for _, p := range argPtrs {
		r.free(ctx, p)
	}
	if err != nil {
		return fmt.Errorf("quickjs: qjs_init_argv: %w", err)
	}
	if int32(res[0]) != 0 {
		return fmt.Errorf("quickjs: qjs_init_argv returned %d", int32(res[0]))
	}

	res, err = r.fnQJSGetContext.Call(ctx)
	if err != nil {
		return fmt.Errorf("quickjs: qjs_get_context: %w", err)
	}
	r.ctxPtr = uint32(res[0])
	if r.ctxPtr == 0 {
		return errors.New("quickjs: qjs_get_context returned null")
	}

	res, err = r.fnGetRuntime.Call(ctx, uint64(r.ctxPtr))
	if err != nil {
		return fmt.Errorf("quickjs: JS_GetRuntime: %w", err)
	}
	r.rtPtr = uint32(res[0])

	if r.scratch, err = r.malloc(ctx, 8); err != nil {
		return err
	}

	if err := r.armEngine(ctx, cfg); err != nil {
		return err
	}

	boot := bootstrapScript(cfg.Bridge)
	if boot != "" {
		if err := r.evalGlobal(ctx, "__aster_bootstrap__.js", boot); err != nil {
			return fmt.Errorf("quickjs: bridge bootstrap: %w", err)
		}
	}
	return nil
}

// armEngine arms the JS stack-overflow guard and applies the configured memory
// limit. It is called after a fresh engine init and again after restoring a
// snapshot, so a restored instance re-captures its own stack top and honors
// cfg's limits (the stack guard was re-enabled on WASI by
// quickjs-wasm/stack-guard.patch).
//
// Calling JS_UpdateStackTop from here — a fresh, shallow wasm entry — captures
// a reliable stack top; engine init ran unguarded, matching upstream. The 1MB
// budget must stay well under both the 8MB linker stack in quickjs.wasm and
// wazero's native call-stack ceiling (each guest frame costs far more native
// stack than guest stack), so runaway recursion (e.g. animation specs bouncing
// off the immediate setTimeout polyfill) raises a catchable RangeError instead
// of overflowing either stack.
func (r *Runtime) armEngine(ctx context.Context, cfg Config) error {
	if _, err := r.fnUpdateStackTop.Call(ctx, uint64(r.rtPtr)); err != nil {
		return fmt.Errorf("quickjs: JS_UpdateStackTop: %w", err)
	}
	if _, err := r.fnSetMaxStackSize.Call(ctx, uint64(r.rtPtr), uint64(uint32(jsStackSize))); err != nil {
		return fmt.Errorf("quickjs: JS_SetMaxStackSize: %w", err)
	}
	if cfg.MemoryLimit > 0 {
		limit := cfg.MemoryLimit
		if limit > math.MaxUint32 {
			limit = math.MaxUint32 // WASM linear memory is 32-bit addressable
		}
		if _, err := r.fnSetMemoryLimit.Call(ctx, uint64(r.rtPtr), limit); err != nil {
			return fmt.Errorf("quickjs: JS_SetMemoryLimit: %w", err)
		}
	}
	return nil
}

// Closed reports whether the runtime can no longer evaluate code, either
// because Close was called or because the timeout watchdog closed the module
// mid-evaluation.
func (r *Runtime) Closed() bool {
	return r.closed || r.mod == nil || r.mod.IsClosed()
}

// Close releases the engine and the underlying WASM runtime.
func (r *Runtime) Close() error {
	if r.closed {
		return nil
	}
	r.closed = true
	ctx := context.Background()
	if r.ctxPtr != 0 && r.mod != nil && !r.mod.IsClosed() {
		_, _ = r.fnQJSDestroy.Call(ctx)
	}
	return r.wrt.Close(ctx)
}

// callCtx starts the evaluation watchdog and returns the call context plus
// a stop function. When the timeout fires the module is closed, which makes
// the running evaluation fail at its next host-call boundary and renders
// the Runtime unusable.
func (r *Runtime) callCtx() (context.Context, context.CancelFunc) {
	ctx := context.Background()
	if r.timeout <= 0 {
		return ctx, func() {}
	}
	timer := time.AfterFunc(r.timeout, func() {
		_ = r.mod.CloseWithExitCode(ctx, uint32(sysDeadlineExceededExitCode))
	})
	return ctx, func() { timer.Stop() }
}

// sysDeadlineExceededExitCode marks a watchdog-triggered module close in the
// resulting "module closed with exit_code(N)" error.
const sysDeadlineExceededExitCode = 0xdead

// EvalGlobal evaluates code as a classic global script.
func (r *Runtime) EvalGlobal(filename, code string) error {
	ctx, cancel := r.callCtx()
	defer cancel()
	return r.evalGlobal(ctx, filename, code)
}

func (r *Runtime) evalGlobal(ctx context.Context, filename, code string) error {
	val, err := r.eval(ctx, filename, code, jsEvalTypeGlobal)
	if err != nil {
		return err
	}
	if isException(val) {
		return r.exceptionError(ctx)
	}
	r.freeValue(ctx, val)
	return nil
}

// LoadModule evaluates code as an ES module registered under name, making it
// importable by later modules. Modules must be loaded in dependency order.
func (r *Runtime) LoadModule(name, code string) error {
	ctx, cancel := r.callCtx()
	defer cancel()

	val, err := r.eval(ctx, name, code, jsEvalTypeModule)
	if err != nil {
		return err
	}
	if isException(val) {
		return r.exceptionError(ctx)
	}
	defer r.freeValue(ctx, val)

	if err := r.pumpJobs(ctx); err != nil {
		return err
	}
	return r.checkPromise(ctx, val)
}

// EvalModule evaluates code as an ES module (top-level await supported) and
// returns its default export converted to a string.
func (r *Runtime) EvalModule(code string) (string, error) {
	ctx, cancel := r.callCtx()
	defer cancel()

	val, err := r.eval(ctx, "__aster_eval__.js", code, jsEvalTypeModule|jsEvalFlagCompileOnly)
	if err != nil {
		return "", err
	}
	if isException(val) {
		return "", r.exceptionError(ctx)
	}
	if valueTag(val) != jsTagModule {
		r.freeValue(ctx, val)
		return "", fmt.Errorf("quickjs: expected module, got tag %d", valueTag(val))
	}
	modDefPtr := valuePtr(val)

	// JS_EvalFunction consumes val.
	res, err := r.fnEvalFunction.Call(ctx, uint64(r.ctxPtr), val)
	if err != nil {
		return "", fmt.Errorf("quickjs: JS_EvalFunction: %w", err)
	}
	prom := res[0]
	if isException(prom) {
		return "", r.exceptionError(ctx)
	}
	defer r.freeValue(ctx, prom)

	if err := r.pumpJobs(ctx); err != nil {
		return "", err
	}
	if err := r.checkPromise(ctx, prom); err != nil {
		return "", err
	}

	nsRes, err := r.fnGetModuleNamespace.Call(ctx, uint64(r.ctxPtr), uint64(modDefPtr))
	if err != nil {
		return "", fmt.Errorf("quickjs: JS_GetModuleNamespace: %w", err)
	}
	ns := nsRes[0]
	if isException(ns) {
		return "", r.exceptionError(ctx)
	}
	defer r.freeValue(ctx, ns)

	dv, err := r.getPropertyStr(ctx, ns, "default")
	if err != nil {
		return "", err
	}
	defer r.freeValue(ctx, dv)

	return r.toGoString(ctx, dv)
}

// eval runs JS_Eval and returns the raw JSValue result.
func (r *Runtime) eval(ctx context.Context, filename, code string, flags int) (uint64, error) {
	codePtr, err := r.writeCString(ctx, code)
	if err != nil {
		return 0, err
	}
	defer r.free(ctx, codePtr)
	filePtr, err := r.writeCString(ctx, filename)
	if err != nil {
		return 0, err
	}
	defer r.free(ctx, filePtr)

	res, err := r.fnEval.Call(ctx,
		uint64(r.ctxPtr), uint64(codePtr), uint64(uint32(len(code))),
		uint64(filePtr), uint64(uint32(flags)))
	if err != nil {
		return 0, fmt.Errorf("quickjs: JS_Eval: %w", err)
	}
	return res[0], nil
}

// pumpJobs drains the pending-job queue (promise reactions / microtasks).
func (r *Runtime) pumpJobs(ctx context.Context) error {
	var firstErr error
	for {
		res, err := r.fnExecutePendingJob.Call(ctx, uint64(r.rtPtr), uint64(r.scratch))
		if err != nil {
			return fmt.Errorf("quickjs: JS_ExecutePendingJob: %w", err)
		}
		n := int32(res[0])
		if n == 0 {
			return firstErr
		}
		if n < 0 && firstErr == nil {
			firstErr = r.exceptionError(ctx)
		}
	}
}

// checkPromise inspects val; when it is a promise it must be settled, and a
// rejection is converted to an error. Non-promise values pass through.
func (r *Runtime) checkPromise(ctx context.Context, val uint64) error {
	res, err := r.fnPromiseState.Call(ctx, uint64(r.ctxPtr), val)
	if err != nil {
		return fmt.Errorf("quickjs: JS_PromiseState: %w", err)
	}
	switch int32(res[0]) {
	case jsPromisePending:
		return errors.New("quickjs: unsettled promise (pending async work)")
	case jsPromiseRejected:
		resultRes, err := r.fnPromiseResult.Call(ctx, uint64(r.ctxPtr), val)
		if err != nil {
			return fmt.Errorf("quickjs: JS_PromiseResult: %w", err)
		}
		reason := resultRes[0]
		defer r.freeValue(ctx, reason)
		return errors.New(r.errorText(ctx, reason))
	default: // fulfilled, or not a promise at all
		return nil
	}
}

// exceptionError takes the current pending exception and formats it.
func (r *Runtime) exceptionError(ctx context.Context) error {
	res, err := r.fnGetException.Call(ctx, uint64(r.ctxPtr))
	if err != nil {
		return fmt.Errorf("quickjs: JS_GetException: %w", err)
	}
	exVal := res[0]
	defer r.freeValue(ctx, exVal)
	return errors.New(r.errorText(ctx, exVal))
}

// errorText renders a JS error value as "message" plus its stack when present.
func (r *Runtime) errorText(ctx context.Context, val uint64) string {
	msg, err := r.toGoString(ctx, val)
	if err != nil {
		return "unknown JavaScript error"
	}
	if sv, err := r.getPropertyStr(ctx, val, "stack"); err == nil {
		if !isUndefined(sv) {
			if stack, err := r.toGoString(ctx, sv); err == nil {
				stack = strings.TrimRight(stack, "\n")
				if stack != "" {
					msg += "\n" + stack
				}
			}
		}
		r.freeValue(ctx, sv)
	}
	return msg
}

func (r *Runtime) getPropertyStr(ctx context.Context, obj uint64, prop string) (uint64, error) {
	propPtr, err := r.writeCString(ctx, prop)
	if err != nil {
		return 0, err
	}
	defer r.free(ctx, propPtr)
	res, err := r.fnGetPropertyStr.Call(ctx, uint64(r.ctxPtr), obj, uint64(propPtr))
	if err != nil {
		return 0, fmt.Errorf("quickjs: JS_GetPropertyStr: %w", err)
	}
	return res[0], nil
}

// toGoString converts a JSValue to a Go string via JS_ToCStringLen2.
func (r *Runtime) toGoString(ctx context.Context, val uint64) (string, error) {
	res, err := r.fnToCStringLen2.Call(ctx, uint64(r.ctxPtr), uint64(r.scratch), val, 0)
	if err != nil {
		return "", fmt.Errorf("quickjs: JS_ToCStringLen2: %w", err)
	}
	strPtr := uint32(res[0])
	if strPtr == 0 {
		return "", r.exceptionError(ctx)
	}
	defer func() { _, _ = r.fnFreeCString.Call(ctx, uint64(r.ctxPtr), uint64(strPtr)) }()

	strLen, ok := r.mod.Memory().ReadUint32Le(r.scratch)
	if !ok {
		return "", errors.New("quickjs: reading string length: out of bounds")
	}
	data, ok := r.mod.Memory().Read(strPtr, strLen)
	if !ok {
		return "", errors.New("quickjs: reading string data: out of bounds")
	}
	return string(data), nil
}

func (r *Runtime) freeValue(ctx context.Context, val uint64) {
	_, _ = r.fnFreeValue.Call(ctx, uint64(r.ctxPtr), val)
}

func (r *Runtime) malloc(ctx context.Context, size uint32) (uint32, error) {
	res, err := r.fnMalloc.Call(ctx, uint64(size))
	if err != nil {
		return 0, fmt.Errorf("quickjs: malloc: %w", err)
	}
	ptr := uint32(res[0])
	if ptr == 0 {
		return 0, errors.New("quickjs: malloc returned null")
	}
	return ptr, nil
}

func (r *Runtime) free(ctx context.Context, ptr uint32) {
	_, _ = r.fnFree.Call(ctx, uint64(ptr))
}

// writeCString copies s into guest memory with a NUL terminator.
func (r *Runtime) writeCString(ctx context.Context, s string) (uint32, error) {
	ptr, err := r.malloc(ctx, uint32(len(s)+1))
	if err != nil {
		return 0, err
	}
	if !r.mod.Memory().Write(ptr, append([]byte(s), 0)) {
		r.free(ctx, ptr)
		return 0, errors.New("quickjs: writing string: out of bounds")
	}
	return ptr, nil
}
