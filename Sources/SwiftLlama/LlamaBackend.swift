import Foundation
import llama

public enum LlamaBackend {
    /// Initialize the llama + ggml backend. Call once at program start.
    public static func initialize() { llama_backend_init() }
    /// Free the backend. Call once at program end.
    public static func shutdown() { llama_backend_free() }

    // MARK: - Reference-counted retain/release

    /// The backend is process-global, but `Llama` instances come and go and can
    /// overlap — a model being warmed up while the previous one is still
    /// answering, for example. Pairing a bare `llama_backend_init()` with a
    /// bare `llama_backend_free()` per instance gets that wrong in both
    /// directions: a throwing initializer leaks the init (Swift does not run
    /// `deinit` for an object that never finished initializing), and the first
    /// instance to be released frees a backend the others are still using.
    ///
    /// Counting fixes both. `retain` initializes on the first caller; `release`
    /// frees only when the last one lets go.
    private static let lock = NSLock()
    /// `nonisolated(unsafe)`: 直上の `lock` で全アクセスを直列化しているため、
    /// コンパイラの追跡外で安全性が保たれている。
    private nonisolated(unsafe) static var retainCount = 0

    /// Initializes the backend if this is the first live user.
    public static func retain() {
        lock.lock()
        defer { lock.unlock() }
        if retainCount == 0 { llama_backend_init() }
        retainCount += 1
    }

    /// Frees the backend when the last user lets go. Safe to call after a
    /// failed initialization; extra calls past zero are ignored rather than
    /// double-freeing.
    public static func release() {
        lock.lock()
        defer { lock.unlock() }
        guard retainCount > 0 else { return }
        retainCount -= 1
        if retainCount == 0 { llama_backend_free() }
    }

    /// Test seam: how many live users the backend believes it has.
    public static var _testRetainCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return retainCount
    }
    /// Whether mmap/mlock/gpu offload/rpc are supported by the compiled library.
    public static var supportsMmap: Bool { llama_supports_mmap() }
    public static var supportsMlock: Bool { llama_supports_mlock() }
    public static var supportsGpuOffload: Bool { llama_supports_gpu_offload() }
    public static var supportsRpc: Bool { llama_supports_rpc() }
    /// Maximum devices and parallel sequences
    public static var maxDevices: Int { Int(llama_max_devices()) }
    public static var maxParallelSequences: Int { Int(llama_max_parallel_sequences()) }

    /// Initialize NUMA with a given strategy.
    public static func numaInit(_ strategy: ggml_numa_strategy) { llama_numa_init(strategy) }

    /// Microsecond timer from llama.cpp
    public static func timeMicros() -> Int64 { llama_time_us() }

    /// Return system info string provided by llama.cpp
    public static func systemInfo() -> String {
        guard let c = llama_print_system_info() else { return "" }
        return String(cString: c)
    }

    /// Attach the library-managed auto threadpool to a context.
    public static func attachAutoThreadpool(to context: LlamaContext) {
        llama_attach_threadpool(context.contextPointer, nil, nil)
    }

    /// Detach any threadpools from the context.
    public static func detachThreadpool(from context: LlamaContext) {
        llama_detach_threadpool(context.contextPointer)
    }
}

