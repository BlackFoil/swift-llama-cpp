//
//  CancellationFlag.swift
//  swift-llama-cpp
//
//  LLM-8 (F1): backend-independent cancellation for generation AND prefill.
//

import Foundation

/// A lock-protected boolean shared between the LlamaService actor, the
/// producer task, and (best-effort) the ggml abort callback.
///
/// One instance is created per stream — a cancelled flag from a previous
/// stream can never abort the next prefill, structurally (no reset needed).
///
/// Thread-safety: a single NSLock around a Bool. `isCancelled` is read from
/// the ggml compute thread via the abort callback — it must never hop actors
/// or allocate; a lock-guarded read is ~ns and cannot deadlock.
final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }
}
