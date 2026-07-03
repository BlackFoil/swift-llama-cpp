import Foundation
import llama

enum NextToken {
    case token(String)
    case endOfString
}

final actor Llama {
    private let model: LlamaModel
    let context: LlamaContext
    private var batch: LlamaBatch
    private var sampler: LlamaSampler!

    // Configuration

    private let config: LlamaConfig
    let maxTokenCount: UInt32
    /// Tracks the current position in the token sequence during decoding.
    var currentTokenPosition: Int32 = 0
    var processedTokens: [llama_token] = []

    init(modelPath: String, config: LlamaConfig) throws {
        self.config = config
        llama_backend_init()
        var model_params = llama_model_default_params()

        if !config.useGPU {
            model_params.n_gpu_layers = 0
        }

        #if targetEnvironment(simulator)
                model_params.n_gpu_layers = 0
                print("Running on simulator, force use n_gpu_layers = 0")
        #endif

        let model = LlamaModel(path: modelPath, parameters: model_params)
        guard let model else {
            print("Could not load model at \(modelPath)")
            throw LlamaError.couldNotInitializeContext
        }

        let n_threads = ProcessInfo.processInfo.processorCount - 1
        print("Using \(n_threads) threads")

        var contextParam = llama_context_default_params()
        contextParam.n_ctx = config.maxTokenCount
        contextParam.n_threads       = 1 // UInt32(n_threads) its actually faster if less threads are doing work
        contextParam.n_threads_batch = 1 // UInt32(n_threads)
        contextParam.n_batch = config.batchSize
        contextParam.n_ubatch = config.batchSize
        contextParam.offload_kqv = true

        let context = LlamaContext(model: model, parameters: contextParam)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }


        self.maxTokenCount = min(UInt32(model.trainedContextSize()), config.maxTokenCount)
        self.model = context.model
        self.context = context
        self.batch = .init(initialSize: Int32(config.batchSize))
    }

    deinit {
        llama_backend_free()
    }

    // Expose some backend/system utilities for convenience
    /// Return system info string from the backend.
    static func printSystemInfo() -> String {
        guard let c = llama_print_system_info() else { return "" }
        return String(cString: c)
    }

    /// Expose the underlying context to trusted callers (tests / advanced users).
    /// Access is actor-isolated; callers must `await`.
    func contextHandle() -> LlamaContext { context }

    // MARK: - Testing & Introspection helpers (actor-safe)

    func getLastLogits() -> [Float]? { context.lastLogits() }
    func getEmbeddings() -> [Float]? { context.embeddings(at: -1) }
    func enableEmbeddingsOutput(_ enabled: Bool) { context.setEmbeddingsOutput(enabled) }
    func saveStateData() -> Data { context.saveState() }
    func loadStateData(_ data: Data) -> Bool { context.loadState(data) }
    func setThreads(nThreads: Int32, nThreadsBatch: Int32) { context.setThreads(nThreads: nThreads, nThreadsBatch: nThreadsBatch) }
    func getThreads() -> (Int32, Int32) { (context.nThreads(), context.nThreadsBatch()) }
    func kvMinPosition() -> Int32 { context.memory.minPosition(for: 0) }
    func kvMaxPosition() -> Int32 { context.memory.maxPosition(for: 0) }
    func clearKV() { context.clearKVCache() }

    /// Return the full processed token id sequence (prompt + generated).
    func getProcessedTokenIds() -> [llama_token] { processedTokens }

    func initializeCompletion(messages: [LlamaChatMessage], addAssistant: Bool? = nil) throws {
        let formattedPrompt = model.applyChatTemplate(to: messages, addAssistant: addAssistant)
        try initializeCompletion(text: formattedPrompt)
    }

    func initializeCompletion(text: String) throws {
        print("attempting to complete \"\(text)\"")

        let tokenList = model.tokenize(text: text, addBos: model.shouldAddBos(), special: true)
        guard !tokenList.isEmpty else {
            throw LlamaError.emptyPrompt
        }
        guard tokenList.count < maxTokenCount - 4 else {
            throw LlamaError.contextSizeLimitExeeded
        }

        if tokenList.starts(with: processedTokens) {
            print("### Using cached processing")
            try processPrompt(tokens: Array(tokenList[processedTokens.count...]), startIndex: processedTokens.count)
        } else {
            // Check if we can optimize by only clearing from the divergence point
            let divergenceIndex = findDivergenceIndex(newTokenList: tokenList, processedTokens: processedTokens)
            
            if divergenceIndex > 0 && shouldUsePartialOptimization(divergenceIndex: divergenceIndex, totalProcessed: processedTokens.count) {
                print("### Using partial optimization from position \(divergenceIndex)")
                do {
                    try optimizedReprocessing(newTokenList: tokenList, divergenceIndex: divergenceIndex)
                } catch LlamaError.aborted {
                    // LLM-8 (Codex R3-M1): a cancellation is not an optimization
                    // failure — the fallback would silently swallow the abort
                    // and burn a full reprocess. Propagate it.
                    throw LlamaError.aborted
                } catch {
                    print("Partial optimization failed, falling back to full reprocessing")
                    clear()
                    try processPrompt(tokens: tokenList, startIndex: 0)
                }
            } else {
                print("### Full reprocessing required")
                clear()
                try processPrompt(tokens: tokenList, startIndex: 0)
            }
        }
    }

    /// Find the index where the two token lists diverge
    private func findDivergenceIndex(newTokenList: [llama_token], processedTokens: [llama_token]) -> Int {
        let minLength = min(newTokenList.count, processedTokens.count)
        for i in 0..<minLength {
            if newTokenList[i] != processedTokens[i] {
                return i
            }
        }
        return minLength
    }
    
    /// Decide whether to use partial optimization based on the divergence point
    private func shouldUsePartialOptimization(divergenceIndex: Int, totalProcessed: Int) -> Bool {
        // Only use partial optimization if:
        // 1. We have a significant amount of processed tokens (at least 10)
        // 2. The divergence is not too early (at least 50% of tokens match)
        // 3. The divergence is not at the very beginning
        
        guard divergenceIndex > 0 && totalProcessed >= 10 else { return false }
        
        let matchPercentage = Double(divergenceIndex) / Double(totalProcessed)
        return matchPercentage >= 0.5 // At least 50% of tokens match
    }
    
    /// Optimized reprocessing that only clears cache from the divergence point
    private func optimizedReprocessing(newTokenList: [llama_token], divergenceIndex: Int) throws {
        // Clear KV cache from the divergence point onward
        context.clearKVCacheFromPosition(Int32(divergenceIndex))
        
        // Update our internal state
        processedTokens = Array(processedTokens[0..<divergenceIndex])
        currentTokenPosition = Int32(divergenceIndex)
        
        // Process only the tokens from the divergence point onward
        let tokensToProcess = Array(newTokenList[divergenceIndex...])
        try processPrompt(tokens: tokensToProcess, startIndex: divergenceIndex)
    }

    func generateNextToken() throws -> NextToken {
        // LLM-8 layer 2: pre-mutation cancellation check — throwing here is a
        // CLEAN abort (state and KV are still consistent, prefix cache kept).
        try checkCancellationFlag()
        // Stop before sampling if we've reached the context limit to avoid mutating sampler state
        if currentTokenPosition >= Int32(maxTokenCount) {
            return .endOfString
        }
        let newTokenId = sampler.sample(context: context)

        if model.isEogToken(newTokenId) || currentTokenPosition >= Int32(maxTokenCount) {
            return .endOfString
        }

        batch.reset()
        batch.addToken(newTokenId, at: currentTokenPosition, logits: true)
        try decodeGuarded(batch: batch)

        // LLM-8 invariant: Swift state mutates only AFTER a successful decode —
        // a failed decode must never leave processedTokens/position ahead of
        // the KV cache (the old pre-decode append corrupted prefix reuse).
        processedTokens.append(newTokenId)
        currentTokenPosition += 1

        return .token(model.piece(from: newTokenId))
    }

    func updateSamplingConfig(_ config: LlamaSamplingConfig) {
        self.sampler = .init(config: config, model: model)
    }

    // MARK: - Cancellation (LLM-8)

    /// Per-stream cancellation flag installed by LlamaService before
    /// initializeCompletion. Layer 2 of the stop guarantee: checked at the
    /// generateNextToken head and between prefill batches — backend-independent
    /// and deterministic (the abort callback, layer 3, is best-effort only).
    private var cancellationFlag: CancellationFlag?

    func setCancellationFlag(_ flag: CancellationFlag?) {
        cancellationFlag = flag
    }

    /// Layer 3 (best-effort): ggml abort callback. Per llama.h it "currently
    /// works only with CPU execution", so it may be inert on Metal — layers
    /// 1/2 carry the actual guarantee. `enabled: false` installs an inert
    /// callback (used by tests to keep clean-cancel runs deterministic).
    func installAbortCallback(enabled: Bool) {
        if enabled, let flag = cancellationFlag {
            context.setAbortCallback { flag.isCancelled }
        } else {
            context.setAbortCallback { false }
        }
    }

    private func checkCancellationFlag() throws {
        if cancellationFlag?.isCancelled == true {
            throw LlamaError.aborted
        }
    }

    /// Dirty-path reset: KV may hold partial writes from a failed decode, so
    /// everything is conservatively discarded. Unlike the legacy `clear()`,
    /// this also rewinds `currentTokenPosition` (clear() relies on a following
    /// full processPrompt to fix it; there is none on the throw path).
    private func resetState() {
        context.clearKVCache()
        processedTokens = []
        currentTokenPosition = 0
        batch = .init(initialSize: Int32(config.batchSize))
    }

    // MARK: - Test seams (LLM-8)

    /// One-shot decode failure injection: the NEXT decode attempt throws
    /// LlamaError.aborted. Consumed unconditionally (auto-false after firing)
    /// so it cannot pollute later streams or tests.
    private var testFailNextDecodeWithAborted = false

    func _testFailNextDecodeWithAborted() {
        testFailNextDecodeWithAborted = true
    }

    /// Single decode funnel with the LLM-8 state-consistency invariant baked
    /// in: if a decode attempt throws, the dirty reset happens HERE, at the
    /// throw site, before rethrowing — callers have no reset responsibility,
    /// and clean throws (flag checks, empty prompt, context limit) never pass
    /// through this function, so the clean/dirty distinction cannot be
    /// miswired. The one-shot injection models a mid-decode abort and takes
    /// the identical path.
    private func decodeGuarded(batch: LlamaBatch) throws {
        if testFailNextDecodeWithAborted {
            testFailNextDecodeWithAborted = false
            resetState()
            throw LlamaError.aborted
        }
        do {
            try context.decode(batch: batch)
        } catch {
            print("llama_decode() failed (\(error)) — resetting state")
            resetState()
            if let llamaError = error as? LlamaError {
                throw llamaError
            }
            throw LlamaError.decodingError
        }
    }

    private func clear() {
        context.clearKVCache()
        processedTokens = []
        batch = .init(initialSize: Int32(config.batchSize))
    }

    private func processBatch() throws {
        // decodeGuarded already converts unknown errors to LlamaError and
        // performs the dirty reset — rethrow untouched so `.aborted` is never
        // masked as `.decodingError` (Codex R2-M1).
        try decodeGuarded(batch: batch)
    }

    private func processPrompt(tokens: [llama_token], startIndex: Int) throws {
        guard !tokens.isEmpty else { return }
        batch.reset()

        // LLM-8 invariant: append to processedTokens only after the batch
        // decoded successfully, so a mid-prefill abort leaves Swift state and
        // KV consistent (clean throw → prefix cache stays valid, no reset).
        var pendingTokens: [llama_token] = []
        for i in 0..<tokens.count {
            let tokenPosition = startIndex + i
            let tokenId = tokens[i]
            batch.addToken(tokenId, at: Int32(tokenPosition), logits: false)
            pendingTokens.append(tokenId)
            if batch.size == config.batchSize {
                // LLM-8 layer 2: batch-boundary cancellation (clean abort).
                try checkCancellationFlag()
                try processBatch()
                processedTokens.append(contentsOf: pendingTokens)
                pendingTokens.removeAll(keepingCapacity: true)
                batch.reset()
            }
        }

        batch.setLastTokenLogits(true)
        try checkCancellationFlag()
        try processBatch()
        processedTokens.append(contentsOf: pendingTokens)

        currentTokenPosition = Int32(processedTokens.count)

    }
}
