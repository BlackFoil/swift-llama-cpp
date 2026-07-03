//
//  LlamaService.swift
//  PrivateAI
//
//  Created by Piotr Gorzelany on 24/01/2024.
//

import Foundation

public final actor LlamaService {

    // MARK: Properties
    private var llama: Llama?
    private var currentTask: Task<(), Error>?
    /// LLM-8: per-stream cancellation flag. Created fresh for every stream
    /// (after stopCompletion, before initializeCompletion) so a cancelled
    /// previous stream can never abort the next prefill.
    private var currentCancellationFlag: CancellationFlag?
    private let modelUrl: URL
    private let config: LlamaConfig

    // MARK: Lifecycle

    public init(modelUrl: URL, config: LlamaConfig) {
        self.modelUrl = modelUrl
        self.config = config
    }

    // MARK: Methods

    public func processMessages(_ messages: [LlamaChatMessage]) async throws {
        let llama = try initializeLlamaIfNecessary()
        await stopCompletion()
        await installFreshCancellationFlag(on: llama)
        try await llama.initializeCompletion(messages: messages, addAssistant: false)
    }

    /// Generate a typed response constrained by a JSON grammar inferred from `T` and decode it.
    /// - Parameters:
    ///   - messages: Chat messages forming the prompt.
    ///   - type: The `Codable` type to generate and decode.
    /// - Returns: A decoded instance of `T` produced by the model.
    public func respond<T: Codable>(to messages: [LlamaChatMessage], generating type: T.Type) async throws -> T {
        func extractLikelyJSON(from text: String) -> String? {
            // Find first opening brace or bracket
            guard let startIndex = text.firstIndex(where: { $0 == "{" || $0 == "[" }) else { return nil }
            let candidate = text[startIndex...]
            // Simple balance-based termination (ignores strings/escapes, good enough for LLM output)
            var depth: Int = 0
            var closingIndex: String.Index?
            for (i, ch) in candidate.enumerated() {
                let idx = candidate.index(candidate.startIndex, offsetBy: i)
                if ch == "{" || ch == "[" { depth += 1 }
                else if ch == "}" || ch == "]" {
                    depth -= 1
                    if depth == 0 { closingIndex = idx; break }
                }
            }
            if let closingIndex {
                return String(candidate[...closingIndex])
            }
            return nil
        }

        var accumulated = ""
        let decoder = JSONDecoder()
        var decodedValue: T?
        let stream = try await streamCompletion(of: messages, generating: type)
        do {
            for try await token in stream {
                accumulated += token
                if let jsonText = extractLikelyJSON(from: accumulated),
                   let data = jsonText.data(using: .utf8),
                   let value = try? decoder.decode(T.self, from: data) {
                    decodedValue = value
                    break
                }
            }
        } catch {
            // Fall through to final decode attempt below
        }
        if let value = decodedValue {
            await stopCompletion()
            return value
        }
        // Final attempt with trimmed JSON if available, otherwise full text
        let finalText = extractLikelyJSON(from: accumulated) ?? accumulated
        guard let finalData = finalText.data(using: .utf8) else {
            throw LlamaError.decodingError
        }
        return try decoder.decode(T.self, from: finalData)
    }

    /// Generate a plain text response using the provided sampling configuration.
    /// - Parameters:
    ///   - messages: Chat messages forming the prompt.
    ///   - samplingConfig: Sampling parameters controlling generation.
    /// - Returns: The full generated text.
    public func respond(to messages: [LlamaChatMessage], samplingConfig: LlamaSamplingConfig) async throws -> String {
        let stream = try await streamCompletion(of: messages, samplingConfig: samplingConfig)
        var output = ""
        for try await token in stream {
            output += token
        }
        return output
    }

    /// Generate a plain text response from a pre-formatted prompt string.
    /// Use this when the caller has already applied a chat template or
    /// the model's built-in template is not compatible.
    public func respond(text: String, samplingConfig: LlamaSamplingConfig) async throws -> String {
        let stream = try await streamCompletion(of: text, samplingConfig: samplingConfig)
        var output = ""
        for try await token in stream {
            output += token
        }
        return output
    }

    /// Stream token-by-token generation from a pre-formatted prompt string.
    public func streamCompletion(of text: String, samplingConfig: LlamaSamplingConfig) async throws -> AsyncThrowingStream<String, Error> {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LlamaError.emptyPrompt
        }
        let llama = try initializeLlamaIfNecessary()
        await stopCompletion()
        // LLM-8: the flag must be live BEFORE initializeCompletion so a
        // stopCompletion issued during the prefill aborts it (layer 2).
        let flag = await installFreshCancellationFlag(on: llama)
        try await llama.initializeCompletion(text: text)
        await llama.updateSamplingConfig(samplingConfig)
        return makeCompletionStream(llama: llama, flag: flag)
    }

    public func streamCompletion<T: Codable>(of messages: [LlamaChatMessage], generating: T.Type) async throws -> AsyncThrowingStream<String, Error> {
        // Default: constrain the output to valid JSON matching the provided type
        let grammarConfig = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: generating)
        let sampling = LlamaSamplingConfig(
            temperature: 0.1,
            seed: 42,
            grammarConfig: grammarConfig
        )
        return try await streamCompletion(of: messages, samplingConfig: sampling)
    }

    public func streamCompletion(of messages: [LlamaChatMessage], samplingConfig: LlamaSamplingConfig) async throws -> AsyncThrowingStream<String, Error> {
        guard !messages.isEmpty else { throw LlamaError.emptyMessageArray }
        let llama = try initializeLlamaIfNecessary()
        await stopCompletion()
        // LLM-8: the flag must be live BEFORE initializeCompletion so a
        // stopCompletion issued during the prefill aborts it (layer 2).
        let flag = await installFreshCancellationFlag(on: llama)
        try await llama.initializeCompletion(messages: messages)
        await llama.updateSamplingConfig(samplingConfig)
        return makeCompletionStream(llama: llama, flag: flag)
    }

    /// Shared producer for both streamCompletion overloads.
    private func makeCompletionStream(llama: Llama, flag: CancellationFlag) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task: Task<(), Error> = Task {
                do {
                    generationLoop: while await (llama.currentTokenPosition < llama.maxTokenCount) {
                        guard !Task.isCancelled else { break }
                        let result = try await llama.generateNextToken()
                        switch result {
                        case .token(let token):
                            continuation.yield(token)
                        case .endOfString:
                            break generationLoop
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            currentTask = task
            // LLM-8 layer 1: consumer termination (loop break, task cancel,
            // iterator deinit) must reach the producer — without this the
            // producer decodes to maxTokenCount into a dead buffer (F1 root
            // cause). Runs synchronously on an arbitrary thread: only the
            // thread-safe Task.cancel() / flag.cancel(), no actor calls.
            continuation.onTermination = { _ in
                task.cancel()
                flag.cancel()
            }
        }
    }

    /// LLM-8: create and install a fresh per-stream flag (+ the best-effort
    /// abort callback, layer 3, unless disabled by the test seam).
    @discardableResult
    private func installFreshCancellationFlag(on llama: Llama) async -> CancellationFlag {
        let flag = CancellationFlag()
        currentCancellationFlag = flag
        await llama.setCancellationFlag(flag)
        await llama.installAbortCallback(enabled: !disableAbortCallbackLayer)
        return flag
    }

    public func stopCompletion() async {
        // Flag first: reaches the prefill (which has no Task to cancel) at the
        // next batch boundary; the task cancel stops generation at the next
        // token boundary; cancelAndWait guarantees the producer is fully done.
        currentCancellationFlag?.cancel()
        await currentTask?.cancelAndWait()
        currentCancellationFlag = nil
        currentTask = nil
    }

    // MARK: - Test seams (LLM-8, internal — @testable only)

    /// When true, streams do not install the best-effort abort callback
    /// (layer 3), so clean-cancel tests exercise ONLY the guaranteed layers
    /// (1/2, token/batch boundary) deterministically on any backend.
    private var disableAbortCallbackLayer = false

    func _testSetDisableAbortCallbackLayer(_ enabled: Bool) {
        disableAbortCallbackLayer = enabled
    }

    /// Current token position of the underlying Llama (-1 if not initialized).
    func _testCurrentTokenPosition() async -> Int32 {
        guard let llama else { return -1 }
        return await llama.currentTokenPosition
    }

    /// Processed-token count of the underlying Llama (-1 if not initialized).
    func _testProcessedTokenCount() async -> Int {
        guard let llama else { return -1 }
        return await llama.processedTokens.count
    }

    /// Bridge to Llama's one-shot decode-abort injection (initializes the
    /// model if needed so a fresh service can inject before its first stream).
    func _testFailNextDecodeWithAborted() async {
        guard let llama = try? initializeLlamaIfNecessary() else { return }
        await llama._testFailNextDecodeWithAborted()
    }

    private func initializeLlamaIfNecessary() throws -> Llama {
        guard let llama else {
            llama = try Llama(modelPath: modelUrl.path(percentEncoded: false), config: config)
            return llama!
        }
        return llama
    }
}
