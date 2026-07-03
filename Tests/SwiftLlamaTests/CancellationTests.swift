//
//  CancellationTests.swift
//  swift-llama-cpp
//
//  LLM-8 (F1, FormAI Tier B 2026-07-03): cancellation propagation.
//  Design: FormAI docs/handoffs/2026-07-03-llm-tier-b-tdd-plan.md §LLM-8.
//
//  Stop guarantee under test = layers 1/2 only (deterministic):
//    layer 1: continuation.onTermination → producer task cancel (token boundary)
//    layer 2: CancellationFlag checked at generateNextToken head and between
//             prefill batches → LlamaError.aborted (batch boundary)
//  layer 3 (llama_set_abort_callback) is best-effort (CPU-only per llama.h) and
//  is explicitly DISABLED via _testSetDisableAbortCallbackLayer in the clean-
//  cancel tests so they stay deterministic on any backend. Its efficacy is
//  measured on device (Metal System Trace), not here.
//

import Testing
import Foundation
@testable import SwiftLlama

@Suite(.serialized)
struct CancellationTests {

    // MARK: - Helpers

    private static let genConfig = LlamaConfig(batchSize: 256, maxTokenCount: 512)

    private func makeService(config: LlamaConfig = CancellationTests.genConfig) -> LlamaService {
        LlamaService(modelUrl: .llama1B, config: config)
    }

    private func samplingConfig() -> LlamaSamplingConfig {
        LlamaSamplingConfig(temperature: 0.7, seed: 42)
    }

    /// Chat-template story request — reliably generates hundreds of tokens
    /// (same shape as LlamaServiceTests' performance prompt). Raw-text prompts
    /// hit EOS after a few tokens on this model, which would make "producer
    /// stopped" indistinguishable from "producer finished".
    private let storyMessages = [
        LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
        LlamaChatMessage(role: .user, content: "Tell me a very long and detailed story about mars colonization.")
    ]

    // MARK: - 8-a: consumer cancel reaches the producer (layer 1)

    /// Actor-based token counter shared with the consumer task.
    private actor TokenCounter {
        private(set) var count = 0
        func increment() { count += 1 }
    }

    /// 8-a: cancelling the CONSUMER task after 3 tokens must stop the producer
    /// at the next token boundary (task cancel → iterator cancellation →
    /// onTermination → producer cancel). Current impl has no `onTermination`,
    /// so the producer keeps decoding to maxTokenCount → position keeps
    /// growing → fail. (Note: a bare `break` out of for-await only drops the
    /// iterator and does NOT terminate the stream — the app's real stop signal
    /// is `llmTask?.cancel()`, which is what this models.)
    @Test func consumerCancelStopsProducer() async throws {
        let service = makeService()
        await service._testSetDisableAbortCallbackLayer(true)
        defer { Task { await service._testSetDisableAbortCallbackLayer(false) } }

        let counter = TokenCounter()
        let consumer = Task {
            let stream = try await service.streamCompletion(of: storyMessages, samplingConfig: samplingConfig())
            for try await _ in stream {
                await counter.increment()
            }
        }
        // Wait until generation is demonstrably running, then cancel the consumer.
        while await counter.count < 3 {
            try await Task.sleep(for: .milliseconds(50))
        }
        consumer.cancel()
        _ = try? await consumer.value

        // Let any in-flight token finish, then sample the position twice.
        try await Task.sleep(for: .milliseconds(500))
        let p1 = await service._testCurrentTokenPosition()
        try await Task.sleep(for: .seconds(1))
        let p2 = await service._testCurrentTokenPosition()
        #expect(p1 == p2, "producer is still generating after consumer cancel (p1=\(p1) p2=\(p2))")
    }

    // MARK: - 8-b: stopCompletion during prefill (layer 2)

    /// 8-b: stopCompletion during a long-prompt prefill must abort at the next
    /// batch boundary — streamCompletion throws LlamaError.aborted and no token
    /// is ever yielded. Current impl has zero stop mechanism during prefill, so
    /// the call completes normally → fail. (No reset assertion: a batch-boundary
    /// flag throw is a CLEAN abort — no decode failed.)
    @Test func stopDuringPrefillAborts() async throws {
        let config = LlamaConfig(batchSize: 64, maxTokenCount: 8192)
        let service = makeService(config: config)
        // ~5600 tokens — with batchSize 64 that is ~90 batch boundaries.
        let longPrompt = String(
            repeating: "The quick brown fox jumps over the lazy dog near the river bank. ",
            count: 400
        )
        let task = Task {
            try await service.respond(text: longPrompt, samplingConfig: samplingConfig())
        }
        // Land inside the prefill window (model load + prefill take seconds).
        try await Task.sleep(for: .milliseconds(300))
        await service.stopCompletion()
        do {
            let text = try await task.value
            Issue.record("stopCompletion during prefill did not abort (returned \(text.count) chars)")
        } catch LlamaError.aborted {
            // expected
        } catch {
            Issue.record("expected LlamaError.aborted, got \(error)")
        }
    }

    // MARK: - 8-c1: clean cancel preserves prefix cache (no reset)

    /// 8-c1: a clean token-boundary cancel must NOT reset state — position and
    /// processed-token count stay > 0 (direct proof, sampled BEFORE the next
    /// stream), and the next stream with the same prompt completes via
    /// prefix-reuse.
    @Test func cleanCancelPreservesPrefixCache() async throws {
        let service = makeService()
        await service._testSetDisableAbortCallbackLayer(true)
        defer { Task { await service._testSetDisableAbortCallbackLayer(false) } }

        let stream = try await service.streamCompletion(of: storyMessages, samplingConfig: samplingConfig())
        var count = 0
        for try await _ in stream {
            count += 1
            if count >= 3 { break }
        }
        await service.stopCompletion()

        let position = await service._testCurrentTokenPosition()
        let processed = await service._testProcessedTokenCount()
        #expect(position > 0, "clean cancel must not reset currentTokenPosition")
        #expect(processed > 0, "clean cancel must not reset processedTokens")

        // Same prompt again → prefix-reuse path must produce a full completion.
        let stream2 = try await service.streamCompletion(of: storyMessages, samplingConfig: samplingConfig())
        var count2 = 0
        for try await _ in stream2 { count2 += 1 }
        #expect(count2 > 0, "prefix-reuse stream after clean cancel yielded no tokens")
    }

    // MARK: - 8-c2: dirty abort resets state and recovers

    /// 8-c2: a decode-site abort (one-shot injection) must reset state
    /// IMMEDIATELY at the throw site (position/count == 0 right after), and the
    /// same service must recover with a full reprocess on the next stream.
    @Test func dirtyAbortResetsStateAndRecovers() async throws {
        let service = makeService(config: LlamaConfig(batchSize: 256, maxTokenCount: 256))
        // Layer 3 disabled: on backends where the abort callback IS effective
        // (it is on this Mac), the setup-phase stopCompletion would abort
        // mid-decode and dirty-reset — this test targets the INJECTED dirty
        // path, so the setup stop must stay clean (Codex R4-M1).
        await service._testSetDisableAbortCallbackLayer(true)
        defer { Task { await service._testSetDisableAbortCallbackLayer(false) } }
        // Build up real state first.
        let stream = try await service.streamCompletion(of: storyMessages, samplingConfig: samplingConfig())
        var count = 0
        for try await _ in stream {
            count += 1
            if count >= 3 { break }
        }
        await service.stopCompletion()
        #expect(await service._testProcessedTokenCount() > 0)

        await service._testFailNextDecodeWithAborted() // one-shot injection
        let divergedMessages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
            LlamaChatMessage(role: .user, content: "Tell me a short emotional poem about mars.")
        ]
        do {
            let s = try await service.streamCompletion(of: divergedMessages, samplingConfig: samplingConfig())
            for try await _ in s {}
            Issue.record("injected decode abort was swallowed (stream completed)")
        } catch LlamaError.aborted {
            // expected
        } catch {
            Issue.record("expected LlamaError.aborted, got \(error)")
        }

        // Dirty reset happened at the throw site, before anything else ran.
        #expect(await service._testCurrentTokenPosition() == 0, "dirty abort must reset position to 0")
        #expect(await service._testProcessedTokenCount() == 0, "dirty abort must clear processedTokens")

        // Full reprocess recovers on the same service.
        let s2 = try await service.streamCompletion(of: storyMessages, samplingConfig: samplingConfig())
        var count2 = 0
        for try await _ in s2 { count2 += 1 }
        #expect(count2 > 0, "recovery stream after dirty reset yielded no tokens")
    }

    // MARK: - 8-d: normal completion regression

    /// 8-d: with no cancel at all, generation runs to completion — twice on the
    /// same service (no flag pollution across streams, onTermination does not
    /// break the normal path).
    @Test func normalCompletionUnaffected() async throws {
        let service = makeService(config: LlamaConfig(batchSize: 256, maxTokenCount: 160))
        let text = try await service.respond(text: "Say hello in one short sentence.", samplingConfig: samplingConfig())
        #expect(!text.isEmpty)
        let text2 = try await service.respond(text: "Say goodbye in one short sentence.", samplingConfig: samplingConfig())
        #expect(!text2.isEmpty)
    }

    // MARK: - 8-e: aborted is not converted to decodingError

    /// 8-e: the injected abort must surface as LlamaError.aborted through
    /// processBatch / optimizedReprocessing / the producer — never converted to
    /// decodingError and never swallowed by the optimization fallback.
    @Test func injectedAbortThrowsAbortedNotDecodingError() async throws {
        let service = makeService(config: LlamaConfig(batchSize: 256, maxTokenCount: 256))
        await service._testFailNextDecodeWithAborted()
        do {
            _ = try await service.respond(text: "Say hello.", samplingConfig: samplingConfig())
            Issue.record("injected decode abort did not fire")
        } catch LlamaError.aborted {
            // expected
        } catch {
            Issue.record("aborted was converted to \(error) (processBatch/fallback must rethrow it)")
        }
    }
}
