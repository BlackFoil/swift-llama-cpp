import Testing
@testable import SwiftLlama

/// N-x29 — a throwing `Llama.init` leaked one backend initialization.
///
/// `Llama.init` called `llama_backend_init()` first thing and freed it in
/// `deinit`. Swift does not run `deinit` for an initializer that throws, so
/// every failed load — a corrupt gguf, a download that stopped early — left one
/// initialization behind for the life of the process. FormAI retries loads, so
/// the count grew with each attempt.
///
/// The same pairing was wrong in the other direction: two overlapping `Llama`
/// instances (one warming up while the previous still answers) meant the first
/// deinit freed a backend the other was still using.
/// `.serialized`: 参照カウントはプロセス全体で 1 つなので、並列実行すると
/// テスト同士の retain/release が混ざって値が壊れる(実際に壊れた)。
@Suite("LlamaBackend retain/release", .serialized)
struct LlamaBackendRetainTests {

    @Test("release without a matching retain does not underflow")
    func releaseBelowZeroIsIgnored() {
        let before = LlamaBackend._testRetainCount
        LlamaBackend.release()
        LlamaBackend.release()
        #expect(LlamaBackend._testRetainCount >= 0)
        _ = before
    }

    @Test("balanced retain/release returns to the starting count")
    func balancedPairsBalance() {
        let before = LlamaBackend._testRetainCount
        LlamaBackend.retain()
        #expect(LlamaBackend._testRetainCount == before + 1)
        LlamaBackend.release()
        #expect(LlamaBackend._testRetainCount == before)
    }

    /// The overlap case: two live users, one goes away, the backend stays up
    /// for the other.
    @Test("the backend survives while a second user still holds it")
    func nestedRetainsKeepTheBackendAlive() {
        let before = LlamaBackend._testRetainCount
        LlamaBackend.retain()
        LlamaBackend.retain()
        #expect(LlamaBackend._testRetainCount == before + 2)
        LlamaBackend.release()
        #expect(LlamaBackend._testRetainCount == before + 1)
        LlamaBackend.release()
        #expect(LlamaBackend._testRetainCount == before)
    }

    /// The leak itself: a `Llama.init` that throws must leave the count where
    /// it found it. A missing model path fails at the first guard, which is the
    /// path a corrupt or absent gguf takes.
    @Test("a failed initialization leaves no backend behind")
    func throwingInitDoesNotLeak() async {
        let before = LlamaBackend._testRetainCount
        for _ in 0..<5 {
            _ = try? Llama(
                modelPath: "/nonexistent/definitely-not-a-model.gguf",
                config: LlamaConfig(batchSize: 8, maxTokenCount: 16))
        }
        #expect(
            LlamaBackend._testRetainCount == before,
            "init が throw するたびに backend の初期化が 1 回ずつ残っている"
        )
    }
}
