//
//  LlamaError.swift
//  swift-llama-cpp
//
//  Created by Piotr Gorzelany on 30/07/2025.
//

public enum LlamaError: Error{
    case couldNotInitializeContext
    case contextSizeLimitExeeded
    case decodingError
    case emptyMessageArray
    case emptyPrompt
    /// Generation/prefill was cancelled (CancellationFlag, abort callback, or
    /// llama_decode returning 2). Distinct from decodingError so callers can
    /// treat a user-initiated stop as non-fatal.
    case aborted
}
