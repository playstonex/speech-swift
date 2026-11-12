import Foundation
import Qwen3Chat

/// Per-request storage; a completion callback can run on the inference executor.
final class NativeInferenceGenerationCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Gemma4GenerationTrace?

    func store(_ trace: Gemma4GenerationTrace) { lock.withLock { value = trace } }
    var trace: Gemma4GenerationTrace? { lock.withLock { value } }
}

/// Explicitly enabled local failure records. Raw content never enters HTTP errors.
struct NativeInferenceFailures: Sendable {
    private let directory: URL

    init(directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        let attributes = try FileManager.default.attributesOfItem(atPath: directory.path)
        guard let permissions = attributes[.posixPermissions] as? Int, permissions & 0o077 == 0 else {
            throw CocoaError(.fileWriteNoPermission, userInfo: [NSFilePathErrorKey: directory.path])
        }
        self.directory = directory
    }

    func write(requestBody: Data, requestID: String, model: String, runtimeRevision: String,
               error: Error, generation: Gemma4GenerationTrace?) throws -> URL {
        struct Record: Encodable {
            let schemaVersion = 1
            let requestID: String
            let model: String
            let runtimeRevision: String
            let requestBody: String?
            let requestBodyBase64: String
            let error: String
            let generation: Gemma4GenerationTrace?
        }
        let record = Record(requestID: requestID, model: model, runtimeRevision: runtimeRevision,
                            requestBody: String(data: requestBody, encoding: .utf8), requestBodyBase64: requestBody.base64EncodedString(),
                            error: String(describing: error), generation: generation)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        // Request IDs are metadata, never filesystem paths.
        let file = directory.appendingPathComponent(UUID().uuidString + ".json")
        try encoder.encode(record).write(to: file, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        return file
    }
}
