import Foundation

/// A JSON value used by chat schemas, tool arguments, and tool results.
public indirect enum ChatJSONValue: Codable, Sendable, Equatable {
    case object([String: ChatJSONValue]), array([ChatJSONValue]), string(String)
    case number(Double), bool(Bool), null

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Double.self) { self = .number(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([ChatJSONValue].self) { self = .array(v) }
        else { self = .object(try c.decode([String: ChatJSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .object(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .null: try c.encodeNil()
        }
    }
}

public struct Gemma4Tool: Codable, Sendable, Equatable {
    public let name: String
    public let description: String
    public let parameters: ChatJSONValue

    public init(name: String, description: String, parameters: ChatJSONValue) {
        self.name = name; self.description = description; self.parameters = parameters
    }
}

public struct Gemma4ToolCall: Codable, Sendable, Equatable {
    public let id: String
    public let name: String
    public let arguments: ChatJSONValue

    public init(id: String = UUID().uuidString, name: String, arguments: ChatJSONValue) {
        self.id = id; self.name = name; self.arguments = arguments
    }
}

/// Complete role history. Tool results reference an earlier call's ID.
public struct Gemma4ConversationMessage: Codable, Sendable, Equatable {
    public enum Role: String, Codable, Sendable { case system, user, assistant, tool }
    public let role: Role
    public let content: String?
    public let toolCalls: [Gemma4ToolCall]?
    public let toolCallID: String?

    public init(role: Role, content: String? = nil,
                toolCalls: [Gemma4ToolCall]? = nil, toolCallID: String? = nil) {
        self.role = role; self.content = content
        self.toolCalls = toolCalls; self.toolCallID = toolCallID
    }
}

public struct Gemma4Response: Codable, Sendable, Equatable {
    public let content: String
    public let toolCalls: [Gemma4ToolCall]
    public let generatedTokens: Int

    public init(content: String, toolCalls: [Gemma4ToolCall], generatedTokens: Int) {
        self.content = content; self.toolCalls = toolCalls; self.generatedTokens = generatedTokens
    }
}

public enum Gemma4ToolError: Error, LocalizedError {
    case malformed(String)
    public var errorDescription: String? {
        switch self { case .malformed(let detail): return "Invalid Gemma tool data: \(detail)" }
    }
}

/// Gemma 4's native declarations, call blocks, and in-turn tool responses.
public enum Gemma4ToolCodec {
    private static let quote = "<|\"|>"

    static func value(_ value: ChatJSONValue) -> String {
        switch value {
        case .string(let s): return quote + s + quote
        case .number(let n):
            return n.isFinite && n.rounded() == n && abs(n) < Double(Int64.max)
                ? String(Int64(n)) : String(n)
        case .bool(let b): return b ? "true" : "false"
        case .null: return "null"
        case .array(let a): return "[" + a.map(Self.value).joined(separator: ",") + "]"
        case .object(let o):
            return "{" + o.keys.sorted().map { $0 + ":" + Self.value(o[$0]!) }.joined(separator: ",") + "}"
        }
    }

    private static func schema(_ value: ChatJSONValue) -> ChatJSONValue {
        func resolve(_ node: ChatJSONValue, seen: Set<String> = []) -> ChatJSONValue {
            switch node {
            case .array(let a): return .array(a.map { resolve($0, seen: seen) })
            case .object(var o):
                var activeReferences = seen
                if case .string(let reference) = o["$ref"], reference.hasPrefix("#/"), !seen.contains(reference) {
                    var target: ChatJSONValue? = value
                    for component in reference.dropFirst(2).split(separator: "/") {
                        let key = String(component).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
                        if case .object(let fields) = target { target = fields[key] } else { target = nil }
                    }
                    if let target, case .object(let fields) = resolve(target, seen: seen.union([reference])) {
                        activeReferences.insert(reference)
                        o.removeValue(forKey: "$ref")
                        o.merge(fields) { current, _ in current }
                    }
                }
                o.removeValue(forKey: "$defs")
                o.removeValue(forKey: "definitions")
            if case .array(let choices) = o["anyOf"] {
                let nonNull = choices.filter {
                    if case .object(let branch) = $0 { return branch["type"] != .string("null") }
                    return true
                }
                if nonNull.count == 1, case .object(let branch) = nonNull[0] {
                    o.removeValue(forKey: "anyOf")
                    o.merge(branch) { current, _ in current }
                    if choices.count != nonNull.count { o["nullable"] = .bool(true) }
                }
            }
            return .object(o.mapValues { resolve($0, seen: activeReferences) }.merging(
                o["type"].map { ["type": uppercaseType($0)] } ?? [:]
            ) { _, type in type })
            default: return node
            }
        }
        return resolve(value)
    }

    private static func uppercaseType(_ value: ChatJSONValue) -> ChatJSONValue {
        if case .string(let s) = value { return .string(s.uppercased()) }
        if case .array(let a) = value { return .array(a.map(uppercaseType)) }
        return value
    }

    /// Render a complete conversation, keeping tool responses inside the model turn.
    public static func prompt(messages: [Gemma4ConversationMessage], tools: [Gemma4Tool] = []) throws -> String {
        try prompt(messages: messages, tools: tools, preserveToolResponseText: false)
    }

    /// Select text tool responses for checkpoints whose chat template wraps role:tool content verbatim.
    static func prompt(messages: [Gemma4ConversationMessage], tools: [Gemma4Tool],
                       preserveToolResponseText: Bool) throws -> String {
        var text = "<bos>"
        var start = 0
        let system = messages.first?.role == .system ? messages.first?.content ?? "" : ""
        if messages.first?.role == .system { start = 1 }
        if !system.isEmpty || !tools.isEmpty {
            text += "<|turn>system\n" + system
            for tool in tools {
                text += "<|tool>declaration:" + tool.name + "{description:" + value(.string(tool.description))
                text += ",parameters:" + value(schema(tool.parameters)) + "}<tool|>"
            }
            text += "<turn|>\n"
        }
        var index = start
        var modelTurnOpen = false
        while index < messages.count {
            let message = messages[index]
            guard message.role != .tool else { throw Gemma4ToolError.malformed("orphan tool response") }
            let role = message.role == .assistant ? "model" : message.role.rawValue
            if modelTurnOpen && message.role != .assistant {
                text += "<turn|>\n"
            }
            if !(message.role == .assistant && modelTurnOpen) { text += "<|turn>" + role + "\n" }
            modelTurnOpen = false
            let calls = message.toolCalls ?? []
            if !calls.isEmpty {
                guard message.role == .assistant else { throw Gemma4ToolError.malformed("tool calls require assistant role") }
                for call in calls {
                    text += "<|tool_call>call:" + call.name + value(call.arguments) + "<tool_call|>"
                }
                var seen = Set<String>()
                while index + 1 < messages.count, messages[index + 1].role == .tool {
                    index += 1
                    let result = messages[index]
                    guard let id = result.toolCallID, let call = calls.first(where: { $0.id == id }), seen.insert(id).inserted else {
                        throw Gemma4ToolError.malformed("unmatched or duplicate tool response")
                    }
                    let body = result.content ?? ""
                    let payload: ChatJSONValue
                    if preserveToolResponseText {
                        payload = .object(["value": .string(body)])
                    } else {
                        let parsed = (try? JSONDecoder().decode(ChatJSONValue.self, from: Data(body.utf8))) ?? .string(body)
                        if case .object = parsed { payload = parsed } else { payload = .object(["value": parsed]) }
                    }
                    text += "<|tool_response>response:" + call.name + value(payload) + "<tool_response|>"
                }
                guard seen.count == calls.count else { throw Gemma4ToolError.malformed("missing tool results") }
                modelTurnOpen = true
            }
            text += message.content ?? ""
            if !(modelTurnOpen && (message.content ?? "").isEmpty) {
                text += "<turn|>\n"; modelTurnOpen = false
            }
            index += 1
        }
        if !modelTurnOpen { text += "<|turn>model\n" }
        return text
    }

    /// Parse all native call blocks without evaluating generated code.
    public static func response(_ raw: String, generatedTokens: Int = 0) throws -> Gemma4Response {
        try response(raw, generatedTokens: generatedTokens, allowRepeatedToolTerminators: false)
    }

    /// Some checkpoints repeat adjacent closing markers after an otherwise complete call.
    static func response(_ raw: String, generatedTokens: Int,
                         allowRepeatedToolTerminators: Bool) throws -> Gemma4Response {
        var text = raw
        while let open = text.range(of: "<|channel>") {
            guard let close = text.range(of: "<channel|>", range: open.upperBound..<text.endIndex) else {
                throw Gemma4ToolError.malformed("unfinished reasoning channel")
            }
            text.removeSubrange(open.lowerBound..<close.upperBound)
        }
        var calls: [Gemma4ToolCall] = []
        while let start = text.range(of: "<|tool_call>") {
            guard let end = text.range(of: "<tool_call|>", range: start.upperBound..<text.endIndex) else {
                throw Gemma4ToolError.malformed("unfinished tool call")
            }
            let body = String(text[start.upperBound..<end.lowerBound])
            guard body.hasPrefix("call:"), let brace = body.firstIndex(of: "{") else {
                throw Gemma4ToolError.malformed("missing call name or arguments")
            }
            let renderedName = String(body.dropFirst(5).prefix(upTo: brace)).trimmingCharacters(in: .whitespacesAndNewlines)
            let name: String
            if renderedName.hasPrefix(quote) || renderedName.hasPrefix("\"") {
                var nameParser = Parser(renderedName)
                guard case .string(let parsed) = try nameParser.parse() else {
                    throw Gemma4ToolError.malformed("call name must be a string")
                }
                name = parsed
            } else { name = renderedName }
            guard !name.isEmpty else { throw Gemma4ToolError.malformed("empty call name") }
            var parser = Parser(String(body[brace...]))
            let arguments = try parser.parse()
            guard case .object = arguments else { throw Gemma4ToolError.malformed("arguments must be an object") }
            calls.append(.init(name: name, arguments: arguments))
            var consumedEnd = end.upperBound
            if allowRepeatedToolTerminators {
                while text[consumedEnd...].hasPrefix("<tool_call|>") {
                    consumedEnd = text.index(consumedEnd, offsetBy: "<tool_call|>".count)
                }
            }
            text.removeSubrange(start.lowerBound..<consumedEnd)
        }
        guard !text.contains("<tool_call|>") else {
            throw Gemma4ToolError.malformed("orphan tool call terminator")
        }
        for marker in ["<eos>", "<turn|>", "<channel|>", "<|tool_response>", "<|think|>"] {
            text = text.replacingOccurrences(of: marker, with: "")
        }
        return .init(content: text.trimmingCharacters(in: .whitespacesAndNewlines), toolCalls: calls, generatedTokens: generatedTokens)
    }

    private struct Parser {
        let text: String
        var index: String.Index
        init(_ text: String) { self.text = text; self.index = text.startIndex }
        mutating func skipSpace() { while index < text.endIndex && text[index].isWhitespace { index = text.index(after: index) } }
        mutating func consume(_ token: String) -> Bool {
            skipSpace()
            guard text[index...].hasPrefix(token) else { return false }
            index = text.index(index, offsetBy: token.count); return true
        }
        mutating func parse() throws -> ChatJSONValue {
            let v = try parseValue(depth: 0); skipSpace()
            guard index == text.endIndex else { throw Gemma4ToolError.malformed("trailing argument text") }
            return v
        }
        mutating func parseString() throws -> String {
            skipSpace()
            if consume(Gemma4ToolCodec.quote) {
                guard let end = text.range(of: Gemma4ToolCodec.quote, range: index..<text.endIndex) else { throw Gemma4ToolError.malformed("unterminated string") }
                let result = String(text[index..<end.lowerBound]); index = end.upperBound; return result
            }
            if index < text.endIndex, text[index] == "\"" {
                let start = index; index = text.index(after: index)
                var escaped = false
                while index < text.endIndex {
                    let c = text[index]; index = text.index(after: index)
                    if c == "\"" && !escaped {
                        return try JSONDecoder().decode(String.self, from: Data(text[start..<index].utf8))
                    }
                    if c == "\\" && !escaped { escaped = true } else { escaped = false }
                }
            }
            throw Gemma4ToolError.malformed("expected string")
        }
        mutating func parseValue(depth: Int) throws -> ChatJSONValue {
            guard depth < 64 else { throw Gemma4ToolError.malformed("arguments too deeply nested") }
            skipSpace()
            if consume("{") {
                var object: [String: ChatJSONValue] = [:]
                if consume("}") { return .object(object) }
                repeat {
                    skipSpace()
                    let key: String
                    if text[index...].hasPrefix(Gemma4ToolCodec.quote) || (index < text.endIndex && text[index] == "\"") { key = try parseString() }
                    else {
                        let start = index
                        while index < text.endIndex && !":,{}[]".contains(text[index]) { index = text.index(after: index) }
                        key = String(text[start..<index]).trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    guard !key.isEmpty, object[key] == nil, consume(":") else { throw Gemma4ToolError.malformed("invalid object key") }
                    object[key] = try parseValue(depth: depth + 1)
                    if consume("}") { return .object(object) }
                } while consume(",")
                throw Gemma4ToolError.malformed("expected object separator")
            }
            if consume("[") {
                var array: [ChatJSONValue] = []
                if consume("]") { return .array(array) }
                repeat {
                    array.append(try parseValue(depth: depth + 1))
                    if consume("]") { return .array(array) }
                } while consume(",")
                throw Gemma4ToolError.malformed("expected array separator")
            }
            if text[index...].hasPrefix(Gemma4ToolCodec.quote) || (index < text.endIndex && text[index] == "\"") { return .string(try parseString()) }
            let start = index
            while index < text.endIndex && !",]}".contains(text[index]) && !text[index].isWhitespace { index = text.index(after: index) }
            let literal = String(text[start..<index])
            if literal == "true" { return .bool(true) }
            if literal == "false" { return .bool(false) }
            if literal == "null" { return .null }
            if let number = Double(literal), number.isFinite { return .number(number) }
            throw Gemma4ToolError.malformed("invalid literal")
        }
    }
}
