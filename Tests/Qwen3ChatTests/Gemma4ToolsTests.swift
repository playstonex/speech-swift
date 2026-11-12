import XCTest
@testable import Qwen3Chat

final class Gemma4ToolsTests: XCTestCase {
    func testRepeatedAdjacentTerminatorsRequireExplicitCheckpointCompatibility() throws {
        let raw = "<|tool_call>call:read_record{id:<|\"|>rec_3<|\"|>}<tool_call|><tool_call|>"
        XCTAssertThrowsError(try Gemma4ToolCodec.response(raw))
        let reply = try Gemma4ToolCodec.response(raw, generatedTokens: 23, allowRepeatedToolTerminators: true)
        XCTAssertEqual(reply.toolCalls.count, 1)
        XCTAssertEqual(reply.toolCalls[0].name, "read_record")
        XCTAssertEqual(reply.toolCalls[0].arguments, .object(["id": .string("rec_3")]))
        XCTAssertEqual(reply.generatedTokens, 23)
        XCTAssertEqual(reply.content, "")
    }

    func testRepeatedTerminatorsPreserveMultipleValidatedCallsAndStringValues() throws {
        let raw = "<|tool_call>call:first{text:<|\"|>a,b}: café<|\"|>}<tool_call|><tool_call|><tool_call|><|tool_call>call:second{ok:true}<tool_call|><tool_call|>"
        let reply = try Gemma4ToolCodec.response(raw, generatedTokens: 0, allowRepeatedToolTerminators: true)
        XCTAssertEqual(reply.toolCalls.map(\.name), ["first", "second"])
        XCTAssertEqual(reply.toolCalls[0].arguments, .object(["text": .string("a,b}: café")]))
        XCTAssertEqual(reply.toolCalls[1].arguments, .object(["ok": .bool(true)]))
        XCTAssertEqual(reply.content, "")
    }

    func testCheckpointCompatibilityStillRejectsOrphansAndInvalidArguments() throws {
        let valid = "<|tool_call>call:read_record{}<tool_call|>"
        for raw in ["<tool_call|>", "done<tool_call|>", "<tool_call|>" + valid,
                    valid + "}<tool_call|>", valid + " <tool_call|>",
                    "<|tool_call>call:read_record{id:rec_3}<tool_call|><tool_call|>",
                    "<|tool_call>call:read_record{id:<|\"|>unfinished}<tool_call|><tool_call|>",
                    "<|tool_call>call:read_record{"] {
            XCTAssertThrowsError(try Gemma4ToolCodec.response(raw, generatedTokens: 0, allowRepeatedToolTerminators: true), raw)
        }
    }

    func testOrphanToolCallTerminatorIsRejectedBeforeSpeech() throws {
        for raw in ["}<tool_call|>", "}<tool_call|>}<tool_call|>",
                    "<|tool_call>call:lookup{}<tool_call|>}<tool_call|>"] {
            XCTAssertThrowsError(try Gemma4ToolCodec.response(raw))
        }
    }

    func testChannelCloserAfterToolResultIsRemovedFromContentOnly() throws {
        let reply = try Gemma4ToolCodec.response("<channel|>I have found your reservation.<turn|>")
        XCTAssertEqual(reply.content, "I have found your reservation.")
        let call = try Gemma4ToolCodec.response("<channel|><|tool_call>call:record_note{text:<|\"|>literal <channel|> value<|\"|>}<tool_call|>")
        XCTAssertEqual(call.content, "")
        XCTAssertEqual(call.toolCalls.first?.arguments, .object(["text": .string("literal <channel|> value")]))
        XCTAssertThrowsError(try Gemma4ToolCodec.response("<|channel>thought\nunfinished"))
    }

    func testQuotedFunctionNamesAreDecodedBeforeToolDispatch() throws {
        for name in ["<|\"|>get_users<|\"|>", "\"get_users\""] {
            let reply = try Gemma4ToolCodec.response("<|tool_call>call:\(name){}<tool_call|>")
            XCTAssertEqual(reply.toolCalls.first?.name, "get_users")
            XCTAssertEqual(reply.toolCalls.first?.arguments, .object([:]))
        }
        XCTAssertThrowsError(try Gemma4ToolCodec.response("<|tool_call>call:<|\"|>get_users{}<tool_call|>"))
    }
    func testParsesNestedArgumentsWithoutTreatingStringPunctuationAsSyntax() throws {
        let response = try Gemma4ToolCodec.response("""
        <|channel>thought\nChoose a tool.\n<channel|>
        <|tool_call>call:update{ok:false,items:[1,null,{label:<|"|>a,b}: café<|"|>}],amount:-1.25e2}<tool_call|>
        """)
        XCTAssertEqual(response.toolCalls.count, 1)
        XCTAssertEqual(response.toolCalls[0].name, "update")
        XCTAssertEqual(response.toolCalls[0].arguments, .object([
            "ok": .bool(false), "amount": .number(-125),
            "items": .array([.number(1), .null, .object(["label": .string("a,b}: café")])])]))
        XCTAssertEqual(response.content, "")
    }

    func testMultipleCallsAndMalformedArguments() throws {
        let result = try Gemma4ToolCodec.response("<|tool_call>call:first{}<tool_call|><|tool_call>call:second{x:true}<tool_call|>")
        XCTAssertEqual(result.toolCalls.map(\.name), ["first", "second"])
        for raw in ["<|tool_call>call:broken{x:}<tool_call|>", "<|tool_call>call:broken{}",
                    "<|tool_call>call:broken{x:1,x:2}<tool_call|>"] {
            XCTAssertThrowsError(try Gemma4ToolCodec.response(raw))
        }
    }

    func testToolResultsContinueTheSameModelTurn() throws {
        let call = Gemma4ToolCall(id: "weather-1", name: "weather", arguments: .object(["city": .string("Tokyo")]))
        let prompt = try Gemma4ToolCodec.prompt(messages: [
            .init(role: .system, content: "Answer using tools."),
            .init(role: .user, content: "Weather in Tokyo?"),
            .init(role: .assistant, toolCalls: [call]),
            .init(role: .tool, content: "{\"temperature\":15}", toolCallID: "weather-1")
        ])
        XCTAssertTrue(prompt.hasSuffix("<|tool_response>response:weather{temperature:15}<tool_response|>"))
        XCTAssertEqual(prompt.components(separatedBy: "<|turn>model\n").count - 1, 1)
        XCTAssertFalse(prompt.contains("<|turn>tool"))
    }

    func testOrphanAndMissingToolResultsAreRejected() throws {
        XCTAssertThrowsError(try Gemma4ToolCodec.prompt(messages: [.init(role: .tool, content: "orphan", toolCallID: "missing")]))
        XCTAssertThrowsError(try Gemma4ToolCodec.prompt(messages: [
            .init(role: .assistant, toolCalls: [.init(id: "call", name: "weather", arguments: .object([:]))])
        ]))
    }

    func testDeclarationsUseNativeTokensAndUppercaseSchemaTypes() throws {
        let tool = Gemma4Tool(name: "weather", description: "Current weather", parameters: .object([
            "type": .string("object"), "properties": .object(["city": .object(["type": .string("string")])]),
            "required": .array([.string("city")])]))
        let prompt = try Gemma4ToolCodec.prompt(messages: [.init(role: .user, content: "Tokyo?")], tools: [tool])
        XCTAssertTrue(prompt.contains("<|tool>declaration:weather"))
        XCTAssertTrue(prompt.contains("type:<|\"|>OBJECT<|\"|>"))
        XCTAssertTrue(prompt.contains("city:{type:<|\"|>STRING<|\"|>}"))
    }

    func testPydanticReferencesAreExpandedInNestedToolSchemas() throws {
        let schema: ChatJSONValue = .object([
            "type": .string("object"),
            "$defs": .object(["Cabin": .object(["type": .string("string"), "enum": .array([.string("economy"), .string("business")])])]),
            "properties": .object(["cabin": .object(["$ref": .string("#/$defs/Cabin")])])])
        let prompt = try Gemma4ToolCodec.prompt(messages: [.init(role: .user, content: "Book economy")],
                                               tools: [.init(name: "book", description: "Book a flight", parameters: schema)])
        XCTAssertFalse(prompt.contains("$ref"))
        XCTAssertFalse(prompt.contains("$defs"))
        XCTAssertTrue(prompt.contains("cabin:{enum:[<|\"|>economy<|\"|>,<|\"|>business<|\"|>],type:<|\"|>STRING<|\"|>}"))
    }
}
