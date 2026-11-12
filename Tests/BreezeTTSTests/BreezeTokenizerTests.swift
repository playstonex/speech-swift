@testable import BreezeTTS
import Hub
import Tokenizers
import XCTest

final class BreezeTokenizerTests: XCTestCase {
    private func fixture() throws -> (Config, Config) {
        let config = try JSONDecoder().decode(Config.self, from: Data(#"{"tokenizer_class":"PreTrainedTokenizer","unk_token":"<unk>","bos_token":"<bos>"}"#.utf8))
        let data = try JSONDecoder().decode(Config.self, from: Data(#"""
        {"model":{"type":"BPE","vocab":{"<unk>":0,"<bos>":1,"a":2,"b":3,"c":4," ":5},"merges":[]},
         "added_tokens":[
           {"id":0,"content":"<unk>","special":true},
           {"id":1,"content":"<bos>","special":true},
           {"id":6,"content":"[S0]","special":false},
           {"id":7,"content":"<ins_bos>","special":false},
           {"id":8,"content":"<ins_eos>","special":false},
           {"id":9,"content":"<unused1>","special":false,"lstrip":true},
           {"id":10,"content":"<unused12>","special":false,"rstrip":true}],
         "post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<bos>":{"id":"<bos>","ids":[1],"tokens":["<bos>"]}}}}
        """#.utf8))
        return (config, data)
    }

    func testOrdinaryTextKeepsSpecialTokensAndDoesNotLoadFallback() throws {
        let (config, data) = try fixture()
        let original = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
        let tokenizer = try BreezeTextTokenizer(config: config, data: data) {
            XCTFail("Ordinary text must use the compact tokenizer")
            return original
        }
        for text in ["abc", "[S0]abc", "[S0]<ins_bos>ab<ins_eos>c", "a\n\nb", "a  b", "你好，Breeze!", "e\u{301}😀"] {
            XCTAssertEqual(try tokenizer.encode(text), original.encode(text: text), text)
        }
        XCTAssertEqual(try tokenizer.encode("[S0]abc"), [1, 6, 2, 3, 4])
    }

    func testLiteralPlaceholdersPreserveOriginalRulesAndLoadFallbackOnce() throws {
        let (config, data) = try fixture()
        let original = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
        var loads = 0
        let tokenizer = try BreezeTextTokenizer(config: config, data: data) { loads += 1; return original }
        for text in ["a  <unused1>b", "<unused12>  c", "<unused1><unused12>", "[S0]<unused999>abc", "<unused", "abc"] {
            XCTAssertEqual(try tokenizer.encode(text), original.encode(text: text), text)
        }
        XCTAssertEqual(loads, 1)
    }

    func testFallbackFailureIsPropagatedAndCanBeRetried() throws {
        enum Failure: Error { case unavailable }
        let (config, data) = try fixture()
        let original = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
        var loads = 0
        let tokenizer = try BreezeTextTokenizer(config: config, data: data) {
            loads += 1
            if loads == 1 { throw Failure.unavailable }
            return original
        }
        XCTAssertThrowsError(try tokenizer.encode("<unused1>"))
        XCTAssertEqual(try tokenizer.encode("abc"), original.encode(text: "abc"))
        XCTAssertEqual(try tokenizer.encode("<unused1>"), original.encode(text: "<unused1>"))
        XCTAssertEqual(loads, 2)
    }
}
