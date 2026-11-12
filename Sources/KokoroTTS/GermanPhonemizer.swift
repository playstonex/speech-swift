import Foundation

/// German text normalization and grapheme-to-phoneme conversion for the
/// Kokoro German voices (the kikiri-tts fine-tunes).
///
/// The voices were trained on espeak-ng 1.52 German phonemes as emitted by
/// misaki's EspeakG2P: diphthongs folded to `I`/`W`, affricates to `ʦ`/`ʧ`/`ʤ`,
/// the short ü written `y`, `ɜ` for a vocalic -er, stress marks placed before
/// the stressed vowel. This type reproduces that convention without espeak:
///
/// 1. `normalize(_:)` spells out numbers, ordinals, dates, times, currency,
///    units, percentages and common abbreviations.
/// 2. `phonemize(_:dictionary:)` resolves each word through the dictionary
///    (`dict_de.json`), then an inflected form of a dictionary word, then a
///    compound of dictionary words, then the rule-based `g2pWord(_:)`.
///
/// The structure is shared with speech-core's `kokoro_german.cpp`; keep the two
/// in lockstep when changing a rule.
final class GermanPhonemizer {

    typealias Dictionary = [String: String]

    init() {}

    // MARK: - Public API

    /// Spell out digits, dates, times, currency amounts, units and abbreviations.
    func normalize(_ text: String) -> String {
        if text.isEmpty { return text }
        var t = Array(text.unicodeScalars)
        for i in t.indices {
            switch t[i] {
            case "„", "“", "«", "»", "‹", "›": t[i] = "\""
            case "‘", "’": t[i] = "'"
            case "\t", "\r", "\u{A0}", "\u{2009}", "\u{202F}": t[i] = " "
            default: break
            }
        }
        t = Self.expandAbbreviations(t)
        var r: [Unicode.Scalar] = []
        var i = 0
        while i < t.count {
            if t[i] == "§" {
                if i + 1 < t.count && t[i + 1] == "§" {
                    r.append(contentsOf: Array("Paragrafen ".unicodeScalars))
                    i += 1
                } else {
                    r.append(contentsOf: Array("Paragraf ".unicodeScalars))
                }
            } else {
                r.append(t[i])
            }
            i += 1
        }
        t = Self.normalizeNumbers(r)
        r = []
        for c in t {
            if c == "%" {
                if let last = r.last, last != " " { r.append(" ") }
                r.append(contentsOf: Array("Prozent".unicodeScalars))
            } else {
                r.append(c)
            }
        }
        var out: [Unicode.Scalar] = []
        for c in r {
            if c == " ", let last = out.last, last == " " { continue }
            out.append(c)
        }
        while let last = out.last, last == " " { out.removeLast() }
        while let first = out.first, first == " " { out.removeFirst() }
        return Self.string(out)
    }

    /// Rule-based grapheme-to-phoneme conversion of one word (no dictionary).
    func g2pWord(_ word: String) -> String {
        Self.g2pLower(Self.foldAccents(Self.lower(Array(word.unicodeScalars))), depth: 0)
    }

    /// Phonemes for one word: dictionary, inflected stem, compound, then rules.
    /// A capitalised homograph such as "Weg" selects the noun reading.
    func phonemizeWord(_ word: String, dictionary: Dictionary) -> String {
        let w = Array(word.unicodeScalars)
        if w.isEmpty { return "" }
        let lw = Self.lower(w)
        if Self.isUpper(w[0]) && Self.string(lw) == "weg" { return "vˈeːk" }
        if let ps = dictionary[Self.string(lw)] { return ps }
        let folded = Self.foldAccents(lw)
        if folded != lw, let ps = dictionary[Self.string(folded)] { return ps }
        if let ps = Self.inflected(folded, dictionary) { return ps }
        if let ps = Self.compound(folded, dictionary, depth: 0) { return ps }
        return Self.g2pLower(folded, depth: 0)
    }

    /// Phonemize text (normalized first). Words are separated by single
    /// spaces; sentence punctuation passes through.
    func phonemize(_ text: String, dictionary: Dictionary) -> String {
        let t = Array(normalize(text).unicodeScalars)
        var out = ""
        var pendingSpace = false
        let n = t.count
        var i = 0
        while i < n {
            let c = t[i]
            if c == " " || c == "\n" || c == "\t" || c == "\r" {
                pendingSpace = true
                i += 1
                continue
            }
            if Self.isLetter(c) || c == "'" {
                var j = i
                while j < n && (Self.isLetter(t[j]) || t[j] == "'" || (t[j] == "-" && j + 1 < n && Self.isLetter(t[j + 1]))) {
                    j += 1
                }
                if pendingSpace && !out.isEmpty { out += " " }
                pendingSpace = false
                // Hyphenated words: each part on its own, later parts demoted.
                var first = true
                var part: [Unicode.Scalar] = []
                func flush() {
                    let clean = part.filter { $0 != "'" }
                    if !clean.isEmpty {
                        let pp = phonemizeWord(Self.string(clean), dictionary: dictionary)
                        out += first ? pp : Self.demotePrimary(pp)
                        first = false
                    }
                    part = []
                }
                for ch in t[i..<j] {
                    if ch == "-" { flush() } else { part.append(ch) }
                }
                flush()
                i = j
                continue
            }
            switch c {
            case ",", ".", "!", "?", ";", ":", "-", "\"", "(", ")":
                out.unicodeScalars.append(c)
                pendingSpace = false
            default:
                break
            }
            i += 1
        }
        return out
    }

    // MARK: - Scalars and letters

    private static func string(_ s: [Unicode.Scalar]) -> String {
        var v = String.UnicodeScalarView()
        v.append(contentsOf: s)
        return String(v)
    }

    private static func string(_ s: ArraySlice<Unicode.Scalar>) -> String { string(Array(s)) }

    private static func isVowel(_ c: Unicode.Scalar) -> Bool {
        switch c {
        case "a", "e", "i", "o", "u", "y", "ä", "ö", "ü": return true
        default: return false
        }
    }

    private static func isDigit(_ c: Unicode.Scalar) -> Bool { c >= "0" && c <= "9" }

    private static func isLetter(_ c: Unicode.Scalar) -> Bool {
        if (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") { return true }
        if c == "ß" || c == "ẞ" { return true }
        if c.value >= 0xC0 && c.value <= 0x24F { return c.value != 0xD7 && c.value != 0xF7 }
        return false
    }

    private static func isAlnum(_ c: Unicode.Scalar) -> Bool { isLetter(c) || isDigit(c) }

    private static func isUpper(_ c: Unicode.Scalar) -> Bool {
        (c >= "A" && c <= "Z") || (c.value >= 0xC0 && c.value <= 0xDE && c.value != 0xD7)
    }

    private static func toLower(_ c: Unicode.Scalar) -> Unicode.Scalar {
        if c >= "A" && c <= "Z" { return Unicode.Scalar(c.value + 32)! }
        if c.value >= 0xC0 && c.value <= 0xDE && c.value != 0xD7 { return Unicode.Scalar(c.value + 32)! }
        if c == "ẞ" { return "ß" }
        if c == "İ" { return "i" }
        return c
    }

    private static func lower(_ s: [Unicode.Scalar]) -> [Unicode.Scalar] { s.map(toLower) }

    /// Fold loan-word accents onto the plain German letters the rules know.
    private static func foldAccents(_ s: [Unicode.Scalar]) -> [Unicode.Scalar] {
        s.map { c -> Unicode.Scalar in
            switch c {
            case "é", "è", "ê", "ë": return "e"
            case "á", "à", "â", "ã": return "a"
            case "ó", "ò", "ô", "õ": return "o"
            case "ú", "ù", "û": return "u"
            case "í", "ì", "î", "ï": return "i"
            case "ñ": return "n"
            case "ç": return "s"
            case "ø": return "ö"
            case "å": return "o"
            case "æ": return "ä"
            case "œ": return "ö"
            default: return c
            }
        }
    }

    private static func countNuclei(_ s: [Unicode.Scalar]) -> Int {
        var n = 0
        var i = 0
        while i < s.count {
            if isVowel(s[i]) {
                n += 1
                while i < s.count && isVowel(s[i]) { i += 1 }
            } else {
                i += 1
            }
        }
        return n
    }

    private static func startsWith(_ s: [Unicode.Scalar], _ pos: Int, _ prefix: String) -> Bool {
        let p = Array(prefix.unicodeScalars)
        if pos + p.count > s.count { return false }
        for k in 0..<p.count where s[pos + k] != p[k] { return false }
        return true
    }

    private static func endsWith(_ s: [Unicode.Scalar], _ suffix: String) -> Bool {
        let p = Array(suffix.unicodeScalars)
        if s.count < p.count { return false }
        let off = s.count - p.count
        for k in 0..<p.count where s[off + k] != p[k] { return false }
        return true
    }

    private static func scalars(_ s: String) -> [Unicode.Scalar] { Array(s.unicodeScalars) }

    private static func firstScalar(_ s: String) -> Unicode.Scalar? { s.unicodeScalars.first }

    // Stress marks
    private static let primary = "ˈ"
    private static let secondary = "ˌ"

    private static func stripStress(_ ps: String) -> String {
        ps.replacingOccurrences(of: primary, with: "").replacingOccurrences(of: secondary, with: "")
    }

    private static func demotePrimary(_ ps: String) -> String {
        ps.replacingOccurrences(of: primary, with: secondary)
    }

    private static func stripMarksPrefix(_ ps: String) -> String {
        var s = ps
        while s.hasPrefix(primary) || s.hasPrefix(secondary) { s.removeFirst() }
        return s
    }

    // MARK: - Tables

    private enum PrefixKind { case unstressed, stressed, secondary }

    private struct Prefix {
        let graph: String
        let phon: String
        let kind: PrefixKind
    }

    // Longer prefixes first within a family (zusammen before zu).
    private static let prefixes: [Prefix] = [
        Prefix(graph: "zusammen", phon: "ʦuːzˈamən", kind: .stressed), Prefix(graph: "zurück", phon: "ʦuːrˈʏk", kind: .stressed),
        Prefix(graph: "wieder", phon: "viːdɜ", kind: .secondary), Prefix(graph: "unter", phon: "ˌʊntɜ", kind: .secondary),
        Prefix(graph: "über", phon: "ˌyːbɜ", kind: .secondary),
        Prefix(graph: "durch", phon: "dˈʊɐç", kind: .stressed), Prefix(graph: "miss", phon: "mˈɪs", kind: .stressed),
        Prefix(graph: "fort", phon: "fˈɔɾt", kind: .stressed), Prefix(graph: "nach", phon: "nˈɑːx", kind: .stressed),
        Prefix(graph: "auf", phon: "ˈWf", kind: .stressed), Prefix(graph: "aus", phon: "ˈWs", kind: .stressed),
        Prefix(graph: "bei", phon: "bˈI", kind: .stressed), Prefix(graph: "ein", phon: "ˈIn", kind: .stressed),
        Prefix(graph: "mit", phon: "mˈɪt", kind: .stressed), Prefix(graph: "vor", phon: "fˈoːɾ", kind: .stressed),
        Prefix(graph: "weg", phon: "vˈɛk", kind: .stressed), Prefix(graph: "her", phon: "hˈɛɾ", kind: .stressed),
        Prefix(graph: "hin", phon: "hˈɪn", kind: .stressed), Prefix(graph: "los", phon: "lˈoːs", kind: .stressed),
        Prefix(graph: "ab", phon: "ˈap", kind: .stressed), Prefix(graph: "an", phon: "ˈan", kind: .stressed),
        Prefix(graph: "um", phon: "ˈʊm", kind: .stressed), Prefix(graph: "un", phon: "ˈʊn", kind: .stressed),
        Prefix(graph: "ur", phon: "ˈuːɾ", kind: .stressed), Prefix(graph: "zu", phon: "ʦˈuː", kind: .stressed),
        Prefix(graph: "ver", phon: "fɛɾ", kind: .unstressed), Prefix(graph: "zer", phon: "ʦɛɾ", kind: .unstressed),
        Prefix(graph: "ent", phon: "ɛnt", kind: .unstressed), Prefix(graph: "emp", phon: "ɛmp", kind: .unstressed),
        Prefix(graph: "be", phon: "bə", kind: .unstressed), Prefix(graph: "ge", phon: "ɡə", kind: .unstressed),
        Prefix(graph: "er", phon: "ɛɾ", kind: .unstressed),
    ]

    // Stressed loan suffixes; the phonemes carry the primary stress.
    private static let stressedSuffixes: [(String, String)] = [
        ("ierungen", "ˈiːrʊŋən"), ("ierung", "ˈiːrʊŋ"), ("ierten", "ˈiːɾtən"), ("ierte", "ˈiːɾtə"),
        ("ieren", "ˈiːrən"), ("ierst", "ˈiːɾst"), ("iert", "ˈiːɾt"), ("iere", "ˈiːrə"),
        ("tionen", "ʦjˈoːnən"), ("tion", "ʦjˈoːn"), ("ssionen", "sjˈoːnən"), ("ssion", "sjˈoːn"),
        ("sionen", "zjˈoːnən"), ("sion", "zjˈoːn"),
        ("itäten", "iːtˈɛːtən"), ("ität", "iːtˈɛːt"), ("täten", "tˈɛːtən"), ("tät", "tˈɛːt"),
        ("ismus", "ˈɪsmʊs"), ("istin", "ˈɪstɪn"), ("isten", "ˈɪstən"), ("ist", "ˈɪst"),
        ("ieurs", "jˈøːɾs"), ("ieur", "jˈøːɾ"), ("euren", "ˈøːrən"), ("eure", "ˈøːrə"), ("eur", "ˈøːɾ"),
        ("iven", "ˈiːvən"), ("ive", "ˈiːvə"), ("iv", "ˈiːf"),
        ("agen", "ˈɑːʒən"), ("age", "ˈɑːʒə"),
        ("ellen", "ˈɛlən"), ("elle", "ˈɛlə"), ("ell", "ˈɛl"),
        ("enzen", "ˈɛnʦən"), ("enz", "ˈɛnʦ"), ("anzen", "ˈanʦən"), ("anz", "ˈanʦ"),
        ("enten", "ˈɛntən"), ("ent", "ˈɛnt"), ("anten", "ˈantən"), ("ant", "ˈant"),
        ("menten", "mˈɛntən"), ("ment", "mˈɛnt"),
        ("uren", "ˈuːrən"), ("ur", "ˈuːɾ"),
        ("inen", "ˈiːnən"), ("ine", "ˈiːnə"), ("ösen", "ˈøːzən"), ("öse", "ˈøːzə"), ("ös", "ˈøːs"),
        ("eien", "ˈIən"), ("ei", "ˈI"), ("ien", "ˈiːən"), ("ie", "ˈiː"),
        ("iken", "ˈiːkən"), ("ik", "ˈiːk"), ("ier", "ˈiːɾ"),
        ("alen", "ˈɑːlən"), ("ale", "ˈɑːlə"), ("al", "ˈɑːl"), ("ären", "ˈɛːrən"), ("är", "ˈɛːɾ"),
        ("ade", "ˈɑːdə"), ("ose", "ˈoːzə"), ("ette", "ˈɛtə"), ("ett", "ˈɛt"), ("een", "ˈeːən"), ("ee", "ˈeː"),
    ]
    private static let minStemBeforeSuffix = 3

    // Unstressed native suffixes with fixed pronunciations.
    private static let nativeSuffixes: [(String, String, Bool)] = [
        // -ig-/-lich- always leave an unstressed syllable before -keit, so the
        // secondary stress is part of the suffix phonemes.
        ("igkeiten", "ɪçkˌItən", false), ("igkeit", "ɪçkˌIt", false),
        ("lichkeiten", "lɪçkˌItən", false), ("lichkeit", "lɪçkˌIt", false),
        ("ungen", "ʊŋən", true), ("ung", "ʊŋ", true),
        ("heiten", "hItən", true), ("heit", "hIt", true),
        ("keiten", "kItən", true), ("keit", "kIt", true),
        ("schaften", "ʃaftən", true), ("schaft", "ʃaft", true),
        ("lichen", "lɪçən", false), ("liche", "lɪçə", false), ("lich", "lɪç", false),
        ("ischen", "ɪʃən", false), ("ische", "ɪʃə", false), ("isch", "ɪʃ", false),
        ("igen", "ɪɡən", false), ("ige", "ɪɡə", false), ("iger", "ɪɡɜ", false), ("ig", "ɪç", false),
        ("chen", "çən", false), ("lein", "lIn", false),
        ("baren", "bɑːrən", false), ("bare", "bɑːrə", false), ("bar", "bɑːɾ", false),
        ("samen", "zɑːmən", false), ("same", "zɑːmə", false), ("sam", "zɑːm", false),
        ("losen", "loːzən", false), ("lose", "loːzə", false), ("los", "loːs", false),
        ("nisse", "nɪsə", false), ("nis", "nɪs", false), ("tums", "tuːms", false), ("tum", "tuːm", false),
        ("lingen", "lɪŋən", false), ("lings", "lɪŋs", false), ("ling", "lɪŋ", false),
        ("innen", "ɪnən", true), ("in", "ɪn", true),
    ]

    private static let digraphs: [String: String] = [
        "ei": "I", "ai": "I", "ey": "I", "ay": "I", "au": "W",
        "eu": "ɔø", "äu": "ɔø", "ie": "iː", "aa": "ɑː", "ee": "eː", "oo": "oː",
    ]

    private static func longVowel(_ c: Unicode.Scalar) -> String {
        switch c {
        case "a": return "ɑː"; case "e": return "eː"; case "i": return "iː"
        case "o": return "oː"; case "u": return "uː"; case "ä": return "ɛː"
        case "ö": return "øː"; case "ü": return "yː"; case "y": return "yː"
        default: return ""
        }
    }

    private static func shortVowel(_ c: Unicode.Scalar) -> String {
        switch c {
        case "a": return "a"; case "e": return "ɛ"; case "i": return "ɪ"
        case "o": return "ɔ"; case "u": return "ʊ"; case "ä": return "ɛ"
        case "ö": return "œ"; case "ü": return "y"; case "y": return "y"
        default: return ""
        }
    }

    private static let vowelPhonemes: Set<Unicode.Scalar> = Set("aeiouyɑɛɪɔʊøœəɜIWA".unicodeScalars)
    private static let backPhonemes: Set<Unicode.Scalar> = Set("aɑɔʊuoW".unicodeScalars)
    private static let frontPhonemes: Set<Unicode.Scalar> = Set("iɪeɛ".unicodeScalars)

    private static func phonemeIsVowel(_ c: Unicode.Scalar?) -> Bool { c.map { vowelPhonemes.contains($0) } ?? false }

    // MARK: - Prefix analysis

    private static func validStem(_ stem: [Unicode.Scalar]) -> Bool {
        if stem.count < 3 || countNuclei(stem) == 0 { return false }
        return isVowel(stem[0]) || stem.count >= 4 || countNuclei(stem) >= 2
    }

    private static let consonantOnlyPrefixes: Set<String> = ["un", "ur", "ab", "ein", "an"]

    private struct PrefixMatch {
        let phon: String
        let kind: PrefixKind
        let stem: [Unicode.Scalar]
    }

    private static func splitPrefix(_ word: [Unicode.Scalar]) -> PrefixMatch? {
        for p in prefixes {
            guard startsWith(word, 0, p.graph) else { continue }
            let stem = Array(word[p.graph.unicodeScalars.count...])
            guard validStem(stem) else { continue }
            if consonantOnlyPrefixes.contains(p.graph) && isVowel(stem[0]) {
                let allowed = (p.graph == "an" && stem[0] == "e") || (p.graph == "un" && stem[0] != "i" && stem[0] != "u")
                if !allowed { continue }
            }
            if (p.graph == "her" || p.graph == "hin") && isVowel(stem[0]) {
                // herunter, hinein: the prefix is unstressed, the stem carries primary
                return PrefixMatch(phon: stripStress(p.phon), kind: .unstressed, stem: stem)
            }
            return PrefixMatch(phon: p.phon, kind: p.kind, stem: stem)
        }
        return nil
    }

    // MARK: - Syllable nuclei

    private struct Nucleus {
        var grapheme: [Unicode.Scalar]
        var start: Int
        var end: Int   // exclusive; includes a lengthening h
        var longH: Bool
        var phoneme: String = ""
    }

    private static func findNuclei(_ w: [Unicode.Scalar]) -> [Nucleus] {
        var nuclei: [Nucleus] = []
        var i = 0
        let n = w.count
        while i < n {
            let c = w[i]
            if c == "q" && i + 1 < n && w[i + 1] == "u" { i += 2; continue }
            var vowel = isVowel(c)
            if c == "y" {
                let initialGlide = i == 0 && i + 1 < n && isVowel(w[i + 1])
                let medialGlide = i > 0 && isVowel(w[i - 1]) && i + 1 < n && isVowel(w[i + 1])
                if initialGlide || medialGlide { vowel = false }
            }
            if !vowel { i += 1; continue }
            var g: [Unicode.Scalar] = [c]
            if i + 1 < n, digraphs[string([w[i], w[i + 1]])] != nil { g = [w[i], w[i + 1]] }
            var j = i + g.count
            var longH = false
            if j < n && w[j] == "h" { longH = true; j += 1 }
            nuclei.append(Nucleus(grapheme: g, start: i, end: j, longH: longH))
            i = j
        }
        return nuclei
    }

    private static func consonantsAfter(_ w: [Unicode.Scalar], _ nuclei: [Nucleus], _ idx: Int) -> [Unicode.Scalar] {
        let end = idx + 1 < nuclei.count ? nuclei[idx + 1].start : w.count
        return Array(w[nuclei[idx].end..<end])
    }

    private static func clusterWeight(_ cl: [Unicode.Scalar]) -> Int {
        if cl.isEmpty { return 0 }
        let s = string(cl)
        if s == "th" || s == "ph" { return 1 }
        if s == "ß" { return 0 }
        return cl.count
    }

    private static func vowelPhoneme(_ w: [Unicode.Scalar], _ nuclei: [Nucleus], _ idx: Int,
                                     _ stressedIdx: Int, _ hasSuffix: Bool) -> String {
        let nu = nuclei[idx]
        let g = nu.grapheme
        let cl = consonantsAfter(w, nuclei, idx)
        let cls = string(cl)
        let lastInCore = idx + 1 == nuclei.count
        let wordFinalSyllable = lastInCore && !hasSuffix
        let base = g[0]
        if let d = digraphs[string(g)] { return d }
        if nu.longH { return longVowel(base) }
        if let first = cl.first, first == "ß" { return longVowel(base) }
        let postStress = idx > stressedIdx
        let preStress = idx < stressedIdx
        let weight = clusterWeight(cl)
        if base == "e" && postStress {
            // e + r before a consonant or at the word end -> vocalic r (Kinder,
            // Kindern, dauerte); e + r + vowel keeps a syllabic r (andere)
            if let first = cl.first, first == "r", cl.count >= 2 || wordFinalSyllable { return "ɜ" }
            if lastInCore && hasSuffix { return "ə" }
            if weight >= 2 || wordFinalSyllable { return "ə" }
            // open post-stress syllable: schwa before a schwa syllable ("redete",
            // "gegebenen"), otherwise tense e ("Kamera", "Lehrerinnen")
            if idx + 1 < nuclei.count && string(nuclei[idx + 1].grapheme) == "e" && cl.count == 1 && "tdnlrsmgbk".unicodeScalars.contains(cl[0]) {
                return "ə"
            }
            return longVowel("e")
        }
        if base == "i" && postStress && weight >= 1 { return "ɪ" }
        if base == "u" && postStress && wordFinalSyllable && ["m", "s", "ms", "ng", "ngs", "nft"].contains(cls) {
            return "ʊ"
        }
        // vowel + r + s(t): long (Kurs, erst, Vers)
        if cl.count >= 2 && cl[0] == "r" && cl[1] == "s" { return longVowel(base) }
        // u + r + consonant -> ʊɐ (Burg, kurz, durch); the r is absorbed
        if base == "u" && cl.count >= 2 && cl[0] == "r" { return "ʊɐ" }
        // u before ch: tense (Buch, Besuch, suchen) but not before chs/cht
        if base == "u" && cls.hasPrefix("ch") && !cls.hasPrefix("chs") && !cls.hasPrefix("cht") { return "uː" }
        if weight == 0 {
            if hasSuffix && lastInCore {
                // "eskala|tion", "temporal|en": an open syllable before the suffix
                return base == "a" ? "a" : longVowel(base)
            }
            if base == "e" { return idx > 0 ? "ə" : longVowel(base) }
            return longVowel(base)
        }
        if weight == 1 {
            if preStress && base == "a" { return "a" }
            return longVowel(base)
        }
        return shortVowel(base)
    }

    // MARK: - Consonants

    private static func collapseDoubles(_ cl: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var out: [Unicode.Scalar] = []
        for ch in cl {
            if let last = out.last, last == ch, "bdfgklmnprtz".unicodeScalars.contains(ch) { continue }
            out.append(ch)
        }
        return out
    }

    /// Map one consonant cluster. `prevVowel`/`nextVowel` are the phonemes of
    /// the surrounding nuclei ("" when none).
    private static func mapConsonants(_ cluster: [Unicode.Scalar], prevVowel: String, nextVowel: String,
                                      atWordStart: Bool, atStemStart: Bool) -> String {
        // Keep "ss" distinct from a single s before collapsing doubles.
        var cl: [Unicode.Scalar] = []
        var k = 0
        while k < cluster.count {
            if cluster[k] == "s" && k + 1 < cluster.count && cluster[k + 1] == "s" {
                cl.append("S")
                k += 2
            } else {
                cl.append(cluster[k])
                k += 1
            }
        }
        cl = collapseDoubles(cl)
        let prevCp = firstScalar(prevVowel)
        let nextCp = firstScalar(nextVowel)
        let prevBack = prevCp.map { backPhonemes.contains($0) } ?? false
        let prevBackNotR = prevBack && prevVowel != "ʊɐ"
        let nextFront = nextCp.map { frontPhonemes.contains($0) } ?? false

        var out = ""
        var i = 0
        let n = cl.count
        while i < n {
            let c = cl[i]
            let nxt: Unicode.Scalar? = i + 1 < n ? cl[i + 1] : nil
            let last = i + 1 >= n
            if startsWith(cl, i, "tsch") { out += (atWordStart && i == 0) ? "ʧ" : "tʃ"; i += 4; continue }
            if startsWith(cl, i, "dsch") { out += "ʤ"; i += 4; continue }
            if startsWith(cl, i, "sch") { out += "ʃ"; i += 3; continue }
            if startsWith(cl, i, "sh") { out += "ʃ"; i += 2; continue }
            if startsWith(cl, i, "chs") { out += (i == 0 && prevBack) ? "xs" : "ks"; i += 3; continue }
            if startsWith(cl, i, "ch") {
                if i == 0 && atWordStart {
                    // Chemie, China -> ç before a front vowel; Chor, Christ -> k
                    let endsCluster = i + 2 >= n
                    out += (endsCluster && nextFront) ? "ç" : "k"
                } else if i == 0 && prevBackNotR {
                    out += "x"
                } else {
                    out += "ç"
                }
                i += 2
                continue
            }
            if startsWith(cl, i, "ck") { out += "k"; i += 2; continue }
            if startsWith(cl, i, "dt") { out += "t"; i += 2; continue }
            if startsWith(cl, i, "th") { out += "t"; i += 2; continue }
            if startsWith(cl, i, "ph") { out += "f"; i += 2; continue }
            if startsWith(cl, i, "pf") { out += "pf"; i += 2; continue }
            if startsWith(cl, i, "qu") { out += "kv"; i += 2; continue }
            if startsWith(cl, i, "tz") { out += "ʦ"; i += 2; continue }
            if startsWith(cl, i, "ng") { out += "ŋ"; i += 2; continue }
            if startsWith(cl, i, "nk") { out += "ŋk"; i += 2; continue }
            if c == "S" { out += "s"; i += 1; continue }
            if startsWith(cl, i, "sp") && i == 0 && atStemStart { out += "ʃp"; i += 2; continue }
            if startsWith(cl, i, "st") && i == 0 && atStemStart { out += "ʃt"; i += 2; continue }

            let onsetBeforeVowel = last && !nextVowel.isEmpty
            let voicedOk = (!last && nxt.map { "lrnmvw".unicodeScalars.contains($0) } == true) || onsetBeforeVowel
            let before = out.unicodeScalars.last
            switch c {
            case "b": out += voicedOk ? "b" : "p"
            case "d": out += voicedOk ? "d" : "t"
            case "g": out += voicedOk ? "ɡ" : "k"
            case "s":
                let voiceless = "kptfçʃx".unicodeScalars
                let afterVoiceless = before.map { voiceless.contains($0) } ?? false
                out += (onsetBeforeVowel && !afterVoiceless) ? "z" : "s"
            case "ß": out += "s"
            case "v": out += ((i == 0 && (atWordStart || atStemStart)) || !onsetBeforeVowel) ? "f" : "v"
            case "w": out += "v"
            case "z": out += "ʦ"
            case "c":
                let frontLetters = "eiyäöü".unicodeScalars
                let frontPh = "eɛiɪyø".unicodeScalars
                let front = (nxt.map { frontLetters.contains($0) } == true) || (last && nextCp.map { frontPh.contains($0) } == true)
                out += front ? "ʦ" : "k"
            case "x": out += "ks"
            case "y": out += "j"
            case "r":
                if onsetBeforeVowel {
                    out += (i == 0 || cl[i - 1] == "f" || out.hasSuffix("ʃ")) ? "r" : "ɾ"
                } else {
                    out += "ɾ"
                }
            default:
                out.unicodeScalars.append(c)
            }
            i += 1
        }
        return out
    }

    /// Insert a stress mark before the first vowel phoneme ("ʃˌaft").
    private static func markBeforeFirstVowel(_ phon: String, _ mark: String) -> String {
        var out = ""
        var placed = false
        for cp in phon.unicodeScalars {
            if !placed && vowelPhonemes.contains(cp) {
                out += mark
                placed = true
            }
            out.unicodeScalars.append(cp)
        }
        return placed ? out : mark + phon
    }

    private static func suffixFirstVowel(_ phon: String) -> String {
        guard let cp = firstScalar(stripMarksPrefix(phon)), vowelPhonemes.contains(cp) else { return "" }
        return String(Character(cp))
    }

    /// Phonemize a core (prefix and suffix stripped).
    private static func stemG2P(_ core: [Unicode.Scalar], stressedIdx: Int, suffixPhon: String,
                                atWordStart: Bool, atStemStart: Bool, secondaryOnFirst: Bool) -> String {
        var nuclei = findNuclei(core)
        let hasSuffix = !suffixPhon.isEmpty
        let suffixFirst = hasSuffix ? suffixFirstVowel(suffixPhon) : ""
        if nuclei.isEmpty {
            return mapConsonants(core, prevVowel: "", nextVowel: suffixFirst, atWordStart: atWordStart, atStemStart: atStemStart)
        }
        for idx in nuclei.indices {
            nuclei[idx].phoneme = vowelPhoneme(core, nuclei, idx, stressedIdx, hasSuffix)
        }
        var out = mapConsonants(Array(core[0..<nuclei[0].start]), prevVowel: "", nextVowel: nuclei[0].phoneme,
                                atWordStart: atWordStart, atStemStart: atStemStart)
        for idx in nuclei.indices {
            let nu = nuclei[idx]
            if idx == stressedIdx { out += primary }
            else if idx == 0 && secondaryOnFirst { out += secondary }
            out += nu.phoneme
            var cl = consonantsAfter(core, nuclei, idx)
            if (nu.phoneme == "ʊɐ" || nu.phoneme == "ɜ"), let first = cl.first, first == "r" { cl.removeFirst() }
            let nxt = idx + 1 < nuclei.count ? nuclei[idx + 1].phoneme : suffixFirst
            out += mapConsonants(cl, prevVowel: nu.phoneme, nextVowel: nxt, atWordStart: false, atStemStart: false)
        }
        return out
    }

    private static func g2pLower(_ w: [Unicode.Scalar], depth: Int) -> String {
        if w.isEmpty { return "" }
        if depth < 3, let pm = splitPrefix(w) {
            // The stem is phonemized as a word of its own (so sp/st, v, initial r
            // and ch follow the word-initial rules) and may carry further
            // prefixes: aus-ge-wählt, be-vor-stehen, un-er-schütterlich.
            let body = g2pLower(pm.stem, depth: depth + 1)
            if pm.kind == .stressed { return pm.phon + demotePrimary(body) }
            return pm.phon + body
        }
        let stem = w
        var suffixPhon = ""
        var suffixLen = 0
        for (graph, phon) in stressedSuffixes {
            let gl = graph.unicodeScalars.count
            if endsWith(stem, graph) && stem.count - gl >= minStemBeforeSuffix && countNuclei(Array(stem[0..<(stem.count - gl)])) >= 1 {
                suffixPhon = phon
                suffixLen = gl
                break
            }
        }
        var nativePhon = ""
        var nativeLen = 0
        var nativeSecondary = false
        if suffixLen == 0 {
            for (graph, phon, sec) in nativeSuffixes {
                guard endsWith(stem, graph) else { continue }
                let gl = graph.unicodeScalars.count
                let coreCandidate = Array(stem[0..<(stem.count - gl)])
                if coreCandidate.count < 2 || countNuclei(coreCandidate) < 1 || isVowel(coreCandidate[coreCandidate.count - 1]) { continue }
                if graph == "chen", let lastLetter = coreCandidate.last, lastLetter == "s" {
                    // waschen, löschen: that is "sch" + "en"; Häuschen keeps the
                    // diminutive thanks to its umlaut
                    let umlaut = coreCandidate.contains("ä") || coreCandidate.contains("ö") || coreCandidate.contains("ü")
                    if !umlaut { continue }
                }
                nativePhon = phon
                nativeLen = gl
                nativeSecondary = sec
                break
            }
        }
        var core = stem
        if suffixLen > 0 { core = Array(stem[0..<(stem.count - suffixLen)]) }
        else if nativeLen > 0 { core = Array(stem[0..<(stem.count - nativeLen)]) }

        let nucleiCount = findNuclei(core).count
        let stressedIdx = suffixLen == 0 ? 0 : nucleiCount
        let secondaryOnFirst = suffixLen > 0 && nucleiCount >= 2
        var body = stemG2P(core, stressedIdx: stressedIdx, suffixPhon: !suffixPhon.isEmpty ? suffixPhon : nativePhon,
                           atWordStart: true, atStemStart: true, secondaryOnFirst: secondaryOnFirst)
        if !suffixPhon.isEmpty {
            body += suffixPhon
        } else if !nativePhon.isEmpty {
            // -ung/-heit/-keit/-schaft/-in carry secondary stress when at least
            // one unstressed syllable separates them from the primary
            // (Wissenschaft, Lehrerin, Beobachtung) but not when the primary sits
            // right before them (Beratung, Landschaft, Gräfin).
            if nativeSecondary && nucleiCount >= 1 && nucleiCount - 1 - stressedIdx >= 1 {
                body += markBeforeFirstVowel(nativePhon, secondary)
            } else {
                body += nativePhon
            }
        }
        return body
    }

    // MARK: - Number normalization

    private static let ones = ["", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
                               "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn",
                               "siebzehn", "achtzehn", "neunzehn"]
    private static let tens = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
    private static let months = ["", "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
                                 "September", "Oktober", "November", "Dezember"]

    static func intToDe(_ n: Int, standalone: Bool = true) -> String {
        if n < 0 { return "minus " + intToDe(-n, standalone: standalone) }
        if n == 0 { return "null" }
        if n == 1 { return standalone ? "eins" : "ein" }
        if n < 20 { return ones[n] }
        if n < 100 {
            let o = n % 10, t = n / 10
            return o != 0 ? ones[o] + "und" + tens[t] : tens[t]
        }
        if n < 1000 {
            let h = n / 100, r = n % 100
            return ones[h] + "hundert" + (r != 0 ? intToDe(r, standalone: false) : "")
        }
        if n < 1_000_000 {
            let t = n / 1000, r = n % 1000
            let prefix = t != 1 ? intToDe(t, standalone: false) : "ein"
            return prefix + "tausend" + (r != 0 ? intToDe(r, standalone: false) : "")
        }
        let scales: [(Int, String, String)] = [
            (1_000_000_000_000, "eine Billion", "Billionen"),
            (1_000_000_000, "eine Milliarde", "Milliarden"),
            (1_000_000, "eine Million", "Millionen"),
        ]
        for (divisor, singular, plural) in scales where n >= divisor {
            let count = n / divisor, r = n % divisor
            let word = count == 1 ? singular : intToDe(count, standalone: false) + " " + plural
            return word + (r != 0 ? " " + intToDe(r, standalone: false) : "")
        }
        return String(n)
    }

    private static func ordinalStem(_ n: Int) -> String {
        switch n {
        case 1: return "erst"; case 2: return "zweit"; case 3: return "dritt"
        case 7: return "siebt"; case 8: return "acht"
        default: break
        }
        var stem = intToDe(n, standalone: false) + (n < 20 ? "t" : "st")
        if n == 100 || n == 1000 { stem = String(stem.dropFirst(3)) }
        return stem
    }

    private static func yearToDe(_ n: Int) -> String {
        if n >= 1100 && n <= 1999 {
            let c = n / 100, r = n % 100
            return intToDe(c, standalone: false) + "hundert" + (r != 0 ? intToDe(r, standalone: false) : "")
        }
        return intToDe(n)
    }

    private static func digitsToDe(_ digits: [Unicode.Scalar]) -> String {
        digits.map { intToDe(Int($0.value - 48)) }.joined(separator: " ")
    }

    private static func toNumber(_ digits: [Unicode.Scalar]) -> Int {
        var v = 0
        for d in digits {
            if v > 100_000_000_000_000 { break }
            v = v * 10 + Int(d.value - 48)
        }
        return v
    }

    private static func currencyName(_ sym: Unicode.Scalar) -> String? {
        switch sym {
        case "€": return "Euro"; case "$": return "Dollar"
        case "£": return "Pfund"; case "¥": return "Yen"
        default: return nil
        }
    }

    private static func currencyWords(_ sym: Unicode.Scalar, whole: [Unicode.Scalar], frac: [Unicode.Scalar]) -> String {
        let euros = whole.isEmpty ? 0 : toNumber(whole)
        var cents = 0
        if !frac.isEmpty {
            var two = Array(frac.prefix(2))
            while two.count < 2 { two.append("0") }
            cents = toNumber(two)
        }
        let word = currencyName(sym) ?? ""
        if cents == 0 { return intToDe(euros, standalone: false) + " " + word }
        return intToDe(euros, standalone: false) + " " + word + " und " + intToDe(cents, standalone: false) + " Cent"
    }

    // Longest first so "kg" wins over "g".
    private static let units: [(String, String)] = [
        ("kWh", "Kilowattstunden"), ("GHz", "Gigahertz"), ("MHz", "Megahertz"), ("kHz", "Kilohertz"),
        ("km/h", "Kilometer pro Stunde"), ("°C", "Grad Celsius"), ("mAh", "Milliamperestunden"),
        ("mg", "Milligramm"), ("kg", "Kilogramm"), ("km", "Kilometer"), ("cm", "Zentimeter"),
        ("mm", "Millimeter"), ("ml", "Milliliter"), ("Hz", "Hertz"), ("g", "Gramm"), ("l", "Liter"),
        ("m", "Meter"), ("°", "Grad"),
    ]

    private static func unitAt(_ text: [Unicode.Scalar], _ i: Int) -> (Int, String)? {
        for (sym, spoken) in units where startsWith(text, i, sym) {
            let end = i + sym.unicodeScalars.count
            if end >= text.count || !isAlnum(text[end]) { return (sym.unicodeScalars.count, spoken) }
        }
        return nil
    }

    private static func readDigits(_ text: [Unicode.Scalar], _ i: Int) -> ([Unicode.Scalar], Int) {
        var j = i
        while j < text.count && isDigit(text[j]) { j += 1 }
        return (Array(text[i..<j]), j)
    }

    private static func wordBefore(_ text: [Unicode.Scalar], _ i: Int) -> String {
        var j = i
        if j > 0 && text[j - 1] == " " { j -= 1 }
        var k = j
        while k > 0 && isLetter(text[k - 1]) { k -= 1 }
        return string(lower(Array(text[k..<j])))
    }

    private static let abbreviations: [(String, String, Bool)] = [
        ("Dr.", "Doktor", false), ("Prof.", "Professor", false), ("Hr.", "Herr", false),
        ("Fr.", "Frau", false), ("Str.", "Straße", false), ("Nr.", "Nummer", true),
        ("Tel.", "Telefon", false), ("Abt.", "Abteilung", false), ("Abs.", "Absatz", true),
        ("z.B.", "zum Beispiel", false), ("z. B.", "zum Beispiel", false),
        ("d.h.", "das heißt", false), ("d. h.", "das heißt", false),
        ("u.a.", "unter anderem", false), ("u. a.", "unter anderem", false),
        ("bzw.", "beziehungsweise", false), ("usw.", "und so weiter", false),
        ("etc.", "et cetera", false), ("ca.", "circa", false), ("vgl.", "vergleiche", false),
        ("inkl.", "inklusive", false), ("exkl.", "exklusive", false), ("ggf.", "gegebenenfalls", false),
        ("i.d.R.", "in der Regel", false), ("o.ä.", "oder ähnliches", false), ("u.U.", "unter Umständen", false),
        ("Jan.", "Januar", false), ("Feb.", "Februar", false), ("Mär.", "März", false), ("Apr.", "April", false),
        ("Jun.", "Juni", false), ("Jul.", "Juli", false), ("Aug.", "August", false), ("Sep.", "September", false),
        ("Okt.", "Oktober", false), ("Nov.", "November", false), ("Dez.", "Dezember", false),
        ("GmbH", "Gesellschaft mit beschränkter Haftung", false), ("AG", "Aktiengesellschaft", false),
    ]

    private static func atWordBoundary(_ text: [Unicode.Scalar], _ i: Int) -> Bool {
        i == 0 || !isAlnum(text[i - 1])
    }

    private static func expandAbbreviations(_ input: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var text = input
        for (abbrString, full, needsDigit) in abbreviations {
            let abbr = scalars(abbrString)
            let fullScalars = scalars(full)
            var start = 0
            while start + abbr.count <= text.count {
                guard let pos = (start..<(text.count - abbr.count + 1)).first(where: { p in
                    for k in 0..<abbr.count where text[p + k] != abbr[k] { return false }
                    return true
                }) else { break }
                let end = pos + abbr.count
                var ok = atWordBoundary(text, pos)
                if abbr.last == "." {
                    ok = ok && (end >= text.count || text[end] == " " || text[end] == "\n")
                } else {
                    ok = ok && (end >= text.count || !isAlnum(text[end]))
                }
                if needsDigit {
                    var k = end
                    while k < text.count && text[k] == " " { k += 1 }
                    ok = ok && k < text.count && isDigit(text[k])
                }
                if ok {
                    text.replaceSubrange(pos..<end, with: fullScalars)
                    start = pos + fullScalars.count
                } else {
                    start = end
                }
            }
        }
        return text
    }

    private static func normalizeNumbers(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var out: [Unicode.Scalar] = []
        let n = text.count
        var i = 0
        func emit(_ s: String) { out.append(contentsOf: Array(s.unicodeScalars)) }
        while i < n {
            let c = text[i]
            // currency symbol before the amount: € 12,50
            if currencyName(c) != nil && i + 1 < n {
                var j = i + 1
                while j < n && text[j] == " " { j += 1 }
                let (digits, k0) = readDigits(text, j)
                if !digits.isEmpty {
                    var k = k0
                    var whole = digits
                    var frac: [Unicode.Scalar] = []
                    while k < n && (text[k] == "." || text[k] == ",") && k + 1 < n && isDigit(text[k + 1]) {
                        let (more, k2) = readDigits(text, k + 1)
                        if text[k] == "," { frac = more } else { whole += more }
                        k = k2
                    }
                    emit(currencyWords(c, whole: whole, frac: frac))
                    i = k
                    continue
                }
            }
            if !isDigit(c) || (i > 0 && isLetter(text[i - 1])) {
                out.append(c)
                i += 1
                continue
            }
            let (digits, j) = readDigits(text, i)
            // time HH:MM
            if j < n && text[j] == ":" && digits.count <= 2 {
                let (mins, k) = readDigits(text, j + 1)
                if mins.count == 2 && (k >= n || !isDigit(text[k])) {
                    let h = toNumber(digits), m = toNumber(mins)
                    if h <= 23 && m <= 59 {
                        var k2 = k
                        if startsWith(text, k2, " Uhr") { k2 += 4 }
                        emit(intToDe(h) + " Uhr" + (m != 0 ? " " + intToDe(m) : ""))
                        i = k2
                        continue
                    }
                }
            }
            // date DD.MM.YYYY
            if j + 1 < n && text[j] == "." && digits.count <= 2 && isDigit(text[j + 1]) {
                let (month, k) = readDigits(text, j + 1)
                if month.count <= 2 && k + 1 < n && text[k] == "." && isDigit(text[k + 1]) {
                    let (year, k2) = readDigits(text, k + 1)
                    if year.count == 4 && (k2 >= n || !isDigit(text[k2])) {
                        let d = toNumber(digits), mo = toNumber(month), y = toNumber(year)
                        if d >= 1 && d <= 31 && mo >= 1 && mo <= 12 {
                            let prev = wordBefore(out, out.count)
                            var suffix = "er"
                            if ["am", "im", "vom", "zum", "den"].contains(prev) { suffix = "en" }
                            else if prev == "der" { suffix = "e" }
                            emit(ordinalStem(d) + suffix + " " + months[mo] + " " + yearToDe(y))
                            i = k2
                            continue
                        }
                    }
                }
            }
            // grouped thousands 1.234.567(,89)
            func groupAt(_ k: Int) -> Bool {
                k + 3 < n && text[k] == "." && isDigit(text[k + 1]) && isDigit(text[k + 2]) && isDigit(text[k + 3]) &&
                    (k + 4 >= n || !isDigit(text[k + 4]))
            }
            if digits.count <= 3 && groupAt(j) {
                var whole = digits
                var k = j
                while groupAt(k) {
                    whole += text[(k + 1)..<(k + 4)]
                    k += 4
                }
                var frac: [Unicode.Scalar] = []
                if k + 1 < n && text[k] == "," && isDigit(text[k + 1]) {
                    let (f, k2) = readDigits(text, k + 1)
                    frac = f
                    k = k2
                }
                emit(intToDe(toNumber(whole)) + (frac.isEmpty ? "" : " Komma " + digitsToDe(frac)))
                i = k
                continue
            }
            // decimal comma 3,14 (optionally followed by a unit or currency)
            if j + 1 < n && text[j] == "," && isDigit(text[j + 1]) {
                let (frac, k) = readDigits(text, j + 1)
                var k2 = k
                while k2 < n && text[k2] == " " { k2 += 1 }
                if k2 < n, currencyName(text[k2]) != nil {
                    emit(currencyWords(text[k2], whole: digits, frac: frac))
                    i = k2 + 1
                    continue
                }
                emit(intToDe(toNumber(digits)) + " Komma " + digitsToDe(frac))
                if let unit = unitAt(text, k2) {
                    emit(" " + unit.1)
                    i = k2 + unit.0
                } else {
                    i = k
                }
                continue
            }
            // currency or unit after the amount: 12 €, 500 g, 1,5 kg
            do {
                var k = j
                while k < n && text[k] == " " { k += 1 }
                if k < n, currencyName(text[k]) != nil {
                    emit(currencyWords(text[k], whole: digits, frac: []))
                    i = k + 1
                    continue
                }
                if let unit = unitAt(text, k) {
                    emit(intToDe(toNumber(digits)) + " " + unit.1)
                    i = k + unit.0
                    continue
                }
            }
            // ordinal "3." before a word
            if j + 2 < n && text[j] == "." && text[j + 1] == " " && isLetter(text[j + 2]) {
                let prev = wordBefore(out, out.count)
                emit(ordinalStem(toNumber(digits)) + (prev == "am" ? "en" : "e"))
                i = j + 1
                continue
            }
            // phone-like digit groups: 0170 123456
            if digits.count >= 2 && digits.count <= 4 && j + 1 < n && (text[j] == " " || text[j] == "-") && isDigit(text[j + 1]) {
                var groups = [digits]
                var k = j
                var ok = true
                while k + 1 < n && (text[k] == " " || text[k] == "-") && isDigit(text[k + 1]) {
                    let (more, k2) = readDigits(text, k + 1)
                    if more.count < 2 || more.count > 6 { ok = false; break }
                    groups.append(more)
                    k = k2
                }
                if ok && groups.count >= 2 {
                    emit(groups.map { digitsToDe($0) }.joined(separator: " "))
                    i = k
                    continue
                }
            }
            // year
            if digits.count == 4 {
                let y = toNumber(digits)
                if y >= 1100 && y <= 2099 {
                    emit(yearToDe(y))
                    i = j
                    continue
                }
            }
            emit(intToDe(toNumber(digits)))
            i = j
        }
        return out
    }

    // MARK: - Word-level fallbacks over the dictionary

    // Tried shortest suffix first (longest stem wins).
    private static let inflections: [(String, String, Bool)] = [
        ("e", "ə", true), ("n", "n", false), ("s", "s", false), ("t", "t", false),
        ("er", "ɜ", true), ("te", "tə", false), ("st", "st", false), ("en", "ən", true),
        ("em", "əm", true), ("es", "əs", true), ("et", "ət", true),
        ("ung", "ʊŋ", true), ("ern", "ɜn", true), ("ers", "ɜs", true), ("ert", "ɜt", true),
        ("est", "əst", true), ("ten", "tən", false), ("tes", "təs", false), ("tet", "tət", false),
        ("ungen", "ʊŋən", true), ("innen", "ɪnən", true),
    ]

    private static let linking: [(String, String)] = [("", ""), ("s", "s"), ("n", "n"), ("en", "ən"), ("es", "əs"), ("er", "ɜ"), ("e", "ə")]

    /// A final devoiced consonant becomes voiced again before a vowel suffix
    /// when the spelling ends in b/d/g/s (Tag -> Tage, Haus -> Häuser).
    private static func revoice(_ stemPhon: String, _ stemGraph: [Unicode.Scalar]) -> String {
        guard !stemPhon.isEmpty, let lastLetter = stemGraph.last else { return stemPhon }
        guard lastLetter == "b" || lastLetter == "d" || lastLetter == "g" || lastLetter == "s" else { return stemPhon }
        if endsWith(stemGraph, "ss") || lastLetter == "ß" { return stemPhon }
        switch stemPhon.last {
        case "p": return String(stemPhon.dropLast()) + "b"
        case "t": return String(stemPhon.dropLast()) + "d"
        case "k": return String(stemPhon.dropLast()) + "ɡ"
        case "s": return String(stemPhon.dropLast()) + "z"
        default: return stemPhon
        }
    }

    private static func joinSuffix(_ stemPhon: String, _ stemGraph: [Unicode.Scalar], _ phon: String, vowelInitial: Bool) -> String {
        var sp = stemPhon
        if vowelInitial {
            sp = revoice(sp, stemGraph)
            if sp.hasSuffix("ɜ") { sp = String(sp.dropLast()) + "ər" }      // geringer -> geringere
            if endsWith(stemGraph, "ig") && sp.hasSuffix("ɪç") { sp = String(sp.dropLast()) + "ɡ" }  // willig -> willigen
        } else {
            if sp.hasSuffix("b") { sp = String(sp.dropLast()) + "p" }
            else if sp.hasSuffix("d") { sp = String(sp.dropLast()) + "t" }
            else if sp.hasSuffix("z") { sp = String(sp.dropLast()) + "s" }
            else if sp.hasSuffix("ɡ") { sp = String(sp.dropLast()) + "k" }
            if sp.hasSuffix("r") { sp = String(sp.dropLast()) + "ɾ" }        // kritisieren -> kritisierst
        }
        return sp + phon
    }

    private static func inflected(_ word: [Unicode.Scalar], _ dict: Dictionary) -> String? {
        for (graph, phon, vowelInitial) in inflections {
            let gl = graph.unicodeScalars.count
            guard endsWith(word, graph), word.count - gl >= 3 else { continue }
            let stem = Array(word[0..<(word.count - gl)])
            if countNuclei(stem) == 0 { continue }
            if let base = dict[string(stem)] {
                return joinSuffix(base, stem, phon, vowelInitial: vowelInitial)
            }
            if let infinitive = dict[string(stem) + "en"], infinitive.hasSuffix("ən") {
                return joinSuffix(String(infinitive.dropLast(2)), stem, phon, vowelInitial: vowelInitial)
            }
        }
        return nil
    }

    private static func compound(_ word: [Unicode.Scalar], _ dict: Dictionary, depth: Int) -> String? {
        if depth > 2 || word.count < 7 { return nil }
        var cut = word.count - 3
        while cut > 3 {
            defer { cut -= 1 }
            let head = Array(word[0..<cut])
            guard let hp = dict[string(head)], countNuclei(head) > 0 else { continue }
            for (link, linkPhon) in linking {
                let rest = Array(word[cut...])
                if !link.isEmpty && !startsWith(rest, 0, link) { continue }
                let tail = Array(rest[link.unicodeScalars.count...])
                if tail.count < 3 || countNuclei(tail) == 0 { continue }
                var tp: String
                if let direct = dict[string(tail)] { tp = direct }
                else if let inf = inflected(tail, dict) { tp = inf }
                else if tail.count >= 7, let sub = compound(tail, dict, depth: depth + 1) { tp = sub }
                else { continue }
                // The head keeps its primary stress; the tail's primary becomes
                // secondary when the compound has three or more syllables
                // (Eisenbahn, Hauptbahnhof) and is dropped otherwise (Bahnhof).
                let tailPhon = countNuclei(word) >= 3 ? demotePrimary(tp) : stripStress(tp)
                var headPhon = hp
                let vowelFollows = link.isEmpty
                    ? phonemeIsVowel(firstScalar(stripMarksPrefix(tailPhon)))
                    : isVowel(link.unicodeScalars.first!)
                if vowelFollows {
                    headPhon = revoice(headPhon, head)   // Kind|es|haus, Tag|e|buch
                    if headPhon.hasSuffix("ɜ") { headPhon = String(headPhon.dropLast()) + "ər" }  // Kleider|ordnung
                }
                return headPhon + linkPhon + tailPhon
            }
        }
        return nil
    }
}
