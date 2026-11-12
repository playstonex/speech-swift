// Adapted from MOSS-VoiceGenerator and MOSS-Audio-Tokenizer.
// Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
// Licensed under Apache 2.0; see LICENSE.
// Native Swift adaptation uses request-local state and bounded causal caches.

import Foundation

/// Request-local state for the upstream staggered sixteen-codebook schedule.
struct MVGDelayPattern {
    static let start=151652,end=151653,gen=151656,delay=151662,imEnd=151645,pad=1024
    var isAudio=false
    var stopped=false
    var audioLength=0
    var delayedLength:Int?=nil
    var step=0
    private(set) var frames:[[Int32]]=[]
    private var delayedRows:[[Int32]]=[]
    private(set) var firstCodebookFrames=0
    private(set) var segments=0

    var forcedText:Int? {
        guard let count=delayedLength else { return nil }
        return count<16 ? Self.delay : Self.end
    }
    var activeCodebooks:[Int] {
        (0..<16).filter { audioLength>$0 && (delayedLength == nil || $0>(delayedLength!-1)) }
    }
    func allowedText(_ token:Int)->Bool {
        if step<=16,token==Self.imEnd { return false }
        if isAudio { return token==Self.gen || token==Self.delay }
        return ![151643,Self.gen,Self.delay,Self.end].contains(token)
    }
    /// Returns at most one newly complete frame. Padding is checked before it
    /// can reach the codec, so a broken schedule cannot silently become noise.
    mutating func append(text:Int,audio:[Int32])throws->[Int32]? {
        guard !stopped,audio.count==16 else { throw MossVoiceGeneratorError.invalidOutput("invalid delayed generation row") }
        if let forcedText {
            guard text==forcedText else { throw MossVoiceGeneratorError.invalidOutput("incorrect delayed end token") }
        } else if !allowedText(text) {
            throw MossVoiceGeneratorError.invalidOutput("text token violates audio schedule")
        }
        let active=Set(activeCodebooks)
        for index in audio.indices {
            let valid=active.contains(index) ? (0..<1024).contains(audio[index]) : audio[index]==1024
            guard valid else { throw MossVoiceGeneratorError.invalidOutput("invalid audio token or padding") }
        }
        if active.contains(0) { firstCodebookFrames+=1 }
        var ready:[Int32]?
        if text==Self.start {
            guard !isAudio else { throw MossVoiceGeneratorError.invalidOutput("nested audio segment") }
            isAudio=true;delayedRows=[];segments+=1
        } else if text==Self.end {
            guard isAudio,delayedRows.count>=16 else { throw MossVoiceGeneratorError.invalidOutput("empty audio segment") }
            isAudio=false;delayedRows=[]
        } else if isAudio {
            delayedRows.append(audio)
            if delayedRows.count>=16 {
                let start=delayedRows.count-16
                let frame=(0..<16).map {delayedRows[start+$0][$0]}
                guard frame.allSatisfy({(0..<1024).contains($0)}) else {
                    throw MossVoiceGeneratorError.invalidOutput("incomplete undelayed codebook frame")
                }
                frames.append(frame);ready=frame
            }
        }
        if text==Self.start || text==Self.gen || text==Self.delay { audioLength+=1 }
        if text==Self.end { audioLength=0 }
        if delayedLength==nil,text==Self.delay { delayedLength=0 }
        if let count=delayedLength { delayedLength=count+1>16 ? nil : count+1 }
        if text==Self.imEnd { stopped=true }
        step+=1
        return ready
    }
    func validateComplete()throws {
        guard stopped,!isAudio,!frames.isEmpty,delayedLength==nil else {
            throw MossVoiceGeneratorError.generationLimit
        }
    }
}
