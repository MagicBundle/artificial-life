import Testing

@testable import AlifeCore

@Suite struct NumbersTests {
    @Test func thousandsAreGroupedByACommaWhateverTheLocale() {
        // G3a's Mac day: "6.216 born" read as a decimal on a European machine (DESIGN §34.7).
        #expect(Numbers.grouped(6216) == "6,216" && Numbers.grouped(42_873) == "42,873" && Numbers.grouped(1_000_000) == "1,000,000")
        #expect(Numbers.grouped(0) == "0" && Numbers.grouped(999) == "999" && Numbers.grouped(1000) == "1,000")
        #expect(Numbers.grouped(-1234) == "-1,234" && Numbers.grouped(-12) == "-12")
        #expect(Numbers.grouped(Int64(450_250)) == "450,250")
        #expect(Numbers.grouped(Int.min).hasPrefix("-9,223,372"))
    }

    @Test func theLineageAndInspectorTextsUseIt() {
        #expect(CastPolicy.deathSummary([3091, 3099, 0, 0, 5]) == "3,091 starved, 3,099 killed, 5 storm")
    }
}
