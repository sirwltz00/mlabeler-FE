package mlabeler.core.oto

import mlabeler.core.dsp.HenkanAudioFeatures
import mlabeler.core.dsp.HenkanBoundaryDetector
import mlabeler.core.dsp.HenkanPhoneClass
import mlabeler.core.model.IntervalTier
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Applies Henkan-style acoustic boundary analysis to MLabeler's existing auto-oto
 * timings. This class does not generate oto entries: AutoOto remains responsible
 * for that. All code is common Kotlin, so the same refinement runs on JVM,
 * Android, iOS and WebAssembly.
 */
object HenkanOtoRefiner {
    private const val SEARCH_SECONDS = 0.080
    private const val MIN_CONFIDENCE = 0.40

    /** Refine aligned phone boundaries while retaining the original tier as a fallback. */
    fun refinePhones(
        samples: FloatArray,
        sampleRate: Int,
        phones: IntervalTier,
        maxShiftSeconds: Double = 0.050,
    ): IntervalTier {
        if (phones.size < 2 || samples.isEmpty()) return phones
        val features = runCatching { HenkanAudioFeatures(samples, sampleRate) }.getOrNull() ?: return phones
        if (features.frameCount < 3) return phones
        val bounds = phones.bounds.toMutableList()
        var changed = false

        for (boundaryIndex in 1 until phones.size) {
            val leftPhone = phoneForAnalysis(phones, boundaryIndex - 1)
            val rightPhone = phoneForAnalysis(phones, boundaryIndex)
            val leftClass = leftPhone.first
            val rightClass = rightPhone.first
            val hint = phones.bounds[boundaryIndex]
            val leftStart = phones.bounds[boundaryIndex - 1]
            val rightEnd = phones.bounds[boundaryIndex + 1]
            val searchLo = max(leftStart + 0.005, hint - SEARCH_SECONDS)
            val searchHi = min(rightEnd - 0.005, hint + SEARCH_SECONDS)
            if (searchHi - searchLo < 0.015) continue

            val estimate = HenkanBoundaryDetector.findBoundary(
                features, leftClass, rightClass, hint, searchLo, searchHi,
                toPhone = rightPhone.second, fromPhone = leftPhone.second,
            )
            if (!estimate.detected) continue
            var confidence = HenkanBoundaryDetector.boundaryConfidence(features, estimate.time, searchLo, searchHi)
            // Existing aligner label confidence is useful corroboration, but it is
            // not treated as acoustic boundary confidence and never forces a move.
            val leftModelConfidence = phones.confidenceOf(boundaryIndex - 1)
            val rightModelConfidence = phones.confidenceOf(boundaryIndex)
            val modelConfidence = listOfNotNull(leftModelConfidence, rightModelConfidence).minOrNull()
            if (modelConfidence != null) confidence *= 0.85 + 0.15 * modelConfidence.coerceIn(0f, 1f)
            if (confidence < MIN_CONFIDENCE) continue

            val leftDuration = hint - leftStart
            val rightDuration = rightEnd - hint
            if (leftDuration <= 0.004 || rightDuration <= 0.004) continue
            val shiftLimit = min(maxShiftSeconds, min(leftDuration, rightDuration) * 0.35)
            val proposed = estimate.time.coerceIn(hint - shiftLimit, hint + shiftLimit)
            val blend = ((confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE)).coerceIn(0.0, 0.70)
            val minAllowed = bounds[boundaryIndex - 1] + 0.002
            val maxAllowed = bounds[boundaryIndex + 1] - 0.002
            if (minAllowed > maxAllowed) continue
            val refined = (hint + (proposed - hint) * blend).coerceIn(minAllowed, maxAllowed)
            if (abs(refined - hint) >= 0.001) {
                bounds[boundaryIndex] = refined
                changed = true
            }
        }
        return if (changed) phones.copy(bounds = bounds) else phones
    }

    /** Refine MLabeler's built-in syllable segmentation using local acoustic evidence. */
    fun refineTimings(
        samples: FloatArray,
        sampleRate: Int,
        timings: List<SyllableTiming>,
        maxShiftSeconds: Double = 0.045,
    ): List<SyllableTiming> {
        if (timings.size < 2 || samples.isEmpty()) return timings
        val features = runCatching { HenkanAudioFeatures(samples, sampleRate) }.getOrNull() ?: return timings
        if (features.frameCount < 3) return timings
        val result = timings.toMutableList()

        // Improve the first onset as well when the recording contains an initial
        // silence. If the recording starts directly with sound, no edge is found
        // and MLabeler's original start is left untouched.
        val first = result.first()
        val firstPhone = first.syllable.consonant.trim().split(Regex("\\s+")).firstOrNull()
            ?.takeIf { it.isNotEmpty() } ?: first.syllable.vowel
        val firstHint = first.cStart
        val firstLo = max(0.0, firstHint - 0.090)
        val firstHi = min(first.vStart - 0.005, firstHint + 0.080)
        if (firstHint > 0.005 && firstHi - firstLo >= 0.015) {
            val onset = HenkanBoundaryDetector.findBoundary(
                features, HenkanPhoneClass.SILENCE, classify(firstPhone),
                firstHint, firstLo, firstHi, toPhone = firstPhone,
            )
            if (onset.detected) {
                val confidence = HenkanBoundaryDetector.boundaryConfidence(features, onset.time, firstLo, firstHi)
                if (confidence >= MIN_CONFIDENCE) {
                    val limit = min(0.035, firstHint - firstLo)
                    val proposed = onset.time.coerceIn(firstHint - limit, firstHint + limit)
                    val blend = ((confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE)).coerceIn(0.0, 0.65)
                    val refined = firstHint + (proposed - firstHint) * blend
                    val delta = refined - firstHint
                    result[0] = first.copy(cStart = refined, vStart = max(refined, first.vStart + delta))
                }
            }
        }

        for (index in 1 until result.size) {
            val previous = result[index - 1]
            val current = result[index]
            val fromClass = if (previous.syllable.vowel.isNotEmpty()) HenkanPhoneClass.VOWEL
                else classify(previous.syllable.consonant)
            val currentPhone = current.syllable.consonant.trim().split(Regex("\\s+")).firstOrNull()
                ?.takeIf { it.isNotEmpty() } ?: current.syllable.vowel
            val toClass = classify(currentPhone)
            val hint = current.cStart
            val searchLo = max(previous.cStart + 0.010, hint - SEARCH_SECONDS)
            val searchHi = min(current.vEnd - 0.005, hint + SEARCH_SECONDS)
            if (searchHi - searchLo < 0.015) continue
            val estimate = HenkanBoundaryDetector.findBoundary(
                features, fromClass, toClass, hint, searchLo, searchHi,
                toPhone = currentPhone, fromPhone = previous.syllable.vowel.ifEmpty { previous.syllable.consonant },
            )
            if (!estimate.detected) continue
            val confidence = HenkanBoundaryDetector.boundaryConfidence(features, estimate.time, searchLo, searchHi)
            if (confidence < MIN_CONFIDENCE) continue
            val durationLimit = min(maxShiftSeconds, min(hint - previous.cStart, current.vEnd - hint).coerceAtLeast(0.0) * 0.30)
            if (durationLimit <= 0.001) continue
            val proposed = estimate.time.coerceIn(hint - durationLimit, hint + durationLimit)
            val blend = ((confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE)).coerceIn(0.0, 0.65)
            val refined = hint + (proposed - hint) * blend
            val delta = refined - hint
            if (abs(delta) < 0.001) continue

            result[index - 1] = previous.copy(vEnd = refined.coerceAtLeast(previous.vStart + 0.002))
            result[index] = current.copy(
                cStart = refined,
                vStart = max(refined, current.vStart + delta),
            )
        }

        // A trailing silence can also provide a reliable endpoint; if the recording
        // ends during speech there is no detected edge and the original end remains.
        val lastIndex = result.lastIndex
        val last = result[lastIndex]
        val lastPhone = last.syllable.vowel.ifEmpty { last.syllable.consonant }
        val finalHint = last.vEnd
        val duration = samples.size.toDouble() / sampleRate
        val finalLo = max(last.cStart + 0.010, finalHint - 0.090)
        val finalHi = min(duration, finalHint + 0.035)
        if (finalHi - finalLo >= 0.015) {
            val estimate = HenkanBoundaryDetector.findBoundary(
                features, classify(lastPhone), HenkanPhoneClass.SILENCE,
                finalHint, finalLo, finalHi, fromPhone = lastPhone,
            )
            if (estimate.detected) {
                val confidence = HenkanBoundaryDetector.boundaryConfidence(features, estimate.time, finalLo, finalHi)
                if (confidence >= MIN_CONFIDENCE) {
                    val limit = min(0.030, max(0.0, duration - finalHint))
                    val refined = estimate.time.coerceIn(finalHint - 0.030, finalHint + limit)
                    result[lastIndex] = last.copy(vEnd = max(last.vStart + 0.002, refined))
                }
            }
        }
        return result
    }

    /**
     * Henkan treats VOCALOID's Japanese dz label as a voiced fricative only
     * between vowels; at a word/sample onset it remains an affricate/stop.
     */
    private fun phoneForAnalysis(phones: IntervalTier, index: Int): Pair<HenkanPhoneClass, String> {
        val raw = phones.texts.getOrNull(index).orEmpty().trim()
        if (raw.equals("dz", ignoreCase = true)) {
            val before = phones.texts.getOrNull(index - 1).orEmpty()
            val after = phones.texts.getOrNull(index + 1).orEmpty()
            if (classify(before) == HenkanPhoneClass.VOWEL && classify(after) == HenkanPhoneClass.VOWEL) {
                return HenkanPhoneClass.VOICED_CONSONANT to "z"
            }
        }
        return classify(raw) to raw
    }

    /** Map common IPA, ARPAbet, Japanese-romaji and simple Latin labels to Henkan classes. */
    fun classify(rawPhone: String): HenkanPhoneClass {
        val trimmed = rawPhone.trim()
        if (trimmed.isEmpty()) return HenkanPhoneClass.SILENCE
        if (trimmed in setOf("D", "Z", "h\\")) return HenkanPhoneClass.VOICED_CONSONANT
        val phone = trimmed.substringBefore(' ').lowercase().trimEnd('0', '1', '2')
        if (phone in setOf("sil", "sp", "ap", "pau", "pause", "<sil>", "_", "-")) return HenkanPhoneClass.SILENCE

        val vowels = setOf(
            "a", "e", "i", "o", "u", "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey",
            "ih", "ix", "iy", "ow", "oy", "uh", "uw", "ux", "ɑ", "ɐ", "ɒ", "æ", "ə", "ɚ", "ɛ", "ɜ", "ɝ",
            "ɞ", "ʌ", "ɤ", "ɘ", "ɵ", "œ", "ø", "ɪ", "ʊ", "ʉ", "ɯ", "ɨ", "ɶ", "ɒ̃", "ɔ", "ɐ̞",
        )
        if (phone in vowels || (phone.length == 1 && phone[0] in "aeiou")) return HenkanPhoneClass.VOWEL

        val unvoicedStops = setOf("p", "t", "k", "q", "c", "ʔ", "ʈ", "ch", "ts", "tʃ", "tɕ", "tS", "pf", "kx")
        if (phone in unvoicedStops || phone.startsWith("ts") || phone.startsWith("ch")) return HenkanPhoneClass.UNVOICED_STOP
        val unvoicedFricatives = setOf("f", "s", "sh", "h", "x", "th", "θ", "ɸ", "ç", "ɕ", "ʃ", "χ", "ħ", "ɧ")
        if (phone in unvoicedFricatives || phone.startsWith("sh")) return HenkanPhoneClass.UNVOICED_FRICATIVE
        val voicedStops = setOf("b", "d", "g", "j", "ɡ", "ɖ", "ɟ", "ɢ", "dz", "jh", "dʒ", "dʑ", "ʤ")
        if (phone in voicedStops || phone.startsWith("dz")) return HenkanPhoneClass.VOICED_STOP
        val voicedConsonants = setOf(
            "v", "z", "zh", "r", "l", "m", "n", "ng", "w", "y", "dh",
            "ʒ", "ð", "β", "ɣ", "ɦ", "ɹ", "ɾ", "ɽ", "ɻ", "ʋ", "ɱ", "ŋ", "ɲ", "ɳ", "ʁ", "ʎ", "ʝ",
        )
        if (phone in voicedConsonants || phone.firstOrNull() in setOf('m', 'n', 'l', 'r', 'v', 'z', 'w', 'y')) return HenkanPhoneClass.VOICED_CONSONANT
        // Romaji digraphs keep their manner when the exact bank spelling is not IPA.
        if (phone.startsWith("k") || phone.startsWith("p") || phone.startsWith("t") || phone.startsWith("c")) return HenkanPhoneClass.UNVOICED_STOP
        if (phone.startsWith("f") || phone.startsWith("s") || phone.startsWith("h") || phone.startsWith("x")) return HenkanPhoneClass.UNVOICED_FRICATIVE
        if (phone.startsWith("b") || phone.startsWith("d") || phone.startsWith("g")) return HenkanPhoneClass.VOICED_STOP
        return HenkanPhoneClass.VOICED_CONSONANT
    }
}
