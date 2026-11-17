package mlabeler.core

import kotlin.math.PI
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import mlabeler.core.dsp.HenkanAudioFeatures
import mlabeler.core.dsp.HenkanBoundaryDetector
import mlabeler.core.dsp.HenkanPhoneClass
import mlabeler.core.oto.HenkanOtoRefiner
import mlabeler.core.oto.Syllable
import mlabeler.core.oto.SyllableTiming
import mlabeler.core.model.IntervalTier

class HenkanAudioAnalysisTest {
    private fun voicedSegment(sampleRate: Int = 16000): FloatArray = FloatArray(sampleRate) { i ->
        if (i in (sampleRate / 10) until (sampleRate * 35 / 100)) {
            (0.35 * sin(2.0 * PI * 220.0 * i / sampleRate)).toFloat()
        } else 0f
    }

    @Test
    fun extractsHenkanFeatureGridAndFiniteMeasurements() {
        val sr = 16000
        val features = HenkanAudioFeatures(voicedSegment(sr), sr)
        assertEquals(193, features.frameCount)
        assertEquals(24, features.bandCount)
        assertTrue(features.energy.all { it.isFinite() })
        assertTrue(features.voicingConfidence.all { it in 0f..1f })
        assertTrue(features.spectralFlux.all { it.isFinite() })
        assertTrue(features.voiced.any { it })
    }

    @Test
    fun detectsAcousticOnsetAndOffsetAroundRoughHints() {
        val sr = 16000
        val features = HenkanAudioFeatures(voicedSegment(sr), sr)
        val onset = HenkanBoundaryDetector.findBoundary(
            features, HenkanPhoneClass.SILENCE, HenkanPhoneClass.VOWEL,
            0.10, 0.02, 0.18,
        )
        val offset = HenkanBoundaryDetector.findBoundary(
            features, HenkanPhoneClass.VOWEL, HenkanPhoneClass.SILENCE,
            0.35, 0.27, 0.43,
        )
        assertTrue(onset.detected)
        assertTrue(offset.detected)
        assertTrue(kotlin.math.abs(onset.time - 0.10) < 0.04)
        assertTrue(kotlin.math.abs(offset.time - 0.35) < 0.04)
        assertTrue(HenkanBoundaryDetector.boundaryConfidence(features, onset.time, 0.02, 0.18) > 0.5)
        assertTrue(HenkanBoundaryDetector.boundaryConfidence(features, offset.time, 0.27, 0.43) > 0.5)
    }

    private fun tone(seconds: Double, amplitude: Double, hz: Double = 220.0): FloatArray {
        val sr = 44100
        return FloatArray((seconds * sr).toInt()) { i ->
            (amplitude * sin(2.0 * PI * hz * i / sr)).toFloat()
        }
    }

    private fun noise(seconds: Double, amplitude: Double, seed: Int): FloatArray {
        val sr = 44100
        var state = seed.toLong()
        return FloatArray((seconds * sr).toInt()) {
            state = state * 6364136223846793005L + 1442695040888963407L
            (((state ushr 33).toInt() / 2147483648.0) * amplitude).toFloat()
        }
    }

    private fun concat(vararg chunks: FloatArray): FloatArray {
        val out = FloatArray(chunks.sumOf { it.size })
        var offset = 0
        for (chunk in chunks) { chunk.copyInto(out, offset); offset += chunk.size }
        return out
    }

    private fun affricatedVoicedFricative(): FloatArray {
        val toneWithNoise = tone(0.12, 0.20)
        val hiss = noise(0.12, 0.035, 4)
        val fricative = FloatArray(toneWithNoise.size) { (toneWithNoise[it] + hiss[it]).coerceIn(-1f, 1f) }
        return concat(tone(0.18), noise(0.04, 0.0002, 3), noise(0.004, 0.25, 4), fricative, tone(0.20))
    }

    @Test
    fun detectsMedialVoicedFricativeAfterStopClosure() {
        val features = HenkanAudioFeatures(affricatedVoicedFricative(), 44100)
        val onset = HenkanBoundaryDetector.findBoundary(
            features, HenkanPhoneClass.VOWEL, HenkanPhoneClass.VOICED_CONSONANT,
            0.20, 0.10, 0.29, toPhone = "z", fromPhone = "a",
        )
        val offset = HenkanBoundaryDetector.findBoundary(
            features, HenkanPhoneClass.VOICED_CONSONANT, HenkanPhoneClass.VOWEL,
            0.34, 0.27, 0.40, toPhone = "a", fromPhone = "z",
        )
        assertTrue(onset.detected)
        assertTrue(onset.time in 0.21..0.235)
        assertTrue(offset.detected)
        assertTrue(kotlin.math.abs(offset.time - 0.344) <= 0.012)
    }

    @Test
    fun refinesMedialDzAsVoicedFricativeButKeepsMonotonicIntervals() {
        val audio = affricatedVoicedFricative()
        val duration = audio.size.toDouble() / 44100.0
        val tier = IntervalTier("phones", listOf(0.0, 0.18, 0.34, duration), listOf("a", "dz", "a"))
        val refined = HenkanOtoRefiner.refinePhones(audio, 44100, tier)
        assertTrue(refined.bounds[1] > tier.bounds[1])
        assertTrue(refined.bounds[2] >= tier.bounds[2])
        assertTrue(refined.bounds.zipWithNext().all { (a, b) -> b > a })
    }

    @Test
    fun mapsCommonPhonemeLabelsToAnalysisClasses() {
        assertEquals(HenkanPhoneClass.VOWEL, HenkanOtoRefiner.classify("aa1"))
        assertEquals(HenkanPhoneClass.UNVOICED_STOP, HenkanOtoRefiner.classify("k"))
        assertEquals(HenkanPhoneClass.UNVOICED_FRICATIVE, HenkanOtoRefiner.classify("sh"))
        assertEquals(HenkanPhoneClass.VOICED_STOP, HenkanOtoRefiner.classify("d"))
        assertEquals(HenkanPhoneClass.VOICED_CONSONANT, HenkanOtoRefiner.classify("z"))
        assertEquals(HenkanPhoneClass.VOICED_STOP, HenkanOtoRefiner.classify("dz"))
        assertEquals(HenkanPhoneClass.SILENCE, HenkanOtoRefiner.classify("SP"))
    }

    @Test
    fun emptyAudioUsesOriginalOtoTimingAsSafeFallback() {
        val tier = IntervalTier("phones", listOf(0.0, 0.1, 0.2), listOf("k", "a"))
        assertTrue(HenkanOtoRefiner.refinePhones(FloatArray(0), 16000, tier) === tier)
        val timings = listOf(SyllableTiming(Syllable("ka", "k", "a"), 0.0, 0.05, 0.2))
        assertTrue(HenkanOtoRefiner.refineTimings(FloatArray(0), 16000, timings) === timings)
    }
}
