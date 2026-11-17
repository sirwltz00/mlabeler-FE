package mlabeler.core.dsp

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Henkan-style short-time acoustic features used to refine phoneme boundaries.
 *
 * This is a Kotlin Multiplatform implementation: it lives in commonMain and uses
 * no Python runtime, native library, desktop process, or network service. The
 * feature grid and principal descriptors follow Henkan FE's u2v/audio.py:
 * 5 ms hop, 40 ms long window, 12 ms edge window, normalized autocorrelation,
 * log-frequency spectral bands, spectral flux, and combined voicing confidence.
 */
class HenkanAudioFeatures(samples: FloatArray, val sampleRate: Int) {
    val frameCount: Int
    val bandCount: Int
    val energy: FloatArray
    val period: FloatArray
    val lag: FloatArray
    val spectrum: FloatArray // frame-major, Henkan log-frequency bands
    val hfRatio: FloatArray
    val spectralCentroid: FloatArray
    val spectralFlatness: FloatArray
    val shortEnergy: FloatArray
    val shortHfRatio: FloatArray
    val shortHfDb: FloatArray
    val spectralFlux: FloatArray
    val periodicityConfidence: FloatArray
    val energyConfidence: FloatArray
    val harmonicBalanceConfidence: FloatArray
    val hnrProxyDb: FloatArray
    val voicingConfidence: FloatArray
    val voiced: BooleanArray
    val peak: Float
    val noiseFloor: Float
    val silenceThreshold: Float

    init {
        require(sampleRate >= 8000) { "Henkan audio analysis requires a sample rate of at least 8000 Hz" }
        val x = FloatArray(samples.size) { i ->
            val v = samples[i]
            when {
                v.isNaN() -> 0f
                v == Float.POSITIVE_INFINITY -> 1f
                v == Float.NEGATIVE_INFINITY -> -1f
                else -> v.coerceIn(-1f, 1f)
            }
        }
        val hop = max(1, (HOP_SECONDS * sampleRate).roundToInt())
        val win = max(32, (0.040 * sampleRate).roundToInt())
        frameCount = if (x.size >= win) max(1, (x.size - win) / hop + 1) else 1
        energy = FloatArray(frameCount)
        period = FloatArray(frameCount)
        lag = FloatArray(frameCount)
        hfRatio = FloatArray(frameCount)
        spectralCentroid = FloatArray(frameCount)
        spectralFlatness = FloatArray(frameCount)
        shortEnergy = FloatArray(frameCount)
        shortHfRatio = FloatArray(frameCount)
        shortHfDb = FloatArray(frameCount)

        val window = DoubleArray(win) { i -> hann(i, win) }
        val nfft = nextPowerOfTwo(2 * win)
        val magBins = nfft / 2
        val edges = bandEdges(magBins)
        bandCount = edges.size - 1
        spectrum = FloatArray(frameCount * bandCount)
        val loLag = max(1, sampleRate / 900)
        val hiLag = min(win - 1, sampleRate / 70)
        require(hiLag > loLag) { "Sample rate/window is too short for pitch analysis" }
        val windowAutocorrelation = DoubleArray(hiLag) { lagIndex ->
            var sum = 0.0
            for (i in 0 until win - lagIndex) sum += window[i] * window[i + lagIndex]
            sum
        }

        val shortWin = max(32, (0.012 * sampleRate).roundToInt())
        val shortHalf = shortWin / 2
        val shortWindow = DoubleArray(shortWin) { i -> hann(i, shortWin) }
        val shortNfft = nextPowerOfTwo(shortWin * 2)
        val shortBins = shortNfft / 2 + 1
        val shortHfStart = (3000.0 * shortNfft / sampleRate).toInt().coerceIn(0, shortBins - 1)
        val shortLfStart = (80.0 * shortNfft / sampleRate).toInt().coerceIn(0, shortBins - 1)
        val shortLfEnd = max(shortLfStart + 1, (1000.0 * shortNfft / sampleRate).toInt()).coerceAtMost(shortBins)

        val longFft = Fft(nfft)
        val shortFft = Fft(shortNfft)
        val re = DoubleArray(nfft)
        val im = DoubleArray(nfft)
        val shortRe = DoubleArray(shortNfft)
        val shortIm = DoubleArray(shortNfft)
        val magnitudes = DoubleArray(magBins)
        val binFreq = DoubleArray(magBins) { it.toDouble() * sampleRate / nfft }
        val bandRanges = edges.zipWithNext().map { (a, b) -> a until b }
        val hfStart = (3000.0 * nfft / sampleRate).toInt().coerceIn(0, magBins - 1)
        val lfStart = (80.0 * nfft / sampleRate).toInt().coerceIn(0, magBins - 1)
        val lfEnd = max(lfStart + 1, (1000.0 * nfft / sampleRate).toInt()).coerceAtMost(magBins)

        for (f in 0 until frameCount) {
            val start = f * hop
            var mean = 0.0
            for (j in 0 until win) mean += if (start + j < x.size) x[start + j].toDouble() else 0.0
            mean /= win
            var squareSum = 0.0
            for (j in 0 until win) {
                val v = ((if (start + j < x.size) x[start + j].toDouble() else 0.0) - mean) * window[j]
                squareSum += v * v
                re[j] = v
                im[j] = 0.0
            }
            for (j in win until nfft) { re[j] = 0.0; im[j] = 0.0 }
            energy[f] = (20.0 * log10(sqrt(squareSum / win + 1e-12) + 1e-9)).toFloat()
            longFft.transform(re, im)

            var magSum = 0.0
            var weightedFreq = 0.0
            var logMagSum = 0.0
            for (b in 0 until magBins) {
                val m = sqrt(re[b] * re[b] + im[b] * im[b])
                magnitudes[b] = m
                magSum += m
                weightedFreq += m * binFreq[b]
                logMagSum += ln(m + 1e-9)
            }
            spectralCentroid[f] = (weightedFreq / max(magSum, 1e-12)).toFloat()
            val geometric = exp(logMagSum / max(1, magBins))
            spectralFlatness[f] = (geometric / (magSum / max(1, magBins) + 1e-9)).toFloat().coerceIn(0f, 1f)
            for ((band, range) in bandRanges.withIndex()) {
                var sum = 0.0
                var count = 0
                for (b in range) if (b < magBins) { sum += magnitudes[b]; count++ }
                spectrum[f * bandCount + band] = ln(sum / max(1, count) + 1e-6).toFloat()
            }
            var hf = 0.0
            for (b in hfStart until magBins) hf += magnitudes[b]
            var lf = 0.0
            for (b in lfStart until lfEnd) lf += magnitudes[b]
            hf /= max(1, magBins - hfStart)
            lf /= max(1, lfEnd - lfStart)
            hfRatio[f] = ln((hf + 1e-9) / (lf + 1e-9)).toFloat()

            // Wiener–Khinchin autocorrelation, normalized by the Hann-window autocorrelation.
            for (b in 0 until nfft) {
                val p = re[b] * re[b] + im[b] * im[b]
                re[b] = p
                im[b] = 0.0
            }
            inverseFft(longFft, re, im)
            val ac0 = re[0] + 1e-12
            var best = 0.0
            var bestLag = loLag
            for (lagIndex in loLag until hiLag) {
                val windowNorm = max(windowAutocorrelation[lagIndex] / max(windowAutocorrelation[0], 1e-12), 1e-6)
                val normalized = re[lagIndex] / ac0 / windowNorm
                if (normalized > best) { best = normalized; bestLag = lagIndex }
            }
            period[f] = best.coerceIn(0.0, 1.0).toFloat()
            lag[f] = bestLag.toFloat()

            // The short window is centred at 20 ms + frame * hop, as in Henkan FE.
            val centre = (0.020 * sampleRate).roundToInt() + f * hop
            val shortStart = centre - shortHalf
            var shortMean = 0.0
            for (j in 0 until shortWin) shortMean += sampleAt(x, shortStart + j)
            shortMean /= shortWin
            var shortSquare = 0.0
            for (j in 0 until shortWin) {
                val v = (sampleAt(x, shortStart + j) - shortMean) * shortWindow[j]
                shortSquare += v * v
                shortRe[j] = v
                shortIm[j] = 0.0
            }
            for (j in shortWin until shortNfft) { shortRe[j] = 0.0; shortIm[j] = 0.0 }
            shortEnergy[f] = (20.0 * log10(sqrt(shortSquare / shortWin + 1e-12) + 1e-9)).toFloat()
            shortFft.transform(shortRe, shortIm)
            var shf = 0.0
            var slf = 0.0
            var shfCount = 0
            var slfCount = 0
            for (b in 0 until shortBins) {
                val m = sqrt(shortRe[b] * shortRe[b] + shortIm[b] * shortIm[b])
                if (b >= shortHfStart) { shf += m; shfCount++ }
                if (b >= shortLfStart && b < shortLfEnd) { slf += m; slfCount++ }
            }
            shf = if (shfCount > 0) shf / shfCount else 0.0
            slf = if (slfCount > 0) slf / slfCount else 0.0
            shortHfRatio[f] = ln((shf + 1e-9) / (slf + 1e-9)).toFloat()
            shortHfDb[f] = (20.0 * log10(shf + 1e-9)).toFloat()
        }

        spectralFlux = FloatArray(frameCount)
        for (f in 1 until frameCount) {
            var sum = 0.0
            var norm = 0.0
            for (b in 0 until bandCount) {
                val d = max(0f, spectrum[f * bandCount + b] - spectrum[(f - 1) * bandCount + b]).toDouble()
                sum += d * d
                val s = spectrum[f * bandCount + b].toDouble()
                norm += s * s
            }
            spectralFlux[f] = (sqrt(sum) / sqrt(norm / bandCount + 1e-6)).toFloat()
        }

        peak = percentile(energy, 0.99f)
        noiseFloor = percentile(energy, 0.05f)
        silenceThreshold = if (peak < -140f) max(-140f, peak + 6f) else min(max(noiseFloor + 8f, peak - 45f), peak - 35f)
        periodicityConfidence = FloatArray(frameCount) { ((period[it] - 0.22f) / 0.68f).coerceIn(0f, 1f) }
        energyConfidence = FloatArray(frameCount) { f ->
            var local = -Float.MAX_VALUE
            for (k in max(0, f - 30)..min(frameCount - 1, f + 30)) local = max(local, energy[k])
            val gate = max(peak - 40f, local - 20f)
            (1.0 / (1.0 + exp((-(energy[f] - gate) / 4f).toDouble().coerceIn(-30.0, 30.0)))).toFloat()
        }
        harmonicBalanceConfidence = FloatArray(frameCount) { ((0.55f - hfRatio[it]) / 1.1f).coerceIn(0f, 1f) }
        hnrProxyDb = FloatArray(frameCount) {
            (10.0 * log10(max(period[it], 1e-5f) / max(1f - period[it], 1e-5f))).toFloat()
        }
        voicingConfidence = FloatArray(frameCount) {
            (0.64f * periodicityConfidence[it] + 0.22f * energyConfidence[it] + 0.14f * harmonicBalanceConfidence[it]).coerceIn(0f, 1f)
        }
        voiced = BooleanArray(frameCount) {
            voicingConfidence[it] >= 0.57f && energy[it] > silenceThreshold && period[it] > 0.32f
        }
    }

    fun timeToFrame(seconds: Double): Int = ((seconds - 0.020) / HOP_SECONDS).roundToInt().coerceIn(0, frameCount - 1)
    fun frameToTime(frame: Int): Double = frame * HOP_SECONDS + 0.020
    fun isSilent(frame: Int): Boolean = energy[frame.coerceIn(0, frameCount - 1)] < silenceThreshold

    fun smoothEnergy(first: Int, last: Int, radius: Int = 3): FloatArray {
        val a = first.coerceIn(0, frameCount - 1)
        val b = last.coerceIn(a, frameCount - 1)
        return FloatArray(b - a + 1) { ix ->
            val f = a + ix
            var sum = 0f
            var count = 0
            for (k in max(0, f - radius)..min(frameCount - 1, f + radius)) { sum += energy[k]; count++ }
            sum / max(1, count)
        }
    }

    companion object {
        const val HOP_SECONDS = 0.005
        private fun sampleAt(x: FloatArray, i: Int): Double = if (i in x.indices) x[i].toDouble() else 0.0
        private fun hann(i: Int, n: Int): Double = if (n <= 1) 1.0 else 0.5 - 0.5 * cos(2.0 * PI * i / (n - 1))
        private fun nextPowerOfTwo(n: Int): Int { var p = 1; while (p < n) p = p shl 1; return p }

        private fun bandEdges(magBins: Int): List<Int> {
            val end = max(5, magBins)
            val out = mutableListOf<Int>()
            for (i in 0..24) {
                val value = if (i == 24) magBins else (4.0 * (end.toDouble() / 4.0).pow(i / 24.0)).toInt().coerceIn(0, magBins)
                if (out.lastOrNull() != value) out += value
            }
            if (out.lastOrNull() != magBins) out += magBins
            return out
        }

        private fun percentile(values: FloatArray, p: Float): Float {
            if (values.isEmpty()) return -180f
            val sorted = values.sorted()
            return sorted[((sorted.size - 1) * p).roundToInt().coerceIn(0, sorted.size - 1)]
        }

        private fun Double.pow(power: Double): Double = exp(ln(this) * power)

        private fun inverseFft(fft: Fft, re: DoubleArray, im: DoubleArray) {
            for (i in im.indices) im[i] = -im[i]
            fft.transform(re, im)
            val scale = 1.0 / re.size
            for (i in re.indices) { re[i] *= scale; im[i] = -im[i] * scale }
        }
    }
}

/** Acoustic class used by Henkan's transition detector. */
enum class HenkanPhoneClass { SILENCE, VOWEL, UNVOICED_STOP, UNVOICED_FRICATIVE, VOICED_STOP, VOICED_CONSONANT }

data class HenkanBoundaryEstimate(val time: Double, val detected: Boolean)

/** Port of Henkan FE's find_boundary() and boundary_confidence(). */
object HenkanBoundaryDetector {
    private const val HOP = 0.005
    private val unvoiced = setOf(HenkanPhoneClass.UNVOICED_STOP, HenkanPhoneClass.UNVOICED_FRICATIVE)

    fun findBoundary(
        features: HenkanAudioFeatures,
        from: HenkanPhoneClass,
        to: HenkanPhoneClass,
        hint: Double,
        lo: Double,
        hi: Double,
        toPhone: String? = null,
        fromPhone: String? = null,
    ): HenkanBoundaryEstimate {
        val f0 = features.timeToFrame(lo)
        val f1 = features.timeToFrame(hi)
        val fh = features.timeToFrame(hint)
        if (f1 - f0 < 2) return HenkanBoundaryEstimate(hint, false)
        val voiced = features.voiced
        val silence = BooleanArray(features.frameCount) { features.isSilent(it) }
        var candidate: Int? = null
        val range = (f0 + 1)..f1

        when {
            to == HenkanPhoneClass.SILENCE -> {
                candidate = nearest(range.filter { f -> !silence[f - 1] && silence[f] && (f until min(features.frameCount, f + 4)).all { silence[it] } }, fh)
            }
            from == HenkanPhoneClass.SILENCE -> {
                candidate = nearest(range.filter { f -> silence[f - 1] && !silence[f] && (f until min(features.frameCount, f + 3)).none { silence[it] } }, fh)
                if (candidate != null && to == HenkanPhoneClass.UNVOICED_STOP) candidate = max(f0, candidate - 4)
            }
            from in setOf(HenkanPhoneClass.VOWEL, HenkanPhoneClass.VOICED_CONSONANT, HenkanPhoneClass.VOICED_STOP) && to in unvoiced -> {
                candidate = nearest(range.filter { f -> voiced[f - 1] && !voiced[f] && (f until min(features.frameCount, f + 3)).none { voiced[it] } }, fh)
                val closure = to == HenkanPhoneClass.UNVOICED_STOP || toPhone in setOf("ts", "tS", "tɕ", "tʃ")
                val centre = candidate ?: fh
                val plateau = median((max(f0, centre - 25)..max(f0, centre - 8)).filter { it < features.frameCount }.map { features.energy[it].toDouble() })
                if (closure && plateau.isFinite()) {
                    val drop = (max(f0 + 1, centre - 25)..min(f1, centre)).firstOrNull { features.energy[it] < plateau - 6f }
                    if (drop != null) {
                        val previousCandidate = candidate
                        if (previousCandidate == null || drop < previousCandidate) candidate = drop
                    }
                }
            }
            from in unvoiced && to in setOf(HenkanPhoneClass.VOWEL, HenkanPhoneClass.VOICED_CONSONANT, HenkanPhoneClass.VOICED_STOP) -> {
                candidate = nearest(range.filter { f -> !voiced[f - 1] && voiced[f] && (f until min(features.frameCount, f + 3)).all { voiced[it] } }, fh)
            }
            from == HenkanPhoneClass.VOWEL && to == HenkanPhoneClass.VOICED_CONSONANT && isHenkanVoicedFricative(toPhone) -> {
                val edge = fricativeEdge(features, f0, f1, fh, +1)
                if (edge != null) return HenkanBoundaryEstimate(edge, true)
                return HenkanBoundaryEstimate(hint, false)
            }
            from == HenkanPhoneClass.VOICED_CONSONANT && to == HenkanPhoneClass.VOWEL && isHenkanVoicedFricative(fromPhone) -> {
                val edge = fricativeEdge(features, f0, f1, fh, -1)
                if (edge != null) return HenkanBoundaryEstimate(edge, true)
                return HenkanBoundaryEstimate(hint, false)
            }
            from == HenkanPhoneClass.VOWEL && to == HenkanPhoneClass.VOWEL -> {
                if (f1 - f0 > 8) {
                    var bestScore = -1.0
                    var bestFrame = -1
                    for (f in max(f0 + 4, 4)..f1) {
                        var d = 0.0
                        for (b in 0 until features.bandCount) {
                            val delta = features.spectrum[f * features.bandCount + b] - features.spectrum[(f - 4) * features.bandCount + b]
                            d += delta * delta
                        }
                        val transition = f - 2
                        val weight = exp(-0.5 * ((transition - fh).toDouble() / 12.0).let { it * it })
                        val score = sqrt(d) * weight
                        if (score > bestScore) { bestScore = score; bestFrame = transition }
                    }
                    if (bestFrame >= 0) candidate = bestFrame
                }
            }
        }

        if (candidate == null && from != to && from != HenkanPhoneClass.SILENCE && to != HenkanPhoneClass.SILENCE) {
            val smooth = features.smoothEnergy(f0, f1)
            var bestScore = 0.0
            var bestFrame: Int? = null
            for (i in 0 until smooth.size - 1) {
                val derivative = smooth[i + 1] - smooth[i]
                val frame = f0 + i + 1
                val weight = exp(-0.5 * ((frame - fh).toDouble() / 8.0).let { it * it })
                val falling = from == HenkanPhoneClass.VOWEL || (from != HenkanPhoneClass.VOWEL && to != HenkanPhoneClass.VOWEL && from == HenkanPhoneClass.VOICED_CONSONANT)
                val score = (if (falling) -derivative else derivative) * weight
                if (score > bestScore) { bestScore = score; bestFrame = frame }
            }
            if (bestScore > 0.5) candidate = bestFrame
        }
        val found = candidate
        return if (found == null) HenkanBoundaryEstimate(hint, false)
        else HenkanBoundaryEstimate(features.frameToTime(found) - HOP / 2.0, true)
    }

    fun boundaryConfidence(features: HenkanAudioFeatures, time: Double, lo: Double? = null, hi: Double? = null, detected: Boolean = true): Double {
        if (!detected || features.frameCount < 3 || !time.isFinite()) return 0.12
        val f = features.timeToFrame(time)
        val left = (max(0, f - 4)..max(0, f - 2)).filter { it < features.frameCount }
        val right = (min(features.frameCount - 1, f + 1)..min(features.frameCount - 1, f + 3)).filter { it in 0 until features.frameCount }
        if (left.isEmpty() || right.isEmpty()) return 0.25
        val specDelta = sqrt((0 until features.bandCount).sumOf { b ->
            val l = left.map { features.spectrum[it * features.bandCount + b].toDouble() }.average()
            val r = right.map { features.spectrum[it * features.bandCount + b].toDouble() }.average()
            val d = r - l
            d * d
        })
        val flux = (max(0, f - 2)..min(features.frameCount - 1, f + 2)).maxOfOrNull { features.spectralFlux[it].toDouble() } ?: 0.0
        val hfDelta = abs(right.map { features.shortHfRatio[it].toDouble() }.average() - left.map { features.shortHfRatio[it].toDouble() }.average())
        val energyDelta = abs(right.map { features.shortEnergy[it].toDouble() }.average() - left.map { features.shortEnergy[it].toDouble() }.average())
        val voiceDelta = abs(right.map { features.voicingConfidence[it].toDouble() }.average() - left.map { features.voicingConfidence[it].toDouble() }.average())
        val cues = listOf(
            (specDelta / 2.5).coerceIn(0.0, 1.0),
            (flux / 0.45).coerceIn(0.0, 1.0),
            (hfDelta / 0.8).coerceIn(0.0, 1.0),
            (energyDelta / 8.0).coerceIn(0.0, 1.0),
            (voiceDelta / 0.65).coerceIn(0.0, 1.0),
        )
        val evidence = 0.30 * cues[0] + 0.18 * cues[1] + 0.18 * cues[2] + 0.16 * cues[3] + 0.18 * cues[4]
        var confidence = 0.20 + 0.80 * evidence
        if (lo != null && hi != null && hi > lo) {
            val edgeDistance = min(abs(time - lo), abs(hi - time)) / max((hi - lo) * 0.15, 1e-4)
            confidence *= 0.75 + 0.25 * edgeDistance.coerceIn(0.0, 1.0)
        }
        return confidence.coerceIn(0.05, 1.0)
    }

    private fun fricativeEdge(ft: HenkanAudioFeatures, f0: Int, f1: Int, fh: Int, direction: Int): Double? {
        var best: Int? = null
        var bestScore = 0.0
        val lo = max(f0 + 4, 4)
        val hi = min(f1 - 4, ft.frameCount - 5)
        for (f in lo..hi) {
            val beforeH = median((f - 4 until f).map { ft.shortHfDb[it].toDouble() })
            val afterH = median((f + 1..f + 4).map { ft.shortHfDb[it].toDouble() })
            val change = direction * (afterH - beforeH)
            if (change < 3.0) continue
            val beforeE = median((f - 4 until f).map { ft.shortEnergy[it].toDouble() })
            val afterE = median((f + 1..f + 4).map { ft.shortEnergy[it].toDouble() })
            val recovery: Double
            if (direction > 0) {
                val futureLow = (f + 1 until min(ft.frameCount, f + 9)).minOfOrNull { ft.shortEnergy[it].toDouble() } ?: afterE
                if (beforeE - futureLow > 8.0) continue
                recovery = max(0.0, afterE - beforeE)
            } else recovery = max(0.0, beforeE - afterE)
            val distance = (f - fh) / 10.0
            val score = (change + 0.025 * recovery) * exp(-0.5 * distance * distance)
            if (score > bestScore) { best = f; bestScore = score }
        }
        val frame = best ?: return null
        var time = ft.frameToTime(frame) - HOP / 2.0
        if (direction > 0) {
            val prior = (max(0, frame - 22) until max(0, frame - 10)).map { ft.shortEnergy[it].toDouble() }
            val recent = (max(0, frame - 10) until frame).map { ft.shortEnergy[it].toDouble() }
            if (prior.size >= 4 && recent.size >= 2 && median(prior) - (recent.minOrNull() ?: median(recent)) > 8.0) time += 0.010
        }
        return min(time, ft.frameToTime(f1))
    }

    private fun isHenkanVoicedFricative(phone: String?): Boolean = phone in setOf("z", "Z", "v", "D", "h\\", "ʒ", "ð")
    private fun nearest(candidates: List<Int>, hint: Int): Int? = candidates.minByOrNull { abs(it - hint) }
    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return Double.NaN
        val s = values.sorted()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2.0
    }
}
