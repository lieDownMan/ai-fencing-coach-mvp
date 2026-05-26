package com.aifencingcoach.runtime

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp

class FenceNetClassifier private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String,
    private val outputName: String,
    private val confidenceThreshold: Float = 0.6f
) : AutoCloseable {
    fun classify(window: List<FloatArray>, frameIndex: Int): ActionPrediction? {
        if (window.size < WindowSize) return null
        val input = FloatArray(Channels * WindowSize)
        val recent = window.takeLast(WindowSize)
        for (t in 0 until WindowSize) {
            val frame = recent[t]
            for (channel in 0 until Channels) {
                input[channel * WindowSize + t] = frame.getOrElse(channel) { 0f }
            }
        }

        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            longArrayOf(1, Channels.toLong(), WindowSize.toLong())
        ).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val raw = result.get(0).value
                val logits = when (raw) {
                    is Array<*> -> (raw[0] as FloatArray)
                    is FloatArray -> raw
                    else -> return null
                }
                val scores = softmax(logits)
                val bestIndex = scores.indices.maxByOrNull { scores[it] } ?: return null
                val confidence = scores[bestIndex]
                val action = if (confidence >= confidenceThreshold) ClassNames[bestIndex] else "Idle"
                return ActionPrediction(
                    action = action,
                    confidence = confidence,
                    startFrame = maxOf(0, frameIndex - WindowSize + 1),
                    endFrame = frameIndex + 1
                )
            }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val WindowSize = 28
        const val Stride = 10
        const val Channels = 18
        val ClassNames = listOf("R", "IS", "WW", "JS", "SF", "SB")

        fun createOrNull(context: Context): FenceNetClassifier? = runCatching {
            val env = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open("fencenet_v2.onnx").use { it.readBytes() }
            val session = env.createSession(modelBytes, OrtSession.SessionOptions())
            FenceNetClassifier(
                env = env,
                session = session,
                inputName = session.inputNames.first(),
                outputName = session.outputNames.first()
            )
        }.getOrNull()

        fun softmax(logits: FloatArray): FloatArray {
            val max = logits.maxOrNull() ?: 0f
            val exps = logits.map { exp((it - max).toDouble()).toFloat() }
            val sum = exps.sum().coerceAtLeast(1e-8f)
            return FloatArray(exps.size) { index -> exps[index] / sum }
        }
    }
}
