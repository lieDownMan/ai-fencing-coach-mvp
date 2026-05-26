package com.aifencingcoach.runtime

class SpatialNormalizer {
    private var referenceNose: Point2? = null
    private var scaleFactor: Float? = null

    fun reset() {
        referenceNose = null
        scaleFactor = null
    }

    fun fit(skeleton: Skeleton) {
        val nose = skeleton["nose"] ?: error("nose not found in first frame skeleton")
        val frontAnkle = skeleton["front_ankle"] ?: error("front_ankle not found in first frame skeleton")
        referenceNose = nose
        val verticalDistance = kotlin.math.abs(frontAnkle.y - nose.y)
        scaleFactor = if (verticalDistance < 1e-6f) 1f else verticalDistance
    }

    fun normalize(skeleton: Skeleton): Skeleton {
        val nose = referenceNose ?: error("Normalizer not fitted")
        val scale = scaleFactor ?: error("Normalizer not fitted")
        return skeleton.mapValues { (_, point) ->
            Point2((point.x - nose.x) / scale, (point.y - nose.y) / scale)
        }
    }

    fun modelArray(normalizedSkeleton: Skeleton): FloatArray {
        val out = FloatArray(ModelJoints.size * 2)
        ModelJoints.forEachIndexed { index, joint ->
            val point = normalizedSkeleton[joint] ?: Point2(0f, 0f)
            out[index * 2] = point.x
            out[index * 2 + 1] = point.y
        }
        return out
    }

    companion object {
        val ModelJoints = listOf(
            "front_wrist",
            "front_elbow",
            "front_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        )
    }
}
