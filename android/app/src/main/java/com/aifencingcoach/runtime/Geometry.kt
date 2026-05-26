package com.aifencingcoach.runtime

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class LimbMap(
    val hip: String,
    val knee: String,
    val ankle: String,
    val wrist: String,
    val elbow: String,
    val shoulder: String
)

object FencingGeometry {
    val frontLimbs = mapOf(
        TargetSide.LEFT to LimbMap(
            hip = "right_hip",
            knee = "right_knee",
            ankle = "right_ankle",
            wrist = "front_wrist",
            elbow = "front_elbow",
            shoulder = "front_shoulder"
        ),
        TargetSide.RIGHT to LimbMap(
            hip = "left_hip",
            knee = "left_knee",
            ankle = "left_ankle",
            wrist = "front_wrist",
            elbow = "front_elbow",
            shoulder = "front_shoulder"
        )
    )

    fun joint(skeleton: Skeleton, name: String): Point2? = skeleton[name]

    fun pelvisCenter(skeleton: Skeleton): Point2? {
        val leftHip = skeleton["left_hip"] ?: return null
        val rightHip = skeleton["right_hip"] ?: return null
        return Point2((leftHip.x + rightHip.x) / 2f, (leftHip.y + rightHip.y) / 2f)
    }

    fun angle(a: Point2, b: Point2, c: Point2): Float {
        val bax = a.x - b.x
        val bay = a.y - b.y
        val bcx = c.x - b.x
        val bcy = c.y - b.y
        val baNorm = sqrt(bax * bax + bay * bay)
        val bcNorm = sqrt(bcx * bcx + bcy * bcy)
        if (baNorm < 1e-8f || bcNorm < 1e-8f) return 180f
        val cos = ((bax * bcx + bay * bcy) / (baNorm * bcNorm)).coerceIn(-1f, 1f)
        return Math.toDegrees(acos(cos).toDouble()).toFloat()
    }

    fun horizontalRange(points: List<Point2>): Float {
        if (points.isEmpty()) return 0f
        var lo = points.first().x
        var hi = points.first().x
        for (point in points) {
            lo = min(lo, point.x)
            hi = max(hi, point.x)
        }
        return hi - lo
    }

    fun bboxHeight(skeletons: List<Skeleton>): Float {
        var lo = Float.POSITIVE_INFINITY
        var hi = Float.NEGATIVE_INFINITY
        for (skeleton in skeletons) {
            for (point in skeleton.values) {
                lo = min(lo, point.y)
                hi = max(hi, point.y)
            }
        }
        return if (lo.isFinite() && hi.isFinite()) abs(hi - lo) else 0f
    }
}
