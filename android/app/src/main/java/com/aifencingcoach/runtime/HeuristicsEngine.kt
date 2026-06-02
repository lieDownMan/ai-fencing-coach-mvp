package com.aifencingcoach.runtime

import kotlin.math.abs
import kotlin.math.atan2

class HeuristicsEngine(
    private var targetSide: TargetSide = TargetSide.LEFT,
    private var trainingMode: TrainingMode = TrainingMode.FREE_BOUTING
) {
    private var evaluationCount = 0

    fun configure(targetSide: TargetSide, trainingMode: TrainingMode) {
        this.targetSide = targetSide
        this.trainingMode = trainingMode
        reset()
    }

    fun reset() {
        evaluationCount = 0
    }

    fun evaluate(action: String, skeletons: List<Skeleton>): List<String> {
        if (skeletons.isEmpty()) return emptyList()
        evaluationCount += 1

        val triggered = mutableListOf<String>()
        val isOffensive = action in OffensiveActions

        if (evaluationCount > BounceWarmupEvaluations) {
            checkBounce(skeletons)?.let(triggered::add)
        }

        checkLunge(skeletons)?.let(triggered::add)
        checkCenterOfMass(skeletons)?.let(triggered::add)
        checkStanceTooHigh(skeletons)?.let(triggered::add)
        checkGuard(skeletons)?.let(triggered::add)
        checkStepWidth(skeletons)?.let(triggered::add)
        checkOverParrying(skeletons)?.let(triggered::add)
        checkHandTooHigh(skeletons)?.let(triggered::add)

        if (trainingMode == TrainingMode.TARGET_PRACTICE && isOffensive) {
            checkFootBeforeHand(skeletons)?.let(triggered::add)
            checkIncompleteArmExtension(skeletons)?.let(triggered::add)
        }

        return triggered.distinct()
    }

    private fun checkBounce(skeletons: List<Skeleton>): String? {
        val pelvisYs = mutableListOf<Float>()
        val allYs = mutableListOf<Float>()
        for (skel in skeletons) {
            FencingGeometry.pelvisCenter(skel)?.let { pelvisYs.add(it.y) }
            for (point in skel.values) allYs.add(point.y)
        }
        if (pelvisYs.size < BounceMinPelvisSamples || allYs.size < 2) return null
        val bboxHeight = allYs.maxOrNull()!! - allYs.minOrNull()!!
        if (bboxHeight < 1e-4f) return null
        val deltaY = pelvisYs.maxOrNull()!! - pelvisYs.minOrNull()!!
        return if (deltaY > BounceRatioThreshold * bboxHeight) "bounce_excessive" else null
    }

    private fun checkLunge(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        var minAngle = Float.POSITIVE_INFINITY
        for (skel in skeletons) {
            val hip = skel[limbs.hip] ?: continue
            val knee = skel[limbs.knee] ?: continue
            val ankle = skel[limbs.ankle] ?: continue
            minAngle = minOf(minAngle, FencingGeometry.angle(hip, knee, ankle))
        }
        if (!minAngle.isFinite()) return null
        return if (minAngle < LungeKneeMinAngleDeg) "lunge_overextension" else null
    }

    private fun checkGuard(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val threshold = if (trainingMode == TrainingMode.FREE_BOUTING) {
            GuardDroppedFreeBoutingThresholdFrames
        } else {
            GuardDroppedThresholdFrames
        }
        var consecutive = 0
        for (skel in skeletons) {
            val wrist = skel[limbs.wrist]
            val pelvis = FencingGeometry.pelvisCenter(skel)
            if (wrist != null && pelvis != null && wrist.y > pelvis.y) {
                consecutive += 1
                if (consecutive > threshold) return "guard_dropped"
            } else {
                consecutive = 0
            }
        }
        return null
    }

    private fun checkFootBeforeHand(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val refWrist = skeletons.first()[limbs.wrist] ?: return null
        val refAnkle = skeletons.first()[limbs.ankle] ?: return null
        val minDisplacement = bodyScale(skeletons) * FootBeforeHandMinDisplacementRatio
        var maxWristDisp = 0f
        var maxAnkleDisp = 0f
        var wristPeakFrame = 0
        var anklePeakFrame = 0

        for (i in skeletons.indices) {
            val skel = skeletons[i]
            skel[limbs.wrist]?.let { wrist ->
                val distance = abs(wrist.x - refWrist.x)
                if (distance > maxWristDisp) {
                    maxWristDisp = distance
                    wristPeakFrame = i
                }
            }
            skel[limbs.ankle]?.let { ankle ->
                val distance = abs(ankle.x - refAnkle.x)
                if (distance > maxAnkleDisp) {
                    maxAnkleDisp = distance
                    anklePeakFrame = i
                }
            }
        }

        return if (
            maxAnkleDisp > minDisplacement &&
            maxWristDisp > minDisplacement &&
            anklePeakFrame < wristPeakFrame
        ) {
            "foot_before_hand"
        } else {
            null
        }
    }

    private fun checkStanceTooHigh(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val angles = skeletons.mapNotNull { skel ->
            val hip = skel[limbs.hip] ?: return@mapNotNull null
            val knee = skel[limbs.knee] ?: return@mapNotNull null
            val ankle = skel[limbs.ankle] ?: return@mapNotNull null
            FencingGeometry.angle(hip, knee, ankle)
        }
        if (angles.size < 3) return null
        val avg = angles.average()
        return if (avg > StanceTooHighAngleDeg) "stance_too_high" else null
    }

    private fun checkIncompleteArmExtension(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val refWrist = skeletons.first()[limbs.wrist] ?: return null
        var maxDisp = 0f
        var peakSkel = skeletons.first()

        for (skel in skeletons) {
            val wrist = skel[limbs.wrist] ?: continue
            val distance = abs(wrist.x - refWrist.x)
            if (distance > maxDisp) {
                maxDisp = distance
                peakSkel = skel
            }
        }

        val shoulder = peakSkel[limbs.shoulder] ?: return null
        val elbow = peakSkel[limbs.elbow] ?: return null
        val wrist = peakSkel[limbs.wrist] ?: return null
        val angle = FencingGeometry.angle(shoulder, elbow, wrist)
        return if (angle < IncompleteArmExtensionAngleDeg) "incomplete_arm_extension" else null
    }

    private fun checkOverParrying(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        var shoulderWidth: Float? = null

        for (skel in skeletons) {
            val shoulder = skel[limbs.shoulder]
            val otherShoulderName = if (targetSide == TargetSide.LEFT) "left_shoulder" else "right_shoulder"
            val otherShoulder = skel[otherShoulderName]
            if (shoulder != null && otherShoulder != null) {
                shoulderWidth = shoulder.distanceTo(otherShoulder)
                break
            }
            val pelvis = FencingGeometry.pelvisCenter(skel)
            if (shoulder != null && pelvis != null) {
                shoulderWidth = shoulder.distanceTo(pelvis) * OverParryShoulderMultiplier
                break
            }
        }

        val width = shoulderWidth ?: return null
        if (width < 1e-6f) return null
        val wristPositions = skeletons.mapNotNull { it[limbs.wrist] }
        if (wristPositions.size < OverParryMinWristSamples) return null

        var maxSweep = 0f
        for (i in wristPositions.indices) {
            for (j in i + 1 until wristPositions.size) {
                maxSweep = maxOf(maxSweep, wristPositions[i].distanceTo(wristPositions[j]))
            }
        }
        return if (maxSweep > OverParryRatioThreshold * width) "over_parrying" else null
    }

    private fun checkStepWidth(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val backAnkleKey = if (targetSide == TargetSide.LEFT) "left_ankle" else "right_ankle"
        val minShoulderWidth = bodyScale(skeletons) * StepMinShoulderWidthRatio

        for (skel in skeletons) {
            val frontAnkle = skel[limbs.ankle] ?: continue
            val backAnkle = skel[backAnkleKey] ?: continue
            val frontShoulder = skel[limbs.shoulder] ?: continue
            val pelvis = FencingGeometry.pelvisCenter(skel) ?: continue
            val shoulderProxy = abs(frontShoulder.x - pelvis.x) * StepShoulderProxyMultiplier
            if (shoulderProxy < minShoulderWidth) continue

            val ratio = frontAnkle.distanceTo(backAnkle) / shoulderProxy
            if (ratio < NarrowStepRatioThreshold) return "narrow_step"
            if (ratio > WideStepRatioThreshold) return "wide_step"
        }
        return null
    }

    private fun checkCenterOfMass(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val backAnkleKey = if (targetSide == TargetSide.LEFT) "left_ankle" else "right_ankle"
        val backShoulderKey = if (targetSide == TargetSide.LEFT) "left_shoulder" else "right_shoulder"

        for (skel in skeletons) {
            val frontAnkle = skel[limbs.ankle] ?: continue
            val backAnkle = skel[backAnkleKey] ?: continue
            val pelvis = FencingGeometry.pelvisCenter(skel) ?: continue

            neckCenter(skel)?.let { neck ->
                val dx = neck.x - pelvis.x
                val dy = pelvis.y - neck.y
                if (dy > 0f) {
                    val thetaDeg = atan2(dx, dy).toDegrees()
                    val forwardTiltDeg = if (frontAnkle.x > backAnkle.x) thetaDeg else -thetaDeg
                    if (forwardTiltDeg > SpineForwardTiltThresholdDeg) return "center_of_mass_in_front"
                    if (forwardTiltDeg < -SpineBackwardTiltThresholdDeg) {
                        return "center_of_mass_leaning_backward"
                    }
                }
            }

            val frontShoulder = skel[limbs.shoulder]
            val backShoulder = skel[backShoulderKey]
            if (frontShoulder != null && backShoulder != null) {
                val dy = frontShoulder.y - backShoulder.y
                val dx = abs(frontShoulder.x - backShoulder.x)
                if (dx > 0.001f || abs(dy) > 0.001f) {
                    val shoulderTiltDeg = atan2(dy, dx).toDegrees()
                    if (shoulderTiltDeg > ShoulderForwardTiltThresholdDeg) return "center_of_mass_in_front"
                    if (shoulderTiltDeg < -ShoulderBackwardTiltThresholdDeg) {
                        return "center_of_mass_leaning_backward"
                    }
                }
            }
        }
        return null
    }

    private fun checkHandTooHigh(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        for (skel in skeletons) {
            val elbow = skel[limbs.elbow] ?: continue
            val wrist = skel[limbs.wrist] ?: continue
            val dy = elbow.y - wrist.y
            val dx = abs(wrist.x - elbow.x)
            if (dy > 0f && atan2(dy, dx).toDegrees() > HandTooHighMinAngleDeg) {
                return "hand_too_high"
            }
        }
        return null
    }

    private fun bodyScale(skeletons: List<Skeleton>): Float =
        FencingGeometry.bboxHeight(skeletons).coerceAtLeast(1f)

    private fun neckCenter(skeleton: Skeleton): Point2? {
        val left = skeleton["left_shoulder"] ?: return null
        val right = skeleton["right_shoulder"] ?: return null
        return Point2((left.x + right.x) / 2f, (left.y + right.y) / 2f)
    }

    private fun Float.toDegrees(): Float =
        Math.toDegrees(toDouble()).toFloat()

    companion object {
        private const val BounceMinPelvisSamples = 5
        private const val BounceRatioThreshold = 0.33f
        private const val BounceWarmupEvaluations = 15
        private const val LungeKneeMinAngleDeg = 120f
        private const val GuardDroppedThresholdFrames = 5
        private const val GuardDroppedFreeBoutingThresholdFrames = 10
        private const val FootBeforeHandMinDisplacementRatio = 0.01f
        private const val StanceTooHighAngleDeg = 160.0
        private const val IncompleteArmExtensionAngleDeg = 155f
        private const val OverParryMinWristSamples = 5
        private const val OverParryShoulderMultiplier = 2f
        private const val OverParryRatioThreshold = 3f
        private const val StepShoulderProxyMultiplier = 2.5f
        private const val StepMinShoulderWidthRatio = 0.02f
        private const val WideStepRatioThreshold = 3f
        private const val NarrowStepRatioThreshold = 1.2f
        private const val SpineForwardTiltThresholdDeg = 15f
        private const val SpineBackwardTiltThresholdDeg = 10f
        private const val ShoulderForwardTiltThresholdDeg = 15f
        private const val ShoulderBackwardTiltThresholdDeg = 15f
        private const val HandTooHighMinAngleDeg = 60f

        private val OffensiveActions = setOf("R", "JS", "WW", "IS")
    }
}
