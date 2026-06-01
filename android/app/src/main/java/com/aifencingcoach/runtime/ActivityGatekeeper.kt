package com.aifencingcoach.runtime

import kotlin.math.sqrt

class ActivityGatekeeper(
    private val fps: Int = 30,
    private val activeKneeAngleDeg: Float = 176f,
    private val idleKneeAngleDeg: Float = 178f,
    private val motionThresholdNorm: Float = 0.005f
) {
    var state: String = StateIdle
        private set

    private var frameCount = 0
    private var activeTriggerCount = 0
    private var activeTriggerThreshold = 5
    private var idleTriggerCount = 0
    private var idleTriggerThreshold = 2 * fps
    private var lastPelvisCenter: Point2? = null

    var lastReasons: Map<String, Any?> = emptyMap()
        private set

    fun reset() {
        state = StateIdle
        frameCount = 0
        activeTriggerCount = 0
        idleTriggerCount = 0
        lastPelvisCenter = null
        lastReasons = emptyMap()
    }

    fun shouldExtractPose(): Boolean {
        frameCount += 1
        if (state == StateIdle) {
            val skipRate = maxOf(1, fps / 5)
            return frameCount % skipRate == 1
        }
        return true
    }

    fun update(
        targetSkeleton: Skeleton?,
        opponentSkeleton: Skeleton?,
        frameWidth: Int,
        targetSide: TargetSide
    ): Boolean {
        if (targetSkeleton == null) {
            if (state == StateActive) {
                idleTriggerCount += 1
                if (idleTriggerCount >= idleTriggerThreshold) {
                    state = StateIdle
                    idleTriggerCount = 0
                }
            } else if (state == StateChecking) {
                state = StateIdle
                activeTriggerCount = 0
            }
            lastReasons = mapOf(
                "has_target" to false,
                "state" to state,
                "reason" to "missing_target"
            )
            return state == StateActive
        }

        val kneeAngle = frontKneeAngle(targetSkeleton, targetSide) ?: 180f
        val shoulderWidth = shoulderWidth(targetSkeleton)
        val isTurnedBack = (shoulderWidth ?: frameWidth.toFloat()) < frameWidth * 0.05f

        val pelvisCenter = FencingGeometry.pelvisCenter(targetSkeleton)
        var pelvisMotion = 0f
        if (pelvisCenter != null && lastPelvisCenter != null) {
            val dx = pelvisCenter.x - lastPelvisCenter!!.x
            val dy = pelvisCenter.y - lastPelvisCenter!!.y
            pelvisMotion = sqrt(dx * dx + dy * dy) / frameWidth.toFloat()
        }

        val moving = lastPelvisCenter == null || pelvisMotion >= motionThresholdNorm
        if (pelvisCenter != null) {
            lastPelvisCenter = pelvisCenter
        }

        val enGardePosture = kneeAngle < activeKneeAngleDeg
        val enGarde = enGardePosture && (moving || state != StateIdle || activeTriggerCount > 0)
        val standingUp = kneeAngle > idleKneeAngleDeg
        val stopCondition = standingUp || isTurnedBack

        if (state == StateIdle) {
            if (enGarde) {
                state = StateChecking
                activeTriggerCount = 1
            }
        } else if (state == StateChecking) {
            if (enGarde) {
                activeTriggerCount += 1
                if (activeTriggerCount >= activeTriggerThreshold) {
                    state = StateActive
                    idleTriggerCount = 0
                }
            } else {
                state = StateIdle
                activeTriggerCount = 0
            }
        } else if (state == StateActive) {
            if (stopCondition) {
                idleTriggerCount += 1
                if (idleTriggerCount >= idleTriggerThreshold) {
                    state = StateIdle
                    idleTriggerCount = 0
                }
            } else {
                idleTriggerCount = 0
            }
        }

        lastReasons = mapOf(
            "has_target" to true,
            "state" to state,
            "knee_angle" to kneeAngle,
            "en_garde" to enGarde,
            "en_garde_posture" to enGardePosture,
            "standing_up" to standingUp,
            "turned_back" to isTurnedBack,
            "moving" to moving
        )
        return state == StateActive
    }

    private fun frontKneeAngle(skeleton: Skeleton, targetSide: TargetSide): Float? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val hip = skeleton[limbs.hip] ?: return null
        val knee = skeleton[limbs.knee] ?: return null
        val ankle = skeleton[limbs.ankle] ?: return null
        return FencingGeometry.angle(hip, knee, ankle)
    }

    private fun shoulderWidth(skeleton: Skeleton): Float? {
        val left = skeleton["left_shoulder"]
        val right = skeleton["right_shoulder"]
        return if (left != null && right != null) left.distanceTo(right) else null
    }

    companion object {
        const val StateIdle = "IDLE"
        const val StateChecking = "CHECKING"
        const val StateActive = "ACTIVE"
    }
}
