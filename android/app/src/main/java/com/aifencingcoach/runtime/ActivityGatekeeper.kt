package com.aifencingcoach.runtime

class ActivityGatekeeper(
    private val fps: Int = 30,
    private val activeKneeAngleDeg: Float = 170f,
    private val idleKneeAngleDeg: Float = 174f,
    private val motionThresholdPx: Float = 1.5f
) {
    var state: String = StateIdle
        private set

    private var frameCount = 0
    private var activeTriggerCount = 0
    private val activeTriggerThreshold = 5
    private var idleTriggerCount = 0
    private val idleTriggerThreshold = 2 * fps
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
                "reason" to "missing_target",
                "active_trigger_count" to activeTriggerCount,
                "idle_trigger_count" to idleTriggerCount
            )
            return state == StateActive
        }

        val kneeAngle = frontKneeAngle(targetSkeleton, targetSide) ?: 180f
        val shoulderWidth = shoulderWidth(targetSkeleton) ?: 100f
        val isTurnedBack = shoulderWidth < frameWidth * 0.05f
        val tooFar = fencerDistanceTooLarge(targetSkeleton, opponentSkeleton, frameWidth)
        val pelvisCenter = FencingGeometry.pelvisCenter(targetSkeleton)
        val pelvisMotion = if (pelvisCenter != null && lastPelvisCenter != null) {
            pelvisCenter.distanceTo(lastPelvisCenter!!)
        } else {
            0f
        }
        val moving = lastPelvisCenter == null || pelvisMotion >= motionThresholdPx
        if (pelvisCenter != null) lastPelvisCenter = pelvisCenter

        val enGardePosture = kneeAngle < activeKneeAngleDeg
        val enGarde = enGardePosture && (moving || state != StateIdle || activeTriggerCount > 0)
        val standingUp = kneeAngle > idleKneeAngleDeg
        val stopCondition = standingUp || isTurnedBack || tooFar

        when (state) {
            StateIdle -> if (enGarde) {
                state = StateChecking
                activeTriggerCount = 1
            }
            StateChecking -> if (enGarde) {
                activeTriggerCount += 1
                if (activeTriggerCount >= activeTriggerThreshold) {
                    state = StateActive
                    idleTriggerCount = 0
                }
            } else {
                state = StateIdle
                activeTriggerCount = 0
            }
            StateActive -> if (stopCondition) {
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
            "active_knee_angle_deg" to activeKneeAngleDeg,
            "idle_knee_angle_deg" to idleKneeAngleDeg,
            "en_garde" to enGarde,
            "en_garde_posture" to enGardePosture,
            "standing_up" to standingUp,
            "shoulder_width" to shoulderWidth,
            "turned_back" to isTurnedBack,
            "too_far" to tooFar,
            "pelvis_motion" to pelvisMotion,
            "moving" to moving,
            "active_trigger_count" to activeTriggerCount,
            "idle_trigger_count" to idleTriggerCount
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

    private fun fencerDistanceTooLarge(target: Skeleton, opponent: Skeleton?, frameWidth: Int): Boolean {
        val opponentSkeleton = opponent ?: return false
        val targetCenter = FencingGeometry.pelvisCenter(target) ?: return false
        val opponentCenter = FencingGeometry.pelvisCenter(opponentSkeleton) ?: return false
        return kotlin.math.abs(targetCenter.x - opponentCenter.x) > frameWidth * 0.6f
    }

    companion object {
        const val StateIdle = "IDLE"
        const val StateChecking = "CHECKING"
        const val StateActive = "ACTIVE"
    }
}
