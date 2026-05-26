package com.aifencingcoach.runtime

class ActivityGatekeeper(private val fps: Int = 30) {
    var state: String = StateIdle
        private set

    private var frameCount = 0
    private var activeTriggerCount = 0
    private val activeTriggerThreshold = 5
    private var idleTriggerCount = 0
    private val idleTriggerThreshold = 2 * fps

    fun reset() {
        state = StateIdle
        frameCount = 0
        activeTriggerCount = 0
        idleTriggerCount = 0
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
            return state == StateActive
        }

        val kneeAngle = frontKneeAngle(targetSkeleton, targetSide) ?: 180f
        val shoulderWidth = shoulderWidth(targetSkeleton) ?: 100f
        val isTurnedBack = shoulderWidth < frameWidth * 0.05f
        val tooFar = fencerDistanceTooLarge(targetSkeleton, opponentSkeleton, frameWidth)
        val enGarde = kneeAngle < 175f
        val stopCondition = kneeAngle > 180f || isTurnedBack || tooFar

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
