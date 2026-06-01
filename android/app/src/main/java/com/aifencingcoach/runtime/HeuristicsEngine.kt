package com.aifencingcoach.runtime

import kotlin.math.abs

class HeuristicsEngine(
    private var targetSide: TargetSide = TargetSide.LEFT,
    private var trainingMode: TrainingMode = TrainingMode.FREE_BOUTING
) {
    fun configure(targetSide: TargetSide, trainingMode: TrainingMode) {
        this.targetSide = targetSide
        this.trainingMode = trainingMode
    }

    fun evaluate(action: String, skeletons: List<Skeleton>): List<String> {
        if (skeletons.isEmpty()) return emptyList()
        val triggered = mutableListOf<String>()
        val isOffensive = action in OffensiveActions
        val isFootwork = action in FootworkActions

        // Footwork-specific checks
        if (trainingMode == TrainingMode.FOOTWORK && isFootwork) {
            checkBounce(skeletons)?.let(triggered::add)
            checkStanceTooHigh(skeletons)?.let(triggered::add)
            checkCenterOfMass(skeletons)?.let(triggered::add)
        }

        // Target Practice offensive checks
        if (trainingMode == TrainingMode.TARGET_PRACTICE && isOffensive) {
            checkLunge(skeletons)?.let(triggered::add)
            checkFootBeforeHand(skeletons)?.let(triggered::add)
            checkIncompleteArmExtension(skeletons)?.let(triggered::add)
        }

        // Guard dropped - all modes
        checkGuard(skeletons)?.let(triggered::add)

        // Step width - all modes unconditionally
        checkStepWidth(skeletons)?.let(triggered::add)

        // Footwork checks in non-Footwork modes
        if (trainingMode != TrainingMode.FOOTWORK && isFootwork) {
            checkStanceTooHigh(skeletons)?.let(triggered::add)
            checkBounce(skeletons)?.let(triggered::add)
            checkCenterOfMass(skeletons)?.let(triggered::add)
        }

        // Over-parrying
        if (action == "SB" || (trainingMode == TrainingMode.FREE_BOUTING && isFootwork)) {
            checkOverParrying(skeletons)?.let(triggered::add)
        }

        return triggered.distinct()
    }

    private fun checkBounce(skeletons: List<Skeleton>): String? {
        val pelvisYs = mutableListOf<Float>()
        val allYs = mutableListOf<Float>()
        for (skel in skeletons) {
            val pc = FencingGeometry.pelvisCenter(skel)
            if (pc != null) pelvisYs.add(pc.y)
            for (v in skel.values) {
                allYs.add(v.y)
            }
        }
        if (pelvisYs.size < 5 || allYs.size < 2) return null
        val bboxHeight = allYs.maxOrNull()!! - allYs.minOrNull()!!
        if (bboxHeight < 1e-4f) return null
        val deltaY = pelvisYs.maxOrNull()!! - pelvisYs.minOrNull()!!
        return if (deltaY > 0.33f * bboxHeight) "bounce_excessive" else null
    }

    private fun checkLunge(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val refAnkle = skeletons.first()[limbs.ankle] ?: return null
        var maxDisp = 0f
        var peakSkel = skeletons.first()
        for (skel in skeletons) {
            val ankle = skel[limbs.ankle]
            if (ankle != null) {
                val d = ankle.distanceTo(refAnkle)
                if (d > maxDisp) {
                    maxDisp = d
                    peakSkel = skel
                }
            }
        }
        val hip = peakSkel[limbs.hip] ?: return null
        val knee = peakSkel[limbs.knee] ?: return null
        val ankle = peakSkel[limbs.ankle] ?: return null
        val angle = FencingGeometry.angle(hip, knee, ankle)
        return if (angle < 90f) "lunge_overextension" else null
    }

    private fun checkGuard(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val threshold = if (trainingMode == TrainingMode.FREE_BOUTING) 20 else 10
        var consecutive = 0
        for (skel in skeletons) {
            val wrist = skel[limbs.wrist]
            val pelvis = FencingGeometry.pelvisCenter(skel)
            if (wrist != null && pelvis != null && wrist.y > pelvis.y) {
                consecutive++
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
        var maxWristDisp = 0f
        var maxAnkleDisp = 0f
        var wristPeakFrame = 0
        var anklePeakFrame = 0
        for (i in skeletons.indices) {
            val skel = skeletons[i]
            val wrist = skel[limbs.wrist]
            val ankle = skel[limbs.ankle]
            if (wrist != null) {
                val d = abs(wrist.x - refWrist.x)
                if (d > maxWristDisp) {
                    maxWristDisp = d
                    wristPeakFrame = i
                }
            }
            if (ankle != null) {
                val d = abs(ankle.x - refAnkle.x)
                if (d > maxAnkleDisp) {
                    maxAnkleDisp = d
                    anklePeakFrame = i
                }
            }
        }
        if (maxAnkleDisp > 5f && maxWristDisp > 5f && anklePeakFrame < wristPeakFrame) {
            return "foot_before_hand"
        }
        return null
    }

    private fun checkStanceTooHigh(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val angles = mutableListOf<Float>()
        for (skel in skeletons) {
            val hip = skel[limbs.hip]
            val knee = skel[limbs.knee]
            val ankle = skel[limbs.ankle]
            if (hip != null && knee != null && ankle != null) {
                angles.add(FencingGeometry.angle(hip, knee, ankle))
            }
        }
        if (angles.size < 3) return null
        val avg = angles.average()
        return if (avg > 173.0) "stance_too_high" else null
    }

    private fun checkIncompleteArmExtension(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val refWrist = skeletons.first()[limbs.wrist] ?: return null
        var maxDisp = 0f
        var peakSkel = skeletons.first()
        for (skel in skeletons) {
            val wrist = skel[limbs.wrist]
            if (wrist != null) {
                val d = abs(wrist.x - refWrist.x)
                if (d > maxDisp) {
                    maxDisp = d
                    peakSkel = skel
                }
            }
        }
        val shoulder = peakSkel[limbs.shoulder] ?: return null
        val elbow = peakSkel[limbs.elbow] ?: return null
        val wrist = peakSkel[limbs.wrist] ?: return null
        val angle = FencingGeometry.angle(shoulder, elbow, wrist)
        return if (angle < 155f) "incomplete_arm_extension" else null
    }

    private fun checkOverParrying(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        var shoulderWidth: Float? = null
        for (skel in skeletons) {
            val shoulder = skel[limbs.shoulder]
            val otherShoulderName = if (targetSide == TargetSide.RIGHT) "left_shoulder" else "right_shoulder"
            val otherShoulder = skel[otherShoulderName]
            if (otherShoulder == null) {
                val pelvis = FencingGeometry.pelvisCenter(skel)
                if (shoulder != null && pelvis != null) {
                    shoulderWidth = abs(shoulder.x - pelvis.x) * 2f
                    break
                }
            } else if (shoulder != null) {
                shoulderWidth = abs(shoulder.x - otherShoulder.x)
                break
            }
        }
        if (shoulderWidth == null || shoulderWidth < 1e-6f) return null
        val wristXs = mutableListOf<Float>()
        for (skel in skeletons) {
            val wrist = skel[limbs.wrist]
            if (wrist != null) wristXs.add(wrist.x)
        }
        if (wristXs.size < 5) return null
        val sweepRange = wristXs.maxOrNull()!! - wristXs.minOrNull()!!
        if (sweepRange > 2.0f * shoulderWidth) {
            return "over_parrying"
        }
        return null
    }

    private fun checkStepWidth(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val backAnkleKey = if (targetSide == TargetSide.LEFT) "left_ankle" else "right_ankle"
        for (skel in skeletons) {
            val frontAnkle = skel[limbs.ankle] ?: continue
            val backAnkle = skel[backAnkleKey] ?: continue
            val frontShoulder = skel[limbs.shoulder] ?: continue
            val pelvis = FencingGeometry.pelvisCenter(skel) ?: continue
            val sw = abs(frontShoulder.x - pelvis.x) * 2.5f
            if (sw < 10f) continue
            val stepWidth = frontAnkle.distanceTo(backAnkle)
            val ratio = stepWidth / sw
            if (ratio < 2.0f) return "narrow_step"
            if (ratio > 3.0f) return "wide_step"
        }
        return null
    }

    private fun checkCenterOfMass(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val backAnkleKey = if (targetSide == TargetSide.LEFT) "left_ankle" else "right_ankle"
        for (skel in skeletons) {
            val frontAnkle = skel[limbs.ankle] ?: continue
            val backAnkle = skel[backAnkleKey] ?: continue
            val pelvis = FencingGeometry.pelvisCenter(skel) ?: continue
            
            val frontX = frontAnkle.x
            val backX = backAnkle.x
            val pelvisX = pelvis.x
            val baseWidth = abs(frontX - backX)
            if (baseWidth < 10f) continue
            
            val ratio = if (frontX > backX) {
                (pelvisX - backX) / baseWidth
            } else {
                (backX - pelvisX) / baseWidth
            }
            if (ratio > 0.65f) return "center_of_mass_in_front"
            if (ratio < 0.35f) return "center_of_mass_leaning_backward"
        }
        return null
    }

    companion object {
        private val OffensiveActions = setOf("R", "JS", "WW", "IS")
        private val FootworkActions = setOf("SF", "SB")
    }
}
