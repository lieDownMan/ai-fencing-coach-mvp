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
        val offensive = action in OffensiveActions
        val footwork = action in FootworkActions

        if (trainingMode == TrainingMode.FOOTWORK && footwork) {
            checkBounce(skeletons)?.let(triggered::add)
            checkStanceTooHigh(skeletons)?.let(triggered::add)
            checkStepWidth(skeletons)?.let(triggered::add)
            checkCenterOfMass(skeletons)?.let(triggered::add)
        }

        if (trainingMode == TrainingMode.TARGET_PRACTICE && offensive) {
            checkLunge(skeletons)?.let(triggered::add)
            checkFootBeforeHand(skeletons)?.let(triggered::add)
            checkIncompleteArmExtension(skeletons)?.let(triggered::add)
        }

        checkGuard(skeletons)?.let(triggered::add)

        if (trainingMode != TrainingMode.FOOTWORK && footwork) {
            checkStanceTooHigh(skeletons)?.let(triggered::add)
            checkBounce(skeletons)?.let(triggered::add)
            checkStepWidth(skeletons)?.let(triggered::add)
            checkCenterOfMass(skeletons)?.let(triggered::add)
        }

        if (action == "SB" || (trainingMode == TrainingMode.FREE_BOUTING && footwork)) {
            checkOverParrying(skeletons)?.let(triggered::add)
        }

        return triggered.distinct()
    }

    private fun checkBounce(skeletons: List<Skeleton>): String? {
        val pelvisYs = skeletons.mapNotNull { FencingGeometry.pelvisCenter(it)?.y }
        if (pelvisYs.size < 5) return null
        val height = FencingGeometry.bboxHeight(skeletons)
        if (height < 1e-6f) return null
        val delta = pelvisYs.maxOrNull()!! - pelvisYs.minOrNull()!!
        return if (delta > 0.25f * height) "bounce_excessive" else null
    }

    private fun checkLunge(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val reference = skeletons.first()[limbs.ankle] ?: return null
        val peak = skeletons.maxByOrNull { it[limbs.ankle]?.distanceTo(reference) ?: 0f } ?: return null
        val hip = peak[limbs.hip] ?: return null
        val knee = peak[limbs.knee] ?: return null
        val ankle = peak[limbs.ankle] ?: return null
        return if (FencingGeometry.angle(hip, knee, ankle) < 90f) "lunge_overextension" else null
    }

    private fun checkGuard(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val threshold = if (trainingMode == TrainingMode.FREE_BOUTING) 20 else 10
        var consecutive = 0
        for (skeleton in skeletons) {
            val wrist = skeleton[limbs.wrist]
            val pelvis = FencingGeometry.pelvisCenter(skeleton)
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
        val referenceWrist = skeletons.first()[limbs.wrist] ?: return null
        val referenceAnkle = skeletons.first()[limbs.ankle] ?: return null
        var maxWrist = 0f
        var wristPeak = 0
        var maxAnkle = 0f
        var anklePeak = 0
        skeletons.forEachIndexed { index, skeleton ->
            skeleton[limbs.wrist]?.let {
                val displacement = abs(it.x - referenceWrist.x)
                if (displacement > maxWrist) {
                    maxWrist = displacement
                    wristPeak = index
                }
            }
            skeleton[limbs.ankle]?.let {
                val displacement = abs(it.x - referenceAnkle.x)
                if (displacement > maxAnkle) {
                    maxAnkle = displacement
                    anklePeak = index
                }
            }
        }
        return if (maxAnkle > 5f && maxWrist > 5f && anklePeak < wristPeak) "foot_before_hand" else null
    }

    private fun checkStanceTooHigh(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val angles = skeletons.mapNotNull {
            val hip = it[limbs.hip] ?: return@mapNotNull null
            val knee = it[limbs.knee] ?: return@mapNotNull null
            val ankle = it[limbs.ankle] ?: return@mapNotNull null
            FencingGeometry.angle(hip, knee, ankle)
        }
        if (angles.size < 3) return null
        return if (angles.average() > 170.0) "stance_too_high" else null
    }

    private fun checkIncompleteArmExtension(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val reference = skeletons.first()[limbs.wrist] ?: return null
        val peak = skeletons.maxByOrNull { abs((it[limbs.wrist]?.x ?: reference.x) - reference.x) } ?: return null
        val shoulder = peak[limbs.shoulder] ?: return null
        val elbow = peak[limbs.elbow] ?: return null
        val wrist = peak[limbs.wrist] ?: return null
        return if (FencingGeometry.angle(shoulder, elbow, wrist) < 155f) "incomplete_arm_extension" else null
    }

    private fun checkOverParrying(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val shoulderWidth = skeletons.firstNotNullOfOrNull { skeleton ->
            val shoulder = skeleton[limbs.shoulder] ?: return@firstNotNullOfOrNull null
            val other = skeleton[if (targetSide == TargetSide.RIGHT) "left_shoulder" else "right_shoulder"]
            if (other != null) {
                abs(shoulder.x - other.x)
            } else {
                val pelvis = FencingGeometry.pelvisCenter(skeleton)
                if (pelvis != null) abs(shoulder.x - pelvis.x) * 2f else null
            }
        } ?: return null
        if (shoulderWidth < 1e-6f) return null
        val wristXs = skeletons.mapNotNull { it[limbs.wrist]?.x }
        if (wristXs.size < 5) return null
        val sweep = wristXs.maxOrNull()!! - wristXs.minOrNull()!!
        return if (sweep > 2f * shoulderWidth) "over_parrying" else null
    }

    private fun checkStepWidth(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val backAnkleName = if (targetSide == TargetSide.LEFT) "left_ankle" else "right_ankle"
        for (skeleton in skeletons) {
            val frontAnkle = skeleton[limbs.ankle] ?: continue
            val backAnkle = skeleton[backAnkleName] ?: continue
            val frontShoulder = skeleton[limbs.shoulder] ?: continue
            val pelvis = FencingGeometry.pelvisCenter(skeleton) ?: continue
            val shoulderWidth = abs(frontShoulder.x - pelvis.x) * 2.5f
            if (shoulderWidth < 10f) continue
            val ratio = abs(frontAnkle.x - backAnkle.x) / shoulderWidth
            if (ratio > 3f) return "wide_step"
            if (ratio < 1f) return "narrow_step"
        }
        return null
    }

    private fun checkCenterOfMass(skeletons: List<Skeleton>): String? {
        val limbs = FencingGeometry.frontLimbs.getValue(targetSide)
        val backAnkleName = if (targetSide == TargetSide.LEFT) "left_ankle" else "right_ankle"
        for (skeleton in skeletons) {
            val frontAnkle = skeleton[limbs.ankle] ?: continue
            val backAnkle = skeleton[backAnkleName] ?: continue
            val pelvis = FencingGeometry.pelvisCenter(skeleton) ?: continue
            val baseWidth = abs(frontAnkle.x - backAnkle.x)
            if (baseWidth < 10f) continue
            val ratio = if (frontAnkle.x > backAnkle.x) {
                (pelvis.x - backAnkle.x) / baseWidth
            } else {
                (backAnkle.x - pelvis.x) / baseWidth
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
