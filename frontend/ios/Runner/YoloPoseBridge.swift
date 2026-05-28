import CoreGraphics
import CoreML
import Flutter
import Foundation
import Vision

/// YOLOv8-Pose CoreML inference bridge exposed to Flutter via MethodChannel.
///
/// Channel: "fencing_coach/yolo_pose"
/// Method "load": loads the yolov8n_pose CoreML model.
/// Method "detectPose": receives BGRA image bytes and returns YOLO detections:
/// [
///   {
///     "bbox": [x1, y1, x2, y2],      // normalized model-space box
///     "confidence": Double,
///     "sourceRank": Int,
///     "keypoints": [
///       {"index": Int, "x": Double, "y": Double, "confidence": Double}
///     ]
///   }
/// ]
@objc class YoloPoseBridge: NSObject, FlutterPlugin {

    static let channelName = "fencing_coach/yolo_pose"
    private var model: VNCoreMLModel?
    private var request: VNCoreMLRequest?

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: channelName,
            binaryMessenger: registrar.messenger()
        )
        let instance = YoloPoseBridge()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "load":
            loadModel(result: result)
        case "detectPose":
            guard let args = call.arguments as? [String: Any] else {
                result(FlutterError(code: "INVALID_ARGS", message: "Expected dictionary", details: nil))
                return
            }
            detectPose(args: args, result: result)
        case "isLoaded":
            result(model != nil)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func loadModel(result: @escaping FlutterResult) {
        if let compiledURL = Bundle.main.url(forResource: "yolov8n_pose", withExtension: "mlmodelc") {
            do {
                let mlModel = try MLModel(contentsOf: compiledURL)
                self.model = try VNCoreMLModel(for: mlModel)
                setupRequest()
                result(true)
            } catch {
                result(FlutterError(code: "LOAD_ERROR", message: error.localizedDescription, details: nil))
            }
            return
        }

        guard let modelURL = Bundle.main.url(forResource: "yolov8n_pose", withExtension: "mlpackage") else {
            result(FlutterError(
                code: "NOT_FOUND",
                message: "yolov8n_pose.mlpackage/mlmodelc not found in bundle",
                details: nil
            ))
            return
        }

        do {
            let compiledURL = try MLModel.compileModel(at: modelURL)
            let mlModel = try MLModel(contentsOf: compiledURL)
            self.model = try VNCoreMLModel(for: mlModel)
            setupRequest()
            result(true)
        } catch {
            result(FlutterError(
                code: "LOAD_ERROR",
                message: "Compilation/load failed: \(error.localizedDescription)",
                details: nil
            ))
        }
    }

    private func setupRequest() {
        guard let model = model else { return }
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill
        self.request = request
    }

    private func detectPose(args: [String: Any], result: @escaping FlutterResult) {
        guard let request = request else {
            result(FlutterError(code: "NOT_LOADED", message: "Model not loaded", details: nil))
            return
        }

        guard let bytesData = args["bytes"] as? FlutterStandardTypedData,
              let width = args["width"] as? Int,
              let height = args["height"] as? Int,
              let bytesPerRow = args["bytesPerRow"] as? Int else {
            result(FlutterError(code: "INVALID_ARGS", message: "Missing frame arguments", details: nil))
            return
        }

        let data = bytesData.data
        guard let provider = CGDataProvider(data: data as CFData) else {
            result(FlutterError(code: "CONVERSION_ERROR", message: "Failed to create data provider", details: nil))
            return
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(
            rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )

        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            result(FlutterError(code: "CONVERSION_ERROR", message: "Failed to create CGImage from bytes", details: nil))
            return
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])

            guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
                  let multiArray = observations.first?.featureValue.multiArrayValue else {
                result([])
                return
            }

            let candidates = parseYoloOutput(multiArray: multiArray)
            let detections = performNMS(candidates: candidates, iouThreshold: 0.45)
            result(formatDetections(detections))
        } catch {
            result(FlutterError(code: "INFERENCE_ERROR", message: error.localizedDescription, details: nil))
        }
    }

    private struct YoloKeypoint {
        let x: Float
        let y: Float
        let confidence: Float
    }

    private struct YoloPoseCandidate {
        let bbox: CGRect
        let score: Float
        let keypoints: [YoloKeypoint]
    }

    private struct OutputLayout {
        let channelsFirst: Bool
        let hasBatch: Bool
        let channels: Int
        let anchors: Int
    }

    private func parseYoloOutput(multiArray: MLMultiArray) -> [YoloPoseCandidate] {
        guard let layout = outputLayout(for: multiArray), layout.channels >= 56 else {
            return []
        }

        let confThreshold: Float = 0.35
        var candidates: [YoloPoseCandidate] = []

        for col in 0..<layout.anchors {
            let score = value(multiArray, row: 4, col: col, layout: layout)
            if score < confThreshold { continue }

            let cx = value(multiArray, row: 0, col: col, layout: layout)
            let cy = value(multiArray, row: 1, col: col, layout: layout)
            let w = value(multiArray, row: 2, col: col, layout: layout)
            let h = value(multiArray, row: 3, col: col, layout: layout)

            let xMin = cx - w / 2.0
            let yMin = cy - h / 2.0

            var keypoints: [YoloKeypoint] = []
            for k in 0..<17 {
                let offset = 5 + k * 3
                keypoints.append(YoloKeypoint(
                    x: value(multiArray, row: offset, col: col, layout: layout),
                    y: value(multiArray, row: offset + 1, col: col, layout: layout),
                    confidence: value(multiArray, row: offset + 2, col: col, layout: layout)
                ))
            }

            candidates.append(YoloPoseCandidate(
                bbox: CGRect(
                    x: CGFloat(xMin),
                    y: CGFloat(yMin),
                    width: CGFloat(w),
                    height: CGFloat(h)
                ),
                score: score,
                keypoints: keypoints
            ))
        }

        return candidates
    }

    private func outputLayout(for multiArray: MLMultiArray) -> OutputLayout? {
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count == 2 || shape.count == 3 else { return nil }

        let hasBatch = shape.count == 3
        let rows = hasBatch ? shape[1] : shape[0]
        let cols = hasBatch ? shape[2] : shape[1]

        if rows >= 56 && cols > rows {
            return OutputLayout(channelsFirst: true, hasBatch: hasBatch, channels: rows, anchors: cols)
        }
        if cols >= 56 && rows > cols {
            return OutputLayout(channelsFirst: false, hasBatch: hasBatch, channels: cols, anchors: rows)
        }
        return nil
    }

    private func value(
        _ multiArray: MLMultiArray,
        row: Int,
        col: Int,
        layout: OutputLayout
    ) -> Float {
        let indexes: [NSNumber]
        if layout.hasBatch {
            indexes = layout.channelsFirst
                ? [NSNumber(value: 0), NSNumber(value: row), NSNumber(value: col)]
                : [NSNumber(value: 0), NSNumber(value: col), NSNumber(value: row)]
        } else {
            indexes = layout.channelsFirst
                ? [NSNumber(value: row), NSNumber(value: col)]
                : [NSNumber(value: col), NSNumber(value: row)]
        }
        return multiArray[indexes].floatValue
    }

    private func performNMS(
        candidates: [YoloPoseCandidate],
        iouThreshold: Float
    ) -> [YoloPoseCandidate] {
        let sorted = candidates.sorted { $0.score > $1.score }
        var keep: [YoloPoseCandidate] = []

        for candidate in sorted {
            if keep.contains(where: { iou(candidate.bbox, $0.bbox) > iouThreshold }) {
                continue
            }
            keep.append(candidate)
            if keep.count >= 8 { break }
        }

        return keep
    }

    private func iou(_ r1: CGRect, _ r2: CGRect) -> Float {
        let intersection = r1.intersection(r2)
        if intersection.isNull { return 0 }
        let areaI = intersection.width * intersection.height
        let areaU = r1.width * r1.height + r2.width * r2.height - areaI
        if areaU <= 0 { return 0 }
        return Float(areaI / areaU)
    }

    private func formatDetections(_ detections: [YoloPoseCandidate]) -> [[String: Any]] {
        detections.enumerated().map { rank, detection -> [String: Any] in
            [
                "bbox": [
                    normalizeCoord(Float(detection.bbox.minX)),
                    normalizeCoord(Float(detection.bbox.minY)),
                    normalizeCoord(Float(detection.bbox.maxX)),
                    normalizeCoord(Float(detection.bbox.maxY))
                ],
                "confidence": Double(detection.score),
                "sourceRank": rank,
                "keypoints": formatKeypoints(detection.keypoints)
            ]
        }
    }

    private func formatKeypoints(_ keypoints: [YoloKeypoint]) -> [[String: Any]] {
        keypoints.enumerated().map { index, keypoint -> [String: Any] in
            [
                "index": index,
                "x": normalizeCoord(keypoint.x),
                "y": normalizeCoord(keypoint.y),
                "confidence": Double(keypoint.confidence)
            ]
        }
    }

    private func normalizeCoord(_ value: Float) -> Double {
        let normalized = value > 2.0 ? Double(value / 640.0) : Double(value)
        return min(max(normalized, 0.0), 1.0)
    }
}
