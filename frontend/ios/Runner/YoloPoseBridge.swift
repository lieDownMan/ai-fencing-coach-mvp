import Foundation
import CoreML
import Vision
import Flutter

/// YOLOv8-Pose CoreML inference bridge exposed to Flutter via MethodChannel.
///
/// Channel: "fencing_coach/yolo_pose"
/// Method "load": loads the yolov8n_pose.mlpackage model
/// Method "detectPose": receives BGRA image bytes and runs the model
///   Returns: Array of dicts [{"index": Int, "x": Double, "y": Double, "confidence": Double}]
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
        // Look for compiled CoreML model in Bundle resources
        guard let compiledURL = Bundle.main.url(
            forResource: "yolov8n_pose",
            withExtension: "mlmodelc"
        ) else {
            // Try raw .mlpackage
            if let modelURL = Bundle.main.url(
                forResource: "yolov8n_pose",
                withExtension: "mlpackage"
            ) {
                do {
                    let compiledURL = try MLModel.compileModel(at: modelURL)
                    let mlModel = try MLModel(contentsOf: compiledURL)
                    self.model = try VNCoreMLModel(for: mlModel)
                    setupRequest()
                    result(true)
                } catch {
                    result(FlutterError(
                        code: "LOAD_ERROR",
                        message: "Compilation/Load failed: \(error.localizedDescription)",
                        details: nil
                    ))
                }
            } else {
                result(FlutterError(
                    code: "NOT_FOUND",
                    message: "yolov8n_pose.mlpackage not found in bundle",
                    details: nil
                ))
            }
            return
        }
        
        do {
            let mlModel = try MLModel(contentsOf: compiledURL)
            self.model = try VNCoreMLModel(for: mlModel)
            setupRequest()
            result(true)
        } catch {
            result(FlutterError(code: "LOAD_ERROR", message: error.localizedDescription, details: nil))
        }
    }
    
    private func setupRequest() {
        guard let model = model else { return }
        request = VNCoreMLRequest(model: model)
        request?.imageCropAndScaleOption = .scaleFill
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
        let providerRef = CGDataProvider(data: data as CFData)
        let colorSpaceRef = CGColorSpaceCreateDeviceRGB()
        // BGRA image is little-endian: noneSkipFirst | byteOrder32Little
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        
        guard let provider = providerRef,
              let cgImage = CGImage(
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bitsPerPixel: 32,
                  bytesPerRow: bytesPerRow,
                  space: colorSpaceRef,
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
            
            // Parse raw YOLOv8-pose output tensor [1, 56, 8400]
            let candidates = parseYoloOutput(multiArray: multiArray)
            let bestDetections = performNMS(candidates: candidates, iouThreshold: 0.45)
            
            if let best = bestDetections.first {
                // Return joints as a dictionary list matching COCO structure
                let response = formatKeypoints(best.keypoints)
                result(response)
            } else {
                result([])
            }
        } catch {
            result(FlutterError(code: "INFERENCE_ERROR", message: error.localizedDescription, details: nil))
        }
    }
    
    // Structs for parsing
    struct YoloKeypoint {
        let x: Float
        let y: Float
        let confidence: Float
    }
    
    struct YoloPoseCandidate {
        let bbox: CGRect
        let score: Float
        let keypoints: [YoloKeypoint]
    }
    
    private func parseYoloOutput(multiArray: MLMultiArray) -> [YoloPoseCandidate] {
        let pointer = UnsafeMutablePointer<Float>(OpaquePointer(multiArray.dataPointer))
        let numCols = 8400 // anchors
        let numRows = 56   // 4 (bbox) + 1 (conf) + 17 * 3 (keypoints)
        let confThreshold: Float = 0.45
        
        var candidates: [YoloPoseCandidate] = []
        
        for col in 0..<numCols {
            let score = pointer[4 * numCols + col]
            if score > confThreshold {
                let cx = pointer[0 * numCols + col]
                let cy = pointer[1 * numCols + col]
                let w = pointer[2 * numCols + col]
                let h = pointer[3 * numCols + col]
                
                let xMin = cx - w / 2.0
                let yMin = cy - h / 2.0
                
                var keypoints: [YoloKeypoint] = []
                for k in 0..<17 {
                    let kx = pointer[(5 + k * 3) * numCols + col]
                    let ky = pointer[(6 + k * 3) * numCols + col]
                    let kconf = pointer[(7 + k * 3) * numCols + col]
                    keypoints.append(YoloKeypoint(x: kx, y: ky, confidence: kconf))
                }
                
                candidates.append(YoloPoseCandidate(
                    bbox: CGRect(x: CGFloat(xMin), y: CGFloat(yMin), width: CGFloat(w), height: CGFloat(h)),
                    score: score,
                    keypoints: keypoints
                ))
            }
        }
        return candidates
    }
    
    private func performNMS(candidates: [YoloPoseCandidate], iouThreshold: Float) -> [YoloPoseCandidate] {
        let sorted = candidates.sorted { $0.score > $1.score }
        var keep: [YoloPoseCandidate] = []
        
        for cand in sorted {
            var overlap = false
            for kept in keep {
                if iou(cand.bbox, kept.bbox) > iouThreshold {
                    overlap = true
                    break
                }
            }
            if !overlap {
                keep.append(cand)
            }
        }
        return keep
    }
    
    private func iou(_ r1: CGRect, _ r2: CGRect) -> Float {
        let intersection = r1.intersection(r2)
        if intersection.isNull { return 0 }
        let areaI = intersection.width * intersection.height
        let areaU = r1.width * r1.height + r2.width * r2.height - areaI
        return Float(areaI / areaU)
    }
    
    private func formatKeypoints(_ keypoints: [YoloKeypoint]) -> [[String: Any]] {
        // Return COCO index mapping format:
        // [{"index": Int, "x": Double, "y": Double, "confidence": Double}]
        // This makes it easy for Dart side to parse.
        var list: [[String: Any]] = []
        for (i, kp) in keypoints.enumerated() {
            list.append([
                "index": i,
                "x": Double(kp.x / 640.0), // Normalize to [0.0, 1.0] relative to 640x640 input
                "y": Double(kp.y / 640.0),
                "confidence": Double(kp.confidence)
            ])
        }
        return list
    }
}
