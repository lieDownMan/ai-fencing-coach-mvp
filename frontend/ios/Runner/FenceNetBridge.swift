import Foundation
import CoreML
import Flutter

/// FenceNet CoreML inference bridge exposed to Flutter via MethodChannel.
///
/// Channel: "fencing_coach/fencenet"
/// Method "classify": arg = [Float] of length 18*28 = 504
///   Returns: {"action": String, "confidence": Double, "probabilities": [Double]}
@objc class FenceNetBridge: NSObject, FlutterPlugin {
    
    static let channelName = "fencing_coach/fencenet"
    private var model: MLModel?
    private let classNames = ["R", "IS", "WW", "JS", "SF", "SB"]
    private let confidenceThreshold: Float = 0.6
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: channelName,
            binaryMessenger: registrar.messenger()
        )
        let instance = FenceNetBridge()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "load":
            loadModel(result: result)
        case "classify":
            guard let args = call.arguments as? [String: Any],
                  let inputArray = args["input"] as? [Double] else {
                result(FlutterError(code: "INVALID_ARGS", message: "Expected input array", details: nil))
                return
            }
            classify(input: inputArray.map { Float($0) }, result: result)
        case "isLoaded":
            result(model != nil)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func loadModel(result: @escaping FlutterResult) {
        guard let modelURL = Bundle.main.url(
            forResource: "fencenet_v2",
            withExtension: "mlpackage"
        ) else {
            // Try .mlmodelc (compiled)
            if let compiledURL = Bundle.main.url(
                forResource: "fencenet_v2",
                withExtension: "mlmodelc"
            ) {
                do {
                    model = try MLModel(contentsOf: compiledURL)
                    result(true)
                } catch {
                    result(FlutterError(code: "LOAD_ERROR", message: error.localizedDescription, details: nil))
                }
            } else {
                result(FlutterError(code: "NOT_FOUND", message: "fencenet_v2.mlpackage not found in bundle", details: nil))
            }
            return
        }
        
        do {
            let compiledURL = try MLModel.compileModel(at: modelURL)
            model = try MLModel(contentsOf: compiledURL)
            result(true)
        } catch {
            result(FlutterError(code: "LOAD_ERROR", message: error.localizedDescription, details: nil))
        }
    }
    
    private func classify(input: [Float], result: @escaping FlutterResult) {
        guard let model = model else {
            result(FlutterError(code: "NOT_LOADED", message: "Model not loaded", details: nil))
            return
        }
        
        guard input.count == 18 * 28 else {
            result(FlutterError(
                code: "INVALID_INPUT",
                message: "Expected \(18*28) floats, got \(input.count)",
                details: nil
            ))
            return
        }
        
        do {
            // Create MLMultiArray with shape [1, 18, 28]
            let multiArray = try MLMultiArray(shape: [1, 18, 28], dataType: .float32)
            for (i, v) in input.enumerated() {
                multiArray[i] = NSNumber(value: v)
            }
            
            let inputFeatures = try MLDictionaryFeatureProvider(
                dictionary: ["input": multiArray]
            )
            
            let prediction = try model.prediction(from: inputFeatures)
            
            // Extract logits output
            guard let logitsArray = prediction.featureValue(for: "logits")?.multiArrayValue else {
                result(FlutterError(code: "OUTPUT_ERROR", message: "No logits output", details: nil))
                return
            }
            
            var logits = [Float](repeating: 0, count: 6)
            for i in 0..<6 {
                logits[i] = logitsArray[i].floatValue
            }
            
            // Softmax
            let maxL = logits.max() ?? 0
            var exps = logits.map { exp($0 - maxL) }
            let sumE = exps.reduce(0, +)
            var probs = exps.map { $0 / sumE }
            
            let maxProb = probs.max() ?? 0
            let maxIdx = probs.firstIndex(of: maxProb) ?? 0
            
            let action = maxProb >= confidenceThreshold ? classNames[maxIdx] : "Idle"
            
            result([
                "action": action,
                "confidence": Double(maxProb),
                "probabilities": probs.map { Double($0) }
            ])
            
        } catch {
            result(FlutterError(code: "INFERENCE_ERROR", message: error.localizedDescription, details: nil))
        }
    }
}
