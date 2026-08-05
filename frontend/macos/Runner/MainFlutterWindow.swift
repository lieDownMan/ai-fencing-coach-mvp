import AVFoundation
import Cocoa
import CoreML
import FlutterMacOS
import Vision

class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    let windowFrame = self.frame
    self.contentViewController = flutterViewController
    self.setFrame(windowFrame, display: true)

    RegisterGeneratedPlugins(registry: flutterViewController)

    // Threshold-tuner bridge: offline video → per-frame YOLO pose detections.
    YoloVideoPoseBridge.register(messenger: flutterViewController.engine.binaryMessenger)

    super.awakeFromNib()
  }
}

// ---------------------------------------------------------------------------
// YoloVideoPoseBridge — macOS port of ios/Runner/YoloPoseBridge.swift for the
// video threshold-tuner (lib/tuner_main.dart).
//
// Lives in this file because the Runner pbxproj is old-style (adding a new
// Swift file would require project-file surgery).
//
// Channel: "fencing_coach/yolo_video_pose"
// Method "analyzeVideo" {videoPath, modelPath}:
//   Decodes the whole video with AVAssetReader and runs YOLOv8-pose CoreML on
//   every frame. Returns:
//   [
//     {"tMs": Int, "detections": [
//        {"bbox": [x1,y1,x2,y2], "confidence": Double, "sourceRank": Int,
//         "keypoints": [{"index": Int, "x": Double, "y": Double,
//                        "confidence": Double}]}
//     ]}
//   ]
//   Coordinates are normalized [0,1] in the UPRIGHT (display-oriented) image,
//   exactly like the live iOS bridge, so the Dart YoloPoseService parsing and
//   the heuristics engine consume them unchanged.
//   Progress is pushed back on the same channel via "progress" {done, total}.
//
// The YOLO output parsing / NMS / coordinate normalization below is a
// line-for-line port of the iOS bridge — keep them in sync.
// ---------------------------------------------------------------------------

class YoloVideoPoseBridge: NSObject {

  static let channelName = "fencing_coach/yolo_video_pose"

  private let channel: FlutterMethodChannel
  private var model: VNCoreMLModel?
  private var loadedModelPath: String?

  private init(channel: FlutterMethodChannel) {
    self.channel = channel
  }

  static func register(messenger: FlutterBinaryMessenger) {
    let channel = FlutterMethodChannel(name: channelName, binaryMessenger: messenger)
    let instance = YoloVideoPoseBridge(channel: channel)
    channel.setMethodCallHandler(instance.handle)
  }

  func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "analyzeVideo":
      guard let args = call.arguments as? [String: Any],
            let videoPath = args["videoPath"] as? String,
            let modelPath = args["modelPath"] as? String else {
        result(FlutterError(code: "INVALID_ARGS", message: "Expected videoPath + modelPath", details: nil))
        return
      }
      DispatchQueue.global(qos: .userInitiated).async {
        self.analyzeVideo(videoPath: videoPath, modelPath: modelPath, result: result)
      }
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  // ── Model loading ─────────────────────────────────────────────────────────

  private func ensureModel(modelPath: String) throws {
    if model != nil && loadedModelPath == modelPath { return }
    let url = URL(fileURLWithPath: modelPath)
    let compiledURL: URL
    if modelPath.hasSuffix(".mlmodelc") {
      compiledURL = url
    } else {
      compiledURL = try MLModel.compileModel(at: url)
    }
    let mlModel = try MLModel(contentsOf: compiledURL)
    model = try VNCoreMLModel(for: mlModel)
    loadedModelPath = modelPath
  }

  // ── Video analysis ────────────────────────────────────────────────────────

  private func analyzeVideo(videoPath: String, modelPath: String, result: @escaping FlutterResult) {
    func fail(_ code: String, _ message: String) {
      DispatchQueue.main.async {
        result(FlutterError(code: code, message: message, details: nil))
      }
    }

    do {
      try ensureModel(modelPath: modelPath)
    } catch {
      fail("LOAD_ERROR", "Model load failed: \(error.localizedDescription)")
      return
    }
    guard let model = model else {
      fail("LOAD_ERROR", "Model unavailable after load")
      return
    }

    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFill

    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    guard let track = asset.tracks(withMediaType: .video).first else {
      fail("NO_VIDEO_TRACK", "No video track in \(videoPath)")
      return
    }

    let orientation = Self.orientation(from: track.preferredTransform)
    let totalFrames = max(1, Int(Double(track.nominalFrameRate) * asset.duration.seconds))

    let reader: AVAssetReader
    do {
      reader = try AVAssetReader(asset: asset)
    } catch {
      fail("READER_ERROR", error.localizedDescription)
      return
    }
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    )
    output.alwaysCopiesSampleData = false
    reader.add(output)
    guard reader.startReading() else {
      fail("READER_ERROR", reader.error?.localizedDescription ?? "startReading failed")
      return
    }

    var frames: [[String: Any]] = []
    var done = 0

    while let sampleBuffer = output.copyNextSampleBuffer() {
      guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }
      let tMs = Int(CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds * 1000.0)

      let handler = VNImageRequestHandler(
        cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
      var detections: [[String: Any]] = []
      do {
        try handler.perform([request])
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
           let multiArray = observations.first?.featureValue.multiArrayValue {
          let candidates = parseYoloOutput(multiArray: multiArray)
          detections = formatDetections(performNMS(candidates: candidates, iouThreshold: 0.45))
        }
      } catch {
        // Skip frames whose inference fails; the tracker copes with gaps.
      }

      frames.append(["tMs": tMs, "detections": detections])
      done += 1
      if done % 15 == 0 {
        let progressDone = done
        DispatchQueue.main.async {
          self.channel.invokeMethod(
            "progress", arguments: ["done": progressDone, "total": max(totalFrames, progressDone)])
        }
      }
    }

    if reader.status == .failed {
      fail("READER_ERROR", reader.error?.localizedDescription ?? "reader failed")
      return
    }

    DispatchQueue.main.async {
      result(frames)
    }
  }

  /// Map the track's preferredTransform rotation to the Vision orientation
  /// that makes the frame upright (iPhone portrait videos are stored as
  /// rotated landscape buffers).
  private static func orientation(from t: CGAffineTransform) -> CGImagePropertyOrientation {
    let angle = atan2(Double(t.b), Double(t.a)) * 180.0 / Double.pi
    switch Int(angle.rounded()) {
    case 90: return .right
    case -90, 270: return .left
    case 180, -180: return .down
    default: return .up
    }
  }

  // ── YOLO output parsing (ported verbatim from the iOS bridge) ────────────

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
