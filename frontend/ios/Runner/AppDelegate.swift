import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    // Register FenceNet CoreML bridge
    FenceNetBridge.register(with: engineBridge.pluginRegistry.registrar(forPlugin: "FenceNetBridge")!)
    // Register YOLOv8-Pose CoreML bridge
    YoloPoseBridge.register(with: engineBridge.pluginRegistry.registrar(forPlugin: "YoloPoseBridge")!)
  }
}
