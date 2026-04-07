"""
AI Fencing Coach & Referee System - Main Application

This is the entry point for the fencing coaching system.
It demonstrates the complete pipeline from video input to coaching feedback.
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np

from src.app_interface import SystemPipeline, FencingCoachUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FencingCoachApplication:
    """Main application controller."""
    
    def __init__(
        self,
        use_bifencenet: bool = False,
        device: str = "auto",
        model_checkpoint: Path = None,
        pose_backend: str = "auto",
        pose_model: str = None
    ):
        """
        Initialize application.
        
        Args:
            use_bifencenet: Use BiFenceNet instead of FenceNet
            device: Device to use (auto, cuda, cpu)
            model_checkpoint: Path to pretrained model
            pose_backend: Pose estimator backend
            pose_model: Optional pose model path
        """
        # Auto-detect device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing AI Fencing Coach on device: {device}")
        
        # Initialize pipeline
        self.pipeline = SystemPipeline(
            use_bifencenet=use_bifencenet,
            device=device,
            model_checkpoint=model_checkpoint,
            pose_backend=pose_backend,
            pose_model_path=pose_model
        )
        
        # Initialize UI
        self.ui = FencingCoachUI()
        
        logger.info("Application initialized successfully")
    
    def process_video(
        self,
        video_path: str,
        fencer_id: str,
        opponent_id: str = None,
        fencer_name: str = "",
        opponent_name: str = ""
    ):
        """
        Process a fencing bout video.
        
        Args:
            video_path: Path to video file
            fencer_id: Fencer identifier string
            opponent_id: Opponent identifier (optional)
            fencer_name: Fencer full name (optional)
            opponent_name: Opponent full name (optional)
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        logger.info(f"Processing video: {video_path}")
        
        # Set fencer and opponent
        self.pipeline.set_fencer(fencer_id, fencer_name)
        if opponent_id:
            self.pipeline.set_opponent(opponent_id, opponent_name)
        
        # Process video
        results = self.pipeline.process_video(video_path, fencer_id)
        
        logger.info(f"Bout Statistics: {results['statistics']}")
        
        # Generate conclusive feedback
        feedback = self.pipeline.get_conclusive_feedback(bout_result="completed")
        logger.info(f"Feedback: {feedback}")
        
        return results
    
    def run_interactive_mode(self):
        """Run interactive UI for real-time coaching."""
        logger.info("Starting interactive mode...")
        
        print("\n" + "="*60)
        print("AI Fencing Coach & Referee System - Interactive Mode")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Process video file")
            print("2. Configure fencer")
            print("3. View fencer profile")
            print("4. Exit")
            
            choice = input("Select option (1-4): ").strip()
            
            if choice == "1":
                video_path = input("Enter video path: ").strip()
                fencer_id = input("Enter fencer ID: ").strip()
                opponent_id = input("Enter opponent ID (optional): ").strip() or None
                
                self.process_video(video_path, fencer_id, opponent_id)
                
            elif choice == "2":
                fencer_id = input("Enter fencer ID: ").strip()
                fencer_name = input("Enter fencer name: ").strip()
                self.pipeline.set_fencer(fencer_id, fencer_name)
                print(f"Fencer configured: {fencer_id}")
                
            elif choice == "3":
                fencer_id = input("Enter fencer ID: ").strip()
                profile = self.pipeline.profile_manager.load_profile(fencer_id)
                if profile:
                    print(f"\nProfile for {fencer_id}:")
                    print(f"  Name: {profile.get('name', 'N/A')}")
                    print(f"  Total Bouts: {profile['overall_stats'].get('total_bouts', 0)}")
                    print(f"  Win Rate: {profile['overall_stats'].get('wins', 0)} wins")
                else:
                    print(f"No profile found for {fencer_id}")
                
            elif choice == "4":
                print("Exiting...")
                self.ui.close()
                break
            else:
                print("Invalid option")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Fencing Coach & Referee System"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--fencer-id",
        type=str,
        default="fencer_001",
        help="Fencer identifier"
    )
    parser.add_argument(
        "--fencer-name",
        type=str,
        default="",
        help="Fencer full name"
    )
    parser.add_argument(
        "--opponent-id",
        type=str,
        help="Opponent identifier"
    )
    parser.add_argument(
        "--use-bifencenet",
        action="store_true",
        help="Use BiFenceNet instead of FenceNet"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--pose-backend",
        type=str,
        default="auto",
        choices=["auto", "ultralytics", "mock"],
        help="Pose estimator backend"
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        help="Path to YOLO pose model weights"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Initialize application
    app = FencingCoachApplication(
        use_bifencenet=args.use_bifencenet,
        device=args.device,
        model_checkpoint=Path(args.model) if args.model else None,
        pose_backend=args.pose_backend,
        pose_model=args.pose_model
    )
    
    if args.interactive:
        app.run_interactive_mode()
    elif args.video:
        app.process_video(
            video_path=args.video,
            fencer_id=args.fencer_id,
            opponent_id=args.opponent_id,
            fencer_name=args.fencer_name
        )
    else:
        # Default interactive mode
        app.run_interactive_mode()


if __name__ == "__main__":
    main()
