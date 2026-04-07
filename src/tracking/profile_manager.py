"""
Profile Manager - Manages athlete profiles and historical statistics.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ProfileManager:
    """
    Manages fencer profiles and historical performance data.
    Stores athlete statistics for long-term progression tracking.
    """
    
    def __init__(self, profiles_dir: str = "data/fencer_profiles/"):
        """
        Initialize Profile Manager.
        
        Args:
            profiles_dir: Directory to store fencer profiles
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
    
    def create_profile(self, fencer_id: str, name: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Create a new fencer profile.
        
        Args:
            fencer_id: Unique identifier for the fencer
            name: Fencer's name
            metadata: Optional additional metadata
            
        Returns:
            Profile dictionary
        """
        profile = {
            "fencer_id": fencer_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "bouts": [],
            "overall_stats": {
                "total_bouts": 0,
                "wins": 0,
                "losses": 0,
                "average_offensive_ratio": 0.0,
                "average_defensive_ratio": 0.0,
                "average_js_sf_ratio": 0.0,
            },
            "metadata": metadata or {}
        }
        
        self._save_profile(profile)
        logger.info(f"Created profile for {name} ({fencer_id})")
        return profile
    
    def load_profile(self, fencer_id: str) -> Optional[Dict]:
        """
        Load fencer profile from disk.
        
        Args:
            fencer_id: Fencer's unique identifier
            
        Returns:
            Profile dictionary or None if not found
        """
        profile_path = self.profiles_dir / f"{fencer_id}.json"
        
        if not profile_path.exists():
            logger.warning(f"Profile not found for {fencer_id}")
            return None
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            return profile
        except Exception as e:
            logger.error(f"Error loading profile {fencer_id}: {e}")
            return None
    
    def save_bout(
        self,
        fencer_id: str,
        bout_data: Dict[str, Any],
        opponent_id: Optional[str] = None,
        result: Optional[str] = None
    ) -> Dict:
        """
        Save bout statistics to fencer profile.
        
        Args:
            fencer_id: Fencer's unique identifier
            bout_data: Bout statistics dictionary
            opponent_id: Optional opponent identifier
            result: Optional bout result ("win" or "loss")
            
        Returns:
            Updated profile dictionary
        """
        profile = self.load_profile(fencer_id)
        
        if profile is None:
            logger.warning(f"Profile not found for {fencer_id}, creating new one")
            profile = self.create_profile(fencer_id, fencer_id)
        
        bout_record = {
            "timestamp": datetime.now().isoformat(),
            "opponent_id": opponent_id,
            "result": result,
            "statistics": bout_data
        }
        
        profile["bouts"].append(bout_record)
        
        # Update overall statistics
        self._update_overall_stats(profile)
        profile["last_updated"] = datetime.now().isoformat()
        
        self._save_profile(profile)
        logger.info(f"Saved bout for {fencer_id}")
        
        return profile
    
    def _update_overall_stats(self, profile: Dict):
        """
        Update overall statistics based on bout history.
        
        Args:
            profile: Fencer profile dictionary
        """
        bouts = profile["bouts"]
        
        if not bouts:
            return
        
        total_bouts = len(bouts)
        wins = sum(1 for bout in bouts if bout.get("result") == "win")
        losses = total_bouts - wins
        
        # Calculate averages
        offensive_ratios = [
            bout["statistics"].get("offensive_ratio", 0.0)
            for bout in bouts
            if "statistics" in bout
        ]
        defensive_ratios = [
            bout["statistics"].get("defensive_ratio", 0.0)
            for bout in bouts
            if "statistics" in bout
        ]
        js_sf_ratios = [
            bout["statistics"].get("js_sf_ratio", 0.0)
            for bout in bouts
            if "statistics" in bout
        ]
        
        profile["overall_stats"]["total_bouts"] = total_bouts
        profile["overall_stats"]["wins"] = wins
        profile["overall_stats"]["losses"] = losses
        profile["overall_stats"]["average_offensive_ratio"] = (
            sum(offensive_ratios) / len(offensive_ratios) if offensive_ratios else 0.0
        )
        profile["overall_stats"]["average_defensive_ratio"] = (
            sum(defensive_ratios) / len(defensive_ratios) if defensive_ratios else 0.0
        )
        profile["overall_stats"]["average_js_sf_ratio"] = (
            sum(js_sf_ratios) / len(js_sf_ratios) if js_sf_ratios else 0.0
        )
    
    def get_progression_metrics(self, fencer_id: str, num_recent_bouts: Optional[int] = None) -> Dict:
        """
        Get fencer's progression metrics.
        
        Args:
            fencer_id: Fencer's unique identifier
            num_recent_bouts: Number of recent bouts to analyze (None for all)
            
        Returns:
            Dictionary with progression metrics
        """
        profile = self.load_profile(fencer_id)
        
        if profile is None:
            logger.warning(f"Profile not found for {fencer_id}")
            return {}
        
        bouts = profile["bouts"]
        if num_recent_bouts is not None:
            bouts = bouts[-num_recent_bouts:]
        
        if not bouts:
            return {}
        
        metrics = {
            "fencer_id": fencer_id,
            "num_bouts": len(bouts),
            "win_rate": 0.0,
            "action_frequency_trend": {},
            "offensive_ratio_trend": [],
            "defensive_ratio_trend": [],
        }
        
        total_bouts = len(bouts)
        wins = sum(1 for bout in bouts if bout.get("result") == "win")
        metrics["win_rate"] = wins / total_bouts if total_bouts > 0 else 0.0
        
        # Collect trends
        for bout in bouts:
            if "statistics" in bout:
                stats = bout["statistics"]
                metrics["offensive_ratio_trend"].append(stats.get("offensive_ratio", 0.0))
                metrics["defensive_ratio_trend"].append(stats.get("defensive_ratio", 0.0))
        
        return metrics
    
    def list_profiles(self) -> List[str]:
        """
        List all available fencer profiles.
        
        Returns:
            List of fencer IDs
        """
        profiles = []
        for file_path in self.profiles_dir.glob("*.json"):
            profiles.append(file_path.stem)
        return sorted(profiles)
    
    def _save_profile(self, profile: Dict):
        """
        Save profile to disk.
        
        Args:
            profile: Profile dictionary
        """
        profile_path = self.profiles_dir / f"{profile['fencer_id']}.json"
        
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving profile: {e}")
            raise
    
    def delete_profile(self, fencer_id: str) -> bool:
        """
        Delete a fencer profile.
        
        Args:
            fencer_id: Fencer's unique identifier
            
        Returns:
            True if successful, False otherwise
        """
        profile_path = self.profiles_dir / f"{fencer_id}.json"
        
        if not profile_path.exists():
            logger.warning(f"Profile not found for {fencer_id}")
            return False
        
        try:
            profile_path.unlink()
            logger.info(f"Deleted profile for {fencer_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting profile: {e}")
            return False
