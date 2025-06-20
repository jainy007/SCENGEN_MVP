#!/usr/bin/env python3
"""
lidar_debug_tool.py - Debug LiDAR Data Loading and Processing

Investigates why LiDAR BEV visualization shows empty data:
- Checks LiDAR file loading
- Analyzes point cloud statistics
- Tests coordinate transformations
- Validates classification logic

Usage: python lidar_debug_tool.py --event_id "hash/event_name"

Author: PEM | June 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from termcolor import colored, cprint

# Add workspace root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_data_loader import MultimodalDataLoader

class LiDARDebugger:
    """Debug LiDAR data loading and processing issues"""
    
    def __init__(self):
        self.data_loader = MultimodalDataLoader()
    
    def get_all_events(self):
        """Get all available events"""
        events = []
        for clip in self.data_loader.usable_clips:
            hash_id = clip["hash_id"]
            session_path = clip["session_path"]
            for usable_event in clip["usable_events"]:
                event_name = usable_event["event_name"]
                events.append({
                    "hash_id": hash_id,
                    "event_name": event_name,
                    "session_path": session_path,
                    "event_id": f"{hash_id}/{event_name}"
                })
        return events
    
    def find_event_by_id(self, event_id: str):
        """Find event by ID"""
        all_events = self.get_all_events()
        for event in all_events:
            if event_id in event["event_id"] or event["hash_id"] == event_id:
                return event
        return None
    
    def debug_lidar_file(self, lidar_file: str):
        """Debug a single LiDAR file"""
        cprint(f"\n🔍 Debugging LiDAR file: {os.path.basename(lidar_file)}", "cyan")
        
        try:
            # Load raw data
            lidar_df = pd.read_feather(lidar_file)
            cprint(f"✅ File loaded successfully", "green")
            
            # Basic statistics
            cprint(f"📊 Basic Statistics:", "blue")
            cprint(f"  - Total points: {len(lidar_df)}", "white")
            cprint(f"  - Columns: {list(lidar_df.columns)}", "white")
            
            if len(lidar_df) == 0:
                cprint(f"❌ Empty LiDAR file!", "red")
                return None
            
            # Coordinate ranges
            points = lidar_df[['x', 'y', 'z', 'intensity']].values
            cprint(f"📏 Coordinate Ranges:", "blue")
            cprint(f"  - X: {points[:, 0].min():.1f} to {points[:, 0].max():.1f} m", "white")
            cprint(f"  - Y: {points[:, 1].min():.1f} to {points[:, 1].max():.1f} m", "white")
            cprint(f"  - Z: {points[:, 2].min():.1f} to {points[:, 2].max():.1f} m", "white")
            cprint(f"  - Intensity: {points[:, 3].min():.0f} to {points[:, 3].max():.0f}", "white")
            
            # Check for reasonable automotive LiDAR ranges
            reasonable_range = 200  # meters
            reasonable_points = (
                (np.abs(points[:, 0]) < reasonable_range) & 
                (np.abs(points[:, 1]) < reasonable_range) &
                (points[:, 2] > -10) & (points[:, 2] < 50)  # Reasonable height range
            )
            
            cprint(f"🎯 Points in reasonable range: {np.sum(reasonable_points)}/{len(points)} ({np.sum(reasonable_points)/len(points)*100:.1f}%)", "yellow")
            
            return points
            
        except Exception as e:
            cprint(f"❌ Error loading LiDAR file: {e}", "red")
            return None
    
    def debug_ego_trajectory(self, event):
        """Debug ego vehicle trajectory"""
        cprint(f"\n🚗 Debugging Ego Trajectory:", "cyan")
        
        ego_traj = event.ego_trajectory
        cprint(f"📊 Trajectory Statistics:", "blue")
        cprint(f"  - Total trajectory points: {len(ego_traj)}", "white")
        cprint(f"  - Time range: {ego_traj['timestamp_ns'].min()} to {ego_traj['timestamp_ns'].max()}", "white")
        cprint(f"  - Position range X: {ego_traj['tx_m'].min():.1f} to {ego_traj['tx_m'].max():.1f} m", "white")
        cprint(f"  - Position range Y: {ego_traj['ty_m'].min():.1f} to {ego_traj['ty_m'].max():.1f} m", "white")
        cprint(f"  - Position range Z: {ego_traj['tz_m'].min():.1f} to {ego_traj['tz_m'].max():.1f} m", "white")
        
        # Sample ego positions
        sample_positions = ego_traj[['tx_m', 'ty_m', 'tz_m']].head(5)
        cprint(f"📍 Sample ego positions:", "blue")
        for i, (_, row) in enumerate(sample_positions.iterrows()):
            cprint(f"  {i+1}. ({row['tx_m']:.1f}, {row['ty_m']:.1f}, {row['tz_m']:.1f})", "white")
    
    def test_coordinate_alignment(self, event, lidar_points, ego_pos):
        """Test if LiDAR coordinates align with ego position"""
        cprint(f"\n🔄 Testing Coordinate Alignment:", "cyan")
        
        ego_x, ego_y = ego_pos
        cprint(f"🚗 Ego position: ({ego_x:.1f}, {ego_y:.1f})", "blue")
        
        # Check points relative to ego
        rel_x = lidar_points[:, 0] - ego_x
        rel_y = lidar_points[:, 1] - ego_y
        distances = np.sqrt(rel_x**2 + rel_y**2)
        
        cprint(f"📏 Distance from ego:", "blue")
        cprint(f"  - Min distance: {distances.min():.1f} m", "white")
        cprint(f"  - Max distance: {distances.max():.1f} m", "white")
        cprint(f"  - Mean distance: {distances.mean():.1f} m", "white")
        
        # Points in various ranges
        ranges = [10, 25, 50, 100, 200]
        for r in ranges:
            in_range = np.sum(distances < r)
            cprint(f"  - Within {r}m: {in_range} points ({in_range/len(distances)*100:.1f}%)", "white")
        
        # Check if most points are very far away
        if distances.min() > 100:
            cprint(f"⚠️  WARNING: All points are >100m from ego vehicle!", "yellow")
            cprint(f"   This suggests coordinate system mismatch", "yellow")
            return False
        
        return True
    
    def test_classification_logic(self, lidar_points):
        """Test point classification logic"""
        cprint(f"\n🏷️  Testing Classification Logic:", "cyan")
        
        if len(lidar_points) == 0:
            cprint(f"❌ No points to classify!", "red")
            return
        
        x, y, z, intensity = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2], lidar_points[:, 3]
        
        # Test each classification category
        categories = {
            'road_surface': (z < 0.5) & (z > -2.0) & (intensity > 10) & (intensity < 100),
            'buildings': (z > 2.0) & (z < 20.0),
            'vegetation': (z > 0.5) & (z < 5.0) & (intensity < 50),
            'traffic_signs': (z > 2.0) & (z < 8.0) & (intensity > 150),
            'other_vehicles': (z > 0.5) & (z < 3.0) & (intensity > 80) & (intensity < 200),
            'pedestrians': (z > 0.8) & (z < 2.2) & (intensity > 60) & (intensity < 150)
        }
        
        total_classified = 0
        for category, mask in categories.items():
            count = np.sum(mask)
            total_classified += count
            cprint(f"  - {category}: {count} points ({count/len(lidar_points)*100:.1f}%)", "white")
        
        unclassified = len(lidar_points) - total_classified
        cprint(f"  - unclassified: {unclassified} points ({unclassified/len(lidar_points)*100:.1f}%)", "white")
        
        if total_classified == 0:
            cprint(f"⚠️  WARNING: No points classified! Classification logic may be too strict.", "yellow")
        
        # Suggest relaxed classification
        cprint(f"\n💡 Relaxed classification (any height/intensity):", "yellow")
        relaxed_count = len(lidar_points)
        cprint(f"  - All points: {relaxed_count} ({100:.1f}%)", "white")
    
    def comprehensive_debug(self, event_id: str):
        """Run comprehensive debug on an event"""
        cprint(f"\n🔬 COMPREHENSIVE LIDAR DEBUG", "blue", attrs=["bold"])
        cprint(f"Event: {event_id}", "blue")
        cprint(f"="*60, "blue")
        
        # Find and load event
        event_info = self.find_event_by_id(event_id)
        if not event_info:
            cprint(f"❌ Event not found: {event_id}", "red")
            return
        
        cprint(f"✅ Found event: {event_info['event_id']}", "green")
        
        # Load event data
        try:
            event = self.data_loader.load_dangerous_event(
                event_info["hash_id"], 
                event_info["event_name"], 
                event_info["session_path"]
            )
        except Exception as e:
            cprint(f"❌ Failed to load event: {e}", "red")
            return
        
        cprint(f"📊 Event loaded: {len(event.lidar_files)} LiDAR files", "green")
        
        # Debug ego trajectory
        self.debug_ego_trajectory(event)
        
        # Debug first few LiDAR files
        for i in range(min(3, len(event.lidar_files))):
            lidar_file = event.lidar_files[i]
            points = self.debug_lidar_file(lidar_file)
            
            if points is not None and len(points) > 0:
                # Get corresponding ego position
                timestamp = event.lidar_timestamps[i]
                time_diffs = np.abs(event.ego_trajectory['timestamp_ns'] - timestamp)
                closest_idx = time_diffs.argmin()
                ego_row = event.ego_trajectory.iloc[closest_idx]
                ego_pos = (ego_row['tx_m'], ego_row['ty_m'])
                
                # Test coordinate alignment
                self.test_coordinate_alignment(event, points, ego_pos)
                
                # Test classification
                self.test_classification_logic(points)
                
                break  # Only need to test one file thoroughly
        
        cprint(f"\n🎯 DEBUG COMPLETE", "green", attrs=["bold"])

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Debug LiDAR data issues")
    parser.add_argument("--event_id", type=str, help="Event ID to debug")
    parser.add_argument("--list", action="store_true", help="List available events")
    args = parser.parse_args()
    
    debugger = LiDARDebugger()
    
    if args.list:
        events = debugger.get_all_events()
        cprint(f"📋 Available Events:", "blue", attrs=["bold"])
        for i, event in enumerate(events[:10], 1):
            cprint(f"  {i:2d}. {event['event_id']}", "white")
    elif args.event_id:
        debugger.comprehensive_debug(args.event_id)
    else:
        # Debug first available event
        events = debugger.get_all_events()
        if events:
            debugger.comprehensive_debug(events[0]["event_id"])

if __name__ == "__main__":
    main()