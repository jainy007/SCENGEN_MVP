#!/usr/bin/env python3
"""
debug_temporal_alignment.py - Debug Temporal Alignment Issues

Investigates the temporal misalignment between video and LiDAR data.
Reports exactly what timestamps are being used vs expected.

Usage: python debug_temporal_alignment.py

Author: PEM | June 2025
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from termcolor import cprint

# Add workspace root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_data_loader import MultimodalDataLoader

def debug_temporal_alignment():
    """Debug the temporal alignment for the problematic event"""
    
    cprint("\n🔍 TEMPORAL ALIGNMENT DEBUG REPORT", "cyan", attrs=["bold"])
    cprint("="*60, "cyan")
    
    # Test case: the problematic event
    hash_id = "04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy"
    event_name = "dangerous_event_1"
    session_path = "/mnt/db/av_dataset/04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy__Spring_2020"
    
    cprint(f"🎯 Testing Event: {hash_id}/{event_name}", "yellow")
    
    # Initialize loader
    loader = MultimodalDataLoader()
    
    # Load event metadata first
    cprint(f"\n📊 STEP 1: Loading Event Metadata", "blue", attrs=["bold"])
    
    try:
        event_metadata, motion_data = loader.load_event_metadata(hash_id, event_name)
        
        # Extract timing info
        event_info = event_metadata["event_info"]
        start_time_s = event_info["start_time_seconds"]
        duration_s = event_info["duration_seconds"]
        end_time_s = event_info["end_time_seconds"]
        start_frame = event_info["start_frame"]
        end_frame = event_info["end_frame"]
        
        cprint(f"  ✅ Event metadata loaded successfully", "green")
        cprint(f"     📅 Temporal range: {start_time_s}s - {end_time_s}s", "white")
        cprint(f"     ⏱️  Duration: {duration_s}s", "white")
        cprint(f"     🎬 Frame range: {start_frame} - {end_frame}", "white")
        cprint(f"     💬 Comment: {event_metadata['annotation_info']['comment']}", "white")
        
    except Exception as e:
        cprint(f"  ❌ Error loading metadata: {e}", "red")
        return
    
    # Load ego trajectory and analyze session timeline
    cprint(f"\n📡 STEP 2: Analyzing Session Timeline", "blue", attrs=["bold"])
    
    try:
        ego_pose_file = os.path.join(session_path, "city_SE3_egovehicle.feather")
        cprint(f"  📂 Loading ego trajectory: {ego_pose_file}", "white")
        
        if not os.path.exists(ego_pose_file):
            cprint(f"  ❌ Ego pose file not found!", "red")
            return
            
        ego_df = pd.read_feather(ego_pose_file)
        
        session_start_ns = ego_df['timestamp_ns'].iloc[0]
        session_end_ns = ego_df['timestamp_ns'].iloc[-1]
        session_duration_s = (session_end_ns - session_start_ns) / 1e9
        
        cprint(f"  ✅ Ego trajectory loaded: {len(ego_df)} poses", "green")
        cprint(f"     🕐 Session start timestamp: {session_start_ns}", "white")
        cprint(f"     🕕 Session end timestamp: {session_end_ns}", "white")
        cprint(f"     ⏱️  Session duration: {session_duration_s:.2f}s", "white")
        
        # Convert to human readable
        session_start_dt = datetime.fromtimestamp(session_start_ns / 1e9)
        session_end_dt = datetime.fromtimestamp(session_end_ns / 1e9)
        cprint(f"     📅 Session start time: {session_start_dt}", "white")
        cprint(f"     📅 Session end time: {session_end_dt}", "white")
        
    except Exception as e:
        cprint(f"  ❌ Error loading ego trajectory: {e}", "red")
        return
    
    # Calculate expected LiDAR timestamp range
    cprint(f"\n🧮 STEP 3: Calculating Expected LiDAR Timestamps", "blue", attrs=["bold"])
    
    try:
        # This is the CURRENT logic in multimodal_data_loader.py
        event_start_ns = session_start_ns + start_time_s * 1e9
        event_end_ns = session_start_ns + end_time_s * 1e9
        
        cprint(f"  📊 Current calculation logic:", "yellow")
        cprint(f"     event_start_ns = session_start_ns + start_time_s * 1e9", "white")
        cprint(f"     event_start_ns = {session_start_ns} + {start_time_s} * 1e9", "white")
        cprint(f"     event_start_ns = {event_start_ns}", "white")
        cprint(f"     event_end_ns = {event_end_ns}", "white")
        
        # Convert to human readable
        event_start_dt = datetime.fromtimestamp(event_start_ns / 1e9)
        event_end_dt = datetime.fromtimestamp(event_end_ns / 1e9)
        cprint(f"     📅 Calculated event start: {event_start_dt}", "white")
        cprint(f"     📅 Calculated event end: {event_end_dt}", "white")
        
        # Calculate offset from session start
        offset_from_session_start = (event_start_ns - session_start_ns) / 1e9
        cprint(f"     ⏰ Offset from session start: {offset_from_session_start:.2f}s", "white")
        
        # Expected vs actual
        cprint(f"  🎯 Expected vs Calculated:", "yellow")
        cprint(f"     Expected offset: {start_time_s}s", "white")
        cprint(f"     Calculated offset: {offset_from_session_start:.2f}s", "white")
        cprint(f"     Match: {'✅' if abs(offset_from_session_start - start_time_s) < 0.1 else '❌'}", 
               "green" if abs(offset_from_session_start - start_time_s) < 0.1 else "red")
        
    except Exception as e:
        cprint(f"  ❌ Error in timestamp calculation: {e}", "red")
        return
    
    # Check actual LiDAR files in the session
    cprint(f"\n📡 STEP 4: Analyzing Available LiDAR Files", "blue", attrs=["bold"])
    
    try:
        lidar_dir = os.path.join(session_path, "sensors", "lidar")
        
        if not os.path.exists(lidar_dir):
            cprint(f"  ❌ LiDAR directory not found: {lidar_dir}", "red")
            return
        
        all_lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.feather')]
        all_lidar_timestamps = [int(f.replace('.feather', '')) for f in all_lidar_files]
        all_lidar_timestamps.sort()
        
        cprint(f"  ✅ Found {len(all_lidar_files)} LiDAR files", "green")
        cprint(f"     🕐 First LiDAR timestamp: {all_lidar_timestamps[0]}", "white")
        cprint(f"     🕕 Last LiDAR timestamp: {all_lidar_timestamps[-1]}", "white")
        
        # Convert to relative times
        first_relative = (all_lidar_timestamps[0] - session_start_ns) / 1e9
        last_relative = (all_lidar_timestamps[-1] - session_start_ns) / 1e9
        
        cprint(f"     ⏰ First LiDAR relative time: {first_relative:.2f}s", "white")
        cprint(f"     ⏰ Last LiDAR relative time: {last_relative:.2f}s", "white")
        
    except Exception as e:
        cprint(f"  ❌ Error analyzing LiDAR files: {e}", "red")
        return
    
    # Filter LiDAR files using current logic
    cprint(f"\n🎯 STEP 5: Testing Current LiDAR Filtering Logic", "blue", attrs=["bold"])
    
    try:
        # This replicates the logic in load_lidar_data()
        event_lidar_files = []
        event_lidar_timestamps = []
        
        for lidar_file in all_lidar_files:
            timestamp = int(lidar_file.replace('.feather', ''))
            if event_start_ns <= timestamp <= event_end_ns:
                event_lidar_files.append(os.path.join(lidar_dir, lidar_file))
                event_lidar_timestamps.append(timestamp)
        
        # Sort by timestamp
        sorted_pairs = sorted(zip(event_lidar_timestamps, event_lidar_files))
        event_lidar_timestamps = [pair[0] for pair in sorted_pairs]
        event_lidar_files = [pair[1] for pair in sorted_pairs]
        
        cprint(f"  📊 LiDAR filtering results:", "yellow")
        cprint(f"     🎯 Target range: {event_start_ns} - {event_end_ns}", "white")
        cprint(f"     ✅ Files found: {len(event_lidar_files)}", "white")
        
        if len(event_lidar_files) > 0:
            # Show first and last matches
            first_match_relative = (event_lidar_timestamps[0] - session_start_ns) / 1e9
            last_match_relative = (event_lidar_timestamps[-1] - session_start_ns) / 1e9
            
            cprint(f"     🕐 First match timestamp: {event_lidar_timestamps[0]}", "white")
            cprint(f"     🕕 Last match timestamp: {event_lidar_timestamps[-1]}", "white")
            cprint(f"     ⏰ First match relative: {first_match_relative:.2f}s", "white")
            cprint(f"     ⏰ Last match relative: {last_match_relative:.2f}s", "white")
            cprint(f"     📏 Actual duration: {last_match_relative - first_match_relative:.2f}s", "white")
            
            # Check if this matches expected
            expected_range = f"{start_time_s:.1f}s - {end_time_s:.1f}s"
            actual_range = f"{first_match_relative:.1f}s - {last_match_relative:.1f}s"
            
            cprint(f"  🔍 Range Comparison:", "yellow")
            cprint(f"     Expected: {expected_range}", "white")
            cprint(f"     Actual: {actual_range}", "white")
            
            range_match = (abs(first_match_relative - start_time_s) < 1.0 and 
                          abs(last_match_relative - end_time_s) < 1.0)
            cprint(f"     Match: {'✅' if range_match else '❌'}", 
                   "green" if range_match else "red")
            
            if not range_match:
                cprint(f"  🚨 TEMPORAL MISALIGNMENT DETECTED!", "red", attrs=["bold"])
                difference = first_match_relative - start_time_s
                cprint(f"     Offset: {difference:.2f}s", "red")
                
                # Calculate what the timestamps should be
                correct_start_ns = session_start_ns + start_time_s * 1e9
                correct_end_ns = session_start_ns + end_time_s * 1e9
                
                cprint(f"  💡 What should happen:", "yellow")
                cprint(f"     Correct start timestamp: {correct_start_ns}", "white")
                cprint(f"     Correct end timestamp: {correct_end_ns}", "white")
        else:
            cprint(f"  ❌ No LiDAR files found in target range!", "red")
        
    except Exception as e:
        cprint(f"  ❌ Error filtering LiDAR files: {e}", "red")
        return
    
    # Cross-reference with known dangerous events timeline
    cprint(f"\n📋 STEP 6: Cross-Reference with Known Timeline", "blue", attrs=["bold"])
    
    dangerous_events_timeline = {
        "dangerous_event_1": {"frames": "451-527", "comment": "narrow near miss"},
        "dangerous_event_2": {"frames": "995-1033", "comment": "oversteer near miss"}, 
        "dangerous_event_3": {"frames": "200-235", "comment": "stop sine overshoot"}
    }
    
    cprint(f"  📊 Known dangerous events:", "yellow")
    for event, info in dangerous_events_timeline.items():
        marker = "👉" if event == event_name else "  "
        cprint(f"     {marker} {event}: frames {info['frames']} - {info['comment']}", "white")
    
    # Estimate frame rate and check frame alignment
    estimated_fps = 10.0  # Based on your data
    expected_frame_start = start_time_s * estimated_fps
    expected_frame_end = end_time_s * estimated_fps
    
    cprint(f"  🎬 Frame timeline analysis (assuming {estimated_fps} FPS):", "yellow")
    cprint(f"     Expected frames: {expected_frame_start:.0f} - {expected_frame_end:.0f}", "white")
    cprint(f"     Metadata frames: {start_frame} - {end_frame}", "white")
    
    frame_match = (abs(expected_frame_start - start_frame) < 10 and 
                   abs(expected_frame_end - end_frame) < 10)
    cprint(f"     Frame alignment: {'✅' if frame_match else '❌'}", 
           "green" if frame_match else "red")
    
    # Summary and recommendations
    cprint(f"\n📋 SUMMARY & DIAGNOSIS", "cyan", attrs=["bold"])
    cprint("="*60, "cyan")
    
    if len(event_lidar_files) > 0:
        actual_time_range = f"{first_match_relative:.1f}s - {last_match_relative:.1f}s"
        expected_time_range = f"{start_time_s:.1f}s - {end_time_s:.1f}s"
        
        if abs(first_match_relative - start_time_s) > 5.0:
            cprint(f"🚨 MAJOR TEMPORAL MISALIGNMENT DETECTED!", "red", attrs=["bold"])
            cprint(f"   Expected: {expected_time_range}", "white")
            cprint(f"   Actual: {actual_time_range}", "white")
            cprint(f"   Offset: {first_match_relative - start_time_s:.1f}s", "red")
            
            # Check if it matches other events
            offset_seconds = first_match_relative - start_time_s
            cprint(f"\n🔍 Checking if LiDAR matches other events:", "yellow")
            
            for other_event, info in dangerous_events_timeline.items():
                if other_event != event_name:
                    frame_range = info["frames"].split("-")
                    other_start_s = int(frame_range[0]) / estimated_fps
                    if abs(first_match_relative - other_start_s) < 2.0:
                        cprint(f"   🎯 LiDAR timeline matches {other_event} ({info['comment']})!", "red")
                        cprint(f"      This explains the 90° turn in LiDAR vs straight road in camera", "white")
        else:
            cprint(f"✅ Temporal alignment looks correct", "green")
            cprint(f"   Expected: {expected_time_range}", "white")
            cprint(f"   Actual: {actual_time_range}", "white")
    else:
        cprint(f"❌ No LiDAR data found - check timestamp calculation", "red")

if __name__ == "__main__":
    debug_temporal_alignment()