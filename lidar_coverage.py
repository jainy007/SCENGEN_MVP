#!/usr/bin/env python3
"""
lidar_coverage_analysis.py - Comprehensive LiDAR Data Coverage Analysis

Analyzes the mapping between dangerous clips and AV dataset LiDAR data:
- Maps hash IDs to UUID format for session lookup
- Checks LiDAR data availability for each dangerous event
- Analyzes timestamp alignment between video clips and sensor data
- Generates coverage statistics and identifies gaps

Usage: python lidar_coverage_analysis.py

Author: PEM | June 2025
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from termcolor import colored, cprint
import traceback
from datetime import datetime

# Configuration
DANGEROUS_CLIPS_DIR = "/home/jainy007/PEM/triage_brain/llm_input_dataset"
AV_DATASET_DIR = "/mnt/db/av_dataset/sensor"
OUTPUT_DIR = "analysis_output"

def setup_output_dir():
    """Create output directory for analysis results"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
def hash_to_uuid(hash_id: str) -> str:
    """Convert 32-char hash to UUID format with dashes"""
    if len(hash_id) != 32:
        return hash_id
    return f"{hash_id[:8]}-{hash_id[8:12]}-{hash_id[12:16]}-{hash_id[16:20]}-{hash_id[20:]}"

def find_av_session_path(hash_id: str) -> Optional[str]:
    """Find the corresponding AV dataset session path"""
    # First try the UUID format in sensor subdirectory
    uuid_format = hash_to_uuid(hash_id)
    for split in ['train', 'val', 'test']:
        session_path = os.path.join(AV_DATASET_DIR, split, uuid_format)
        if os.path.exists(session_path):
            return session_path
    
    # Try the new format: /mnt/db/av_dataset/HASH__SEASON_YEAR/sensors/
    base_dir = os.path.dirname(AV_DATASET_DIR)  # /mnt/db/av_dataset
    for item in os.listdir(base_dir):
        if item.startswith(hash_id + "__"):
            session_path = os.path.join(base_dir, item)
            if os.path.exists(session_path):
                return session_path
    
    return None

def analyze_lidar_availability(session_path: str) -> Dict:
    """Analyze LiDAR data availability for a session"""
    lidar_dir = os.path.join(session_path, 'sensors', 'lidar')
    ego_pose_file = os.path.join(session_path, 'city_SE3_egovehicle.feather')
    
    result = {
        'lidar_dir_exists': os.path.exists(lidar_dir),
        'ego_pose_exists': os.path.exists(ego_pose_file),
        'lidar_files': [],
        'lidar_timestamps': [],
        'ego_timeline': None,
        'session_duration': None
    }
    
    try:
        # Analyze LiDAR files
        if result['lidar_dir_exists']:
            lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.feather')]
            result['lidar_files'] = sorted(lidar_files)
            result['lidar_timestamps'] = [int(f.replace('.feather', '')) for f in lidar_files]
            result['lidar_timestamps'].sort()
        
        # Analyze ego pose timeline
        if result['ego_pose_exists']:
            ego_df = pd.read_feather(ego_pose_file)
            start_ns = ego_df['timestamp_ns'].iloc[0]
            end_ns = ego_df['timestamp_ns'].iloc[-1]
            duration_s = (end_ns - start_ns) / 1e9
            
            result['ego_timeline'] = {
                'start_ns': int(start_ns),
                'end_ns': int(end_ns), 
                'duration_seconds': float(duration_s),
                'num_poses': len(ego_df)
            }
            result['session_duration'] = float(duration_s)
            
    except Exception as e:
        result['error'] = str(e)
        
    return result

def load_dangerous_event_metadata(event_dir: str) -> Dict:
    """Load metadata for a dangerous event"""
    metadata_file = os.path.join(event_dir, 'event_metadata.json')
    motion_file = os.path.join(event_dir, 'motion_data.json')
    
    result = {
        'event_metadata': None,
        'motion_data_points': 0,
        'video_exists': False
    }
    
    try:
        # Load event metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                result['event_metadata'] = json.load(f)
        
        # Check motion data
        if os.path.exists(motion_file):
            with open(motion_file, 'r') as f:
                motion_data = json.load(f)
                result['motion_data_points'] = len(motion_data)
        
        # Check video
        video_file = os.path.join(event_dir, 'video_segment.mp4')
        result['video_exists'] = os.path.exists(video_file)
        
    except Exception as e:
        result['error'] = str(e)
        
    return result

def check_timestamp_alignment(dangerous_event: Dict, av_analysis: Dict) -> Dict:
    """Check if dangerous event timestamps align with AV sensor data"""
    alignment = {
        'within_session': False,
        'lidar_coverage': False,
        'closest_lidar_files': [],
        'time_offset_analysis': None
    }
    
    try:
        if not dangerous_event['event_metadata'] or not av_analysis['ego_timeline']:
            return alignment
            
        event_info = dangerous_event['event_metadata']['event_info']
        start_time_s = event_info['start_time_seconds']
        duration_s = event_info['duration_seconds']
        end_time_s = start_time_s + duration_s
        
        session_duration = av_analysis['session_duration']
        
        # Check if event fits within session duration
        alignment['within_session'] = end_time_s <= session_duration
        
        # If not within session, check if it's using different time reference
        if not alignment['within_session']:
            alignment['time_offset_analysis'] = {
                'event_start': start_time_s,
                'event_end': end_time_s,
                'session_duration': session_duration,
                'overflow_seconds': end_time_s - session_duration
            }
        
        # Check LiDAR coverage
        if av_analysis['lidar_timestamps'] and alignment['within_session']:
            ego_start_ns = av_analysis['ego_timeline']['start_ns']
            event_start_ns = ego_start_ns + start_time_s * 1e9
            event_end_ns = ego_start_ns + end_time_s * 1e9
            
            # Find closest LiDAR files
            closest_files = []
            for ts in av_analysis['lidar_timestamps']:
                if event_start_ns <= ts <= event_end_ns:
                    diff_s = abs(ts - event_start_ns) / 1e9
                    closest_files.append({
                        'timestamp': ts,
                        'filename': f"{ts}.feather",
                        'offset_seconds': diff_s
                    })
            
            alignment['closest_lidar_files'] = sorted(closest_files, key=lambda x: x['offset_seconds'])
            alignment['lidar_coverage'] = len(closest_files) > 0
            
    except Exception as e:
        alignment['error'] = str(e)
        
    return alignment

def run_comprehensive_analysis():
    """Run comprehensive coverage analysis on all dangerous clips"""
    cprint("\n🔍 STARTING COMPREHENSIVE LIDAR COVERAGE ANALYSIS", "blue", attrs=["bold"])
    
    setup_output_dir()
    
    # Initialize results
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_clips': 0,
        'clips_with_av_data': 0,
        'clips_with_lidar': 0,
        'total_dangerous_events': 0,
        'events_with_lidar_coverage': 0,
        'clips_analysis': {},
        'summary_stats': {},
        'coverage_gaps': []
    }
    
    # Get all dangerous clip directories
    clip_dirs = [d for d in os.listdir(DANGEROUS_CLIPS_DIR) 
                if len(d) == 32 and os.path.isdir(os.path.join(DANGEROUS_CLIPS_DIR, d))]
    
    results['total_clips'] = len(clip_dirs)
    cprint(f"📊 Found {len(clip_dirs)} dangerous clip directories", "green")
    
    # Analyze each clip
    for i, hash_id in enumerate(sorted(clip_dirs)):
        cprint(f"\n📋 Analyzing clip {i+1}/{len(clip_dirs)}: {hash_id}", "cyan")
        
        clip_analysis = {
            'hash_id': hash_id,
            'uuid_format': hash_to_uuid(hash_id),
            'av_session_path': None,
            'av_data_available': False,
            'lidar_analysis': None,
            'dangerous_events': {},
            'timestamp_alignments': {}
        }
        
        # Find corresponding AV session
        session_path = find_av_session_path(hash_id)
        if session_path:
            clip_analysis['av_session_path'] = session_path
            clip_analysis['av_data_available'] = True
            results['clips_with_av_data'] += 1
            
            # Analyze LiDAR availability
            lidar_analysis = analyze_lidar_availability(session_path)
            clip_analysis['lidar_analysis'] = lidar_analysis
            
            if lidar_analysis['lidar_dir_exists'] and lidar_analysis['lidar_files']:
                results['clips_with_lidar'] += 1
                cprint(f"  ✅ LiDAR data available: {len(lidar_analysis['lidar_files'])} files", "green")
            else:
                cprint(f"  ❌ No LiDAR data found", "red")
        else:
            cprint(f"  ❌ No AV session found for hash {hash_id}", "red")
            results['coverage_gaps'].append({
                'hash_id': hash_id,
                'issue': 'av_session_not_found'
            })
        
        # Analyze dangerous events in this clip
        clip_dir = os.path.join(DANGEROUS_CLIPS_DIR, hash_id)
        event_dirs = [d for d in os.listdir(clip_dir) if d.startswith('dangerous_event_')]
        
        for event_dir in sorted(event_dirs):
            event_path = os.path.join(clip_dir, event_dir)
            if os.path.isdir(event_path):
                results['total_dangerous_events'] += 1
                
                # Load event metadata
                event_analysis = load_dangerous_event_metadata(event_path)
                clip_analysis['dangerous_events'][event_dir] = event_analysis
                
                # Check timestamp alignment
                if clip_analysis['av_data_available'] and clip_analysis['lidar_analysis']:
                    alignment = check_timestamp_alignment(event_analysis, clip_analysis['lidar_analysis'])
                    clip_analysis['timestamp_alignments'][event_dir] = alignment
                    
                    if alignment['lidar_coverage']:
                        results['events_with_lidar_coverage'] += 1
                        cprint(f"    ✅ {event_dir}: LiDAR coverage available", "green")
                    else:
                        cprint(f"    ⚠️  {event_dir}: No LiDAR coverage", "yellow")
                        if alignment.get('time_offset_analysis'):
                            overflow = alignment['time_offset_analysis']['overflow_seconds']
                            cprint(f"       Event extends {overflow:.1f}s beyond session", "yellow")
        
        results['clips_analysis'][hash_id] = clip_analysis
    
    # Generate summary statistics
    results['summary_stats'] = {
        'av_data_coverage_percent': (results['clips_with_av_data'] / results['total_clips'] * 100) if results['total_clips'] > 0 else 0,
        'lidar_coverage_percent': (results['clips_with_lidar'] / results['total_clips'] * 100) if results['total_clips'] > 0 else 0,
        'event_lidar_coverage_percent': (results['events_with_lidar_coverage'] / results['total_dangerous_events'] * 100) if results['total_dangerous_events'] > 0 else 0
    }
    
    # Save detailed results
    output_file = os.path.join(OUTPUT_DIR, 'lidar_coverage_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print_analysis_summary(results)
    
    return results

def print_analysis_summary(results: Dict):
    """Print a comprehensive summary of the analysis"""
    cprint("\n" + "="*80, "blue")
    cprint("📊 LIDAR COVERAGE ANALYSIS SUMMARY", "blue", attrs=["bold"])
    cprint("="*80, "blue")
    
    stats = results['summary_stats']
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Total dangerous clips: {results['total_clips']}")
    print(f"  Clips with AV data: {results['clips_with_av_data']} ({stats['av_data_coverage_percent']:.1f}%)")
    print(f"  Clips with LiDAR: {results['clips_with_lidar']} ({stats['lidar_coverage_percent']:.1f}%)")
    print(f"  Total dangerous events: {results['total_dangerous_events']}")
    print(f"  Events with LiDAR coverage: {results['events_with_lidar_coverage']} ({stats['event_lidar_coverage_percent']:.1f}%)")
    
    # Coverage gaps analysis
    if results['coverage_gaps']:
        cprint(f"\n⚠️  COVERAGE GAPS ({len(results['coverage_gaps'])}):", "yellow", attrs=["bold"])
        for gap in results['coverage_gaps']:
            print(f"  - {gap['hash_id']}: {gap['issue']}")
    
    # Timestamp alignment issues
    cprint(f"\n🕐 TIMESTAMP ALIGNMENT ANALYSIS:", "cyan", attrs=["bold"])
    alignment_issues = 0
    perfect_alignments = 0
    
    for hash_id, clip in results['clips_analysis'].items():
        for event_name, alignment in clip.get('timestamp_alignments', {}).items():
            if alignment.get('within_session') and alignment.get('lidar_coverage'):
                perfect_alignments += 1
            else:
                alignment_issues += 1
                if alignment.get('time_offset_analysis'):
                    overflow = alignment['time_offset_analysis']['overflow_seconds']
                    print(f"  ⚠️  {hash_id}/{event_name}: Extends {overflow:.1f}s beyond session")
    
    print(f"  Perfect alignments: {perfect_alignments}")
    print(f"  Alignment issues: {alignment_issues}")
    
    # Recommendations
    cprint(f"\n🎯 RECOMMENDATIONS:", "green", attrs=["bold"])
    
    if stats['lidar_coverage_percent'] > 80:
        cprint(f"  ✅ Excellent LiDAR coverage! Ready for multimodal analysis.", "green")
    elif stats['lidar_coverage_percent'] > 50:
        cprint(f"  ⚠️  Good LiDAR coverage. Consider focusing on covered clips first.", "yellow")
    else:
        cprint(f"  ❌ Limited LiDAR coverage. May need alternative approach.", "red")
    
    if alignment_issues > 0:
        cprint(f"  🔧 {alignment_issues} timestamp alignment issues detected.", "yellow")
        cprint(f"     Consider implementing time reference correction.", "yellow")
    
    cprint(f"\n📁 Detailed results saved to: {OUTPUT_DIR}/lidar_coverage_analysis.json", "blue")

def generate_usable_clips_list(results: Dict):
    """Generate a list of clips ready for multimodal analysis"""
    usable_clips = []
    
    for hash_id, clip in results['clips_analysis'].items():
        if (clip['av_data_available'] and 
            clip['lidar_analysis'] and 
            clip['lidar_analysis']['lidar_dir_exists']):
            
            usable_events = []
            for event_name, alignment in clip.get('timestamp_alignments', {}).items():
                if alignment.get('lidar_coverage'):
                    usable_events.append({
                        'event_name': event_name,
                        'lidar_files_count': len(alignment.get('closest_lidar_files', [])),
                        'perfect_alignment': alignment.get('within_session', False)
                    })
            
            if usable_events:
                usable_clips.append({
                    'hash_id': hash_id,
                    'session_path': clip['av_session_path'],
                    'usable_events': usable_events,
                    'total_events': len(usable_events)
                })
    
    # Save usable clips list
    output_file = os.path.join(OUTPUT_DIR, 'usable_clips_for_multimodal.json')
    with open(output_file, 'w') as f:
        json.dump(usable_clips, f, indent=2)
    
    cprint(f"\n📋 Generated usable clips list: {len(usable_clips)} clips ready for multimodal analysis", "green")
    cprint(f"   Saved to: {output_file}", "green")
    
    return usable_clips

if __name__ == "__main__":
    try:
        # Run comprehensive analysis
        results = run_comprehensive_analysis()
        
        # Generate usable clips list for next steps
        usable_clips = generate_usable_clips_list(results)
        
        cprint(f"\n🎉 Analysis complete! Ready for multimodal LLM pipeline.", "green", attrs=["bold"])
        
    except Exception as e:
        cprint(f"\n💥 Analysis failed: {e}", "red", attrs=["bold"])
        print(f"🔍 Traceback: {traceback.format_exc()}")