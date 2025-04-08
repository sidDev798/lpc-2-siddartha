#!/usr/bin/env python3
"""
Log Viewer - A utility script for viewing and filtering backend logs.

Usage:
  python logviewer.py [options]

Options:
  --file=FILENAME     Log file to view (default: backend_debug.log)
  --level=LEVEL       Minimum log level to display (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  --filter=PATTERN    Only show lines containing this pattern
  --follow            Continuously follow the log file (like tail -f)
  --last=N            Show only the last N lines (default: all)
  --api               Show API logs (uses api_requests.log file)
"""

import os
import re
import sys
import time
import argparse
from datetime import datetime
import signal

# ANSI color codes for colorful output
COLORS = {
    'RESET': '\033[0m',
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'CRITICAL': '\033[35m', # Magenta
    'HIGHLIGHT': '\033[1m'  # Bold
}

def colorize_log_line(line):
    """Add color to log line based on log level."""
    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        if f" - {level} - " in line:
            return line.replace(f" - {level} - ", f" - {COLORS[level]}{level}{COLORS['RESET']} - ")
    return line

def highlight_pattern(line, pattern):
    """Highlight the search pattern in the log line."""
    if not pattern:
        return line
    
    highlighted = re.sub(
        f'({re.escape(pattern)})', 
        f'{COLORS["HIGHLIGHT"]}\\1{COLORS["RESET"]}', 
        line, 
        flags=re.IGNORECASE
    )
    return highlighted

def get_log_level_value(level):
    """Convert log level string to numeric value for comparison."""
    levels = {
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50
    }
    return levels.get(level.upper(), 0)

def extract_log_level(line):
    """Extract log level from a log line."""
    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        if f" - {level} - " in line:
            return level
    return None

def print_log_line(line, min_level, pattern):
    """Print a log line if it meets the filter criteria."""
    line_level = extract_log_level(line)
    
    # Skip lines that don't have a proper log level
    if not line_level:
        return
    
    # Skip lines below minimum level
    if get_log_level_value(line_level) < get_log_level_value(min_level):
        return
    
    # Skip lines that don't match pattern (if pattern is specified)
    if pattern and pattern.lower() not in line.lower():
        return
    
    # Colorize and highlight the line
    colored_line = colorize_log_line(line)
    if pattern:
        colored_line = highlight_pattern(colored_line, pattern)
    
    print(colored_line, end='')

def view_logs(filename, min_level, pattern, follow, last_n):
    """View and filter log files."""
    if not os.path.exists(filename):
        print(f"Error: Log file '{filename}' not found.")
        return
    
    try:
        # If we only want to see the last N lines
        if last_n > 0 and not follow:
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines[-last_n:]:
                    print_log_line(line, min_level, pattern)
            return
            
        # First read existing content
        with open(filename, 'r') as f:
            # Skip to end if follow mode and no last_n specified
            if follow and last_n <= 0:
                f.seek(0, os.SEEK_END)
            # Otherwise, print existing content (or last N lines)
            else:
                if last_n > 0:
                    lines = f.readlines()
                    start_idx = max(0, len(lines) - last_n)
                    for line in lines[start_idx:]:
                        print_log_line(line, min_level, pattern)
                else:
                    for line in f:
                        print_log_line(line, min_level, pattern)
            
            # If follow mode, continue reading new content
            if follow:
                print(f"Following log file {filename}... (Press Ctrl+C to exit)")
                while True:
                    line = f.readline()
                    if line:
                        print_log_line(line, min_level, pattern)
                    else:
                        time.sleep(0.1)  # Sleep briefly to avoid cpu spinning
                        
    except KeyboardInterrupt:
        print("\nLog viewing stopped.")
    except Exception as e:
        print(f"Error reading log file: {str(e)}")

def view_api_logs(filename, pattern, follow, last_n):
    """View API request/response logs."""
    if not os.path.exists(filename):
        print(f"Error: API log file '{filename}' not found.")
        return
    
    try:
        # If we only want to see the last N lines
        if last_n > 0 and not follow:
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines[-last_n:]:
                    if pattern and pattern.lower() not in line.lower():
                        continue
                    
                    # Color code requests and responses
                    if "REQUEST" in line:
                        print(f"{COLORS['INFO']}{line.rstrip()}{COLORS['RESET']}")
                    elif "RESPONSE" in line:
                        print(f"{COLORS['WARNING']}{line.rstrip()}{COLORS['RESET']}")
                    else:
                        print(line.rstrip())
            return
            
        # First read existing content
        with open(filename, 'r') as f:
            # Skip to end if follow mode and no last_n specified
            if follow and last_n <= 0:
                f.seek(0, os.SEEK_END)
            # Otherwise, print existing content (or last N lines)
            else:
                if last_n > 0:
                    lines = f.readlines()
                    start_idx = max(0, len(lines) - last_n)
                    for line in lines[start_idx:]:
                        if pattern and pattern.lower() not in line.lower():
                            continue
                        
                        # Color code requests and responses
                        if "REQUEST" in line:
                            print(f"{COLORS['INFO']}{line.rstrip()}{COLORS['RESET']}")
                        elif "RESPONSE" in line:
                            print(f"{COLORS['WARNING']}{line.rstrip()}{COLORS['RESET']}")
                        else:
                            print(line.rstrip())
                else:
                    for line in f:
                        if pattern and pattern.lower() not in line.lower():
                            continue
                        
                        # Color code requests and responses
                        if "REQUEST" in line:
                            print(f"{COLORS['INFO']}{line.rstrip()}{COLORS['RESET']}")
                        elif "RESPONSE" in line:
                            print(f"{COLORS['WARNING']}{line.rstrip()}{COLORS['RESET']}")
                        else:
                            print(line.rstrip())
            
            # If follow mode, continue reading new content
            if follow:
                print(f"Following API log file {filename}... (Press Ctrl+C to exit)")
                while True:
                    line = f.readline()
                    if line:
                        if pattern and pattern.lower() not in line.lower():
                            continue
                        
                        # Color code requests and responses
                        if "REQUEST" in line:
                            print(f"{COLORS['INFO']}{line.rstrip()}{COLORS['RESET']}")
                        elif "RESPONSE" in line:
                            print(f"{COLORS['WARNING']}{line.rstrip()}{COLORS['RESET']}")
                        else:
                            print(line.rstrip())
                    else:
                        time.sleep(0.1)  # Sleep briefly to avoid cpu spinning
                        
    except KeyboardInterrupt:
        print("\nLog viewing stopped.")
    except Exception as e:
        print(f"Error reading log file: {str(e)}")

def main():
    """Main function to parse arguments and start log viewing."""
    parser = argparse.ArgumentParser(description='View and filter log files for debugging.')
    parser.add_argument('--file', default='backend_debug.log', help='Log file to view')
    parser.add_argument('--level', default='DEBUG', help='Minimum log level to display')
    parser.add_argument('--filter', help='Only show lines containing this pattern')
    parser.add_argument('--follow', action='store_true', help='Continuously follow the log file')
    parser.add_argument('--last', type=int, default=0, help='Show only the last N lines')
    parser.add_argument('--api', action='store_true', help='Show API logs instead of backend logs')
    
    args = parser.parse_args()
    
    if args.api:
        view_api_logs(
            filename='api_requests.log', 
            pattern=args.filter, 
            follow=args.follow, 
            last_n=args.last
        )
    else:
        view_logs(
            filename=args.file, 
            min_level=args.level, 
            pattern=args.filter, 
            follow=args.follow, 
            last_n=args.last
        )

if __name__ == '__main__':
    main() 