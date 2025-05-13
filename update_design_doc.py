#!/usr/bin/env python3
"""
Utility script to check implementation status and update the working design document.
Run this in the background to periodically update the document based on implementation progress.
"""

import os
import time
import json
import hashlib
from datetime import datetime

# Configuration
DESIGN_DOC_PATH = "working_design_document.md"
CHECK_INTERVAL = 600  # 10 minutes in seconds

# Files/directories to track
TRACKED_PATHS = [
    "utils/persona",
    "persona_configs",
    "data_sources",
    "app.py",
    "insight_state.py",
    "download_data.py"
]

# Status tracking
file_hashes = {}
component_status = {
    "Project Structure": "Not Started",
    "Persona Configs": "Not Started",
    "Data Downloads": "Not Started",
    "Base Persona Classes": "Not Started",
    "LangGraph Nodes": "Not Started",
    "Chainlit Integration": "Not Started",
    "Testing": "Not Started"
}

def get_file_hash(filepath):
    """Get MD5 hash of a file"""
    if not os.path.exists(filepath):
        return ""
    
    if os.path.isdir(filepath):
        # For directories, hash the directory listing
        dir_contents = sorted(os.listdir(filepath))
        return hashlib.md5(str(dir_contents).encode()).hexdigest()
    else:
        # For files, hash the content
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

def update_component_status():
    """Check implementation files and update component status"""
    # Check Project Structure
    if (os.path.exists("utils/persona") and 
        os.path.exists("persona_configs") and 
        os.path.exists("data_sources")):
        component_status["Project Structure"] = "Completed"
    elif os.path.exists("utils") or os.path.exists("persona_configs"):
        component_status["Project Structure"] = "In Progress"
    
    # Check Persona Configs
    if os.path.exists("persona_configs"):
        config_files = [f for f in os.listdir("persona_configs") if f.endswith(".json")]
        if len(config_files) >= 9:  # At least our 9 target personas/personalities
            component_status["Persona Configs"] = "Completed"
        elif len(config_files) > 0:
            component_status["Persona Configs"] = "In Progress"
    
    # Check Data Downloads
    if os.path.exists("data_sources"):
        persona_dirs = [d for d in os.listdir("data_sources") 
                       if os.path.isdir(os.path.join("data_sources", d))]
        if len(persona_dirs) >= 9:  # All persona types and personalities
            component_status["Data Downloads"] = "Completed"
        elif len(persona_dirs) > 0:
            component_status["Data Downloads"] = "In Progress"
    
    # Check Base Persona Classes
    if os.path.exists("utils/persona/base.py"):
        component_status["Base Persona Classes"] = "Completed"
    elif os.path.exists("utils/persona"):
        component_status["Base Persona Classes"] = "In Progress"
    
    # Check LangGraph Nodes
    if os.path.exists("app.py"):
        with open("app.py", "r") as f:
            content = f.read()
            if "insight_flow_graph" in content and "InsightFlowState" in content:
                component_status["LangGraph Nodes"] = "Completed"
            elif "InsightFlowState" in content:
                component_status["LangGraph Nodes"] = "In Progress"
    
    # Check Chainlit Integration
    if os.path.exists("app.py"):
        with open("app.py", "r") as f:
            content = f.read()
            if "@cl.on_action" in content and "select_personas" in content:
                component_status["Chainlit Integration"] = "Completed"
            elif "@cl.on_chat_start" in content:
                component_status["Chainlit Integration"] = "In Progress"
    
    # Check Testing
    test_paths = ["tests", "test_app.py", "test_personas.py"]
    if any(os.path.exists(path) for path in test_paths):
        component_status["Testing"] = "In Progress"
        # Full completion would require checking test coverage

def update_design_document():
    """Update the working design document with current status"""
    if not os.path.exists(DESIGN_DOC_PATH):
        print(f"Warning: {DESIGN_DOC_PATH} not found")
        return
    
    try:
        with open(DESIGN_DOC_PATH, "r") as f:
            content = f.read()
        
        # Update the implementation progress table
        status_table = "\n## Implementation Progress Tracking\n\n"
        status_table += "| Component | Status | Notes |\n"
        status_table += "|-----------|--------|-------|\n"
        
        for component, status in component_status.items():
            status_table += f"| {component} | {status} | |\n"
        
        # Add timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_table += f"\n*Last Updated: {now}*"
        
        # Replace the existing table or add if not exists
        if "## Implementation Progress Tracking" in content:
            parts = content.split("## Implementation Progress Tracking")
            before = parts[0]
            after = parts[1].split("*Last Updated:")[0].split("\n\n", 1)[1] if "\n\n" in parts[1] else ""
            new_content = before + status_table
        else:
            new_content = content + "\n" + status_table
        
        with open(DESIGN_DOC_PATH, "w") as f:
            f.write(new_content)
        
        print(f"Updated {DESIGN_DOC_PATH} at {now}")
    
    except Exception as e:
        print(f"Error updating design document: {e}")

def check_for_changes():
    """Check if any tracked files have changed"""
    changes_detected = False
    
    for path in TRACKED_PATHS:
        current_hash = get_file_hash(path)
        if path in file_hashes and file_hashes[path] != current_hash:
            changes_detected = True
            print(f"Changes detected in {path}")
        file_hashes[path] = current_hash
    
    return changes_detected

def main():
    """Main loop to periodically check for changes and update the document"""
    print(f"Starting document update monitor. Checking every {CHECK_INTERVAL/60} minutes.")
    print(f"Press Ctrl+C to stop.")
    
    # Initial check
    for path in TRACKED_PATHS:
        file_hashes[path] = get_file_hash(path)
    
    update_component_status()
    update_design_document()
    
    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            if check_for_changes():
                update_component_status()
                update_design_document()
    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    main() 