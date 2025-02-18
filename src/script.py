#!/usr/bin/env python3

import sys
from pathlib import Path
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_suffix
from ultralytics.utils.downloads import attempt_download_asset

def pull(name):
    check_suffix(name)
    print(f"downloaded {attempt_download_asset(Path('/files/oltralytics')/name)}")

def main(command, *params):
    commands = {
        "pull": pull,
    }
    
    if command in commands:
        commands[command](*params)
    else:
        print(f"{command} is an invalid command.")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], *sys.argv[2:])
    else:
        print("Please provide a command (pull)")