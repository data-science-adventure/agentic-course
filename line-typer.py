
#!/usr/bin/env python3

import argparse
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()

    lines = args.file.read_text().splitlines()

    print(f"Loaded {len(lines)} lines.")
    print("Focus your target editor.")
    input("Press ENTER to start...")

    for index, line in enumerate(lines, start=1):
        print(f"[{index}/{len(lines)}] {line}")

        input("Press ENTER to type this line...")

        subprocess.run(
            ["ydotool", "type", "--", line],
            check=True
        )

        subprocess.run(
            ["ydotool", "key", "28:1", "28:0"],
            check=True
        )

        time.sleep(args.delay)

    print("Finished!")


if __name__ == "__main__":
    main()
