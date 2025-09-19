import argparse
import shlex
import subprocess
from pathlib import Path

import json as _json


def run(cmd: str):
    subprocess.check_call(cmd, shell=True)


def read_first_imu_timestamp(imu_path: Path) -> float:
    with imu_path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Try JSON
            try:
                obj = _json.loads(s)
                ts = obj.get("timestamp", None)
                if ts is not None:
                    # Heuristic: treat >= 1e9 as microseconds
                    ts = float(ts)
                    if ts >= 1e9:
                        return ts / 1_000_000.0
                    return ts
            except Exception:
                pass
            # Fallback: space-separated
            parts = s.split()
            try:
                return float(parts[0])
            except Exception:
                continue
    raise RuntimeError(f"No IMU timestamp found in {imu_path}")


def scan_imu_duplicates(imu_path: Path) -> dict[int, int]:
    """
    Single-pass scan of IMU file; returns a dict of duplicate timestamps -> counts (>1).

    Supports two line formats:
    - JSON per line containing key "timestamp" (treated as microseconds if >= 1e9, else seconds)
    - Space-separated where the first token is the timestamp (seconds or microseconds)

    Timestamps are normalized to integer microseconds for robust equality checks.
    """

    def _normalize_to_microseconds(ts_value: float) -> int:
        tsf = float(ts_value)
        if tsf >= 1e9:
            return int(round(tsf))
        return int(round(tsf * 1_000_000.0))

    counts: dict[int, int] = {}
    with imu_path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Try JSON first
            try:
                obj = _json.loads(s)
                ts = obj.get("timestamp", None)
                if ts is None:
                    continue
                micro = _normalize_to_microseconds(ts)
            except Exception:
                # Fallback: first token as timestamp
                parts = s.split()
                if not parts:
                    continue
                try:
                    t = float(parts[0])
                except Exception:
                    continue
                micro = _normalize_to_microseconds(t)

            counts[micro] = counts.get(micro, 0) + 1

    return {ts: c for ts, c in counts.items() if c > 1}


def check_imu_duplicate_timestamps(imu_path: Path) -> bool:
    """Compatibility wrapper: returns True if any duplicates exist (uses single-pass scan)."""
    return bool(scan_imu_duplicates(imu_path))


def find_duplicate_imu_timestamps(imu_path: Path) -> dict[int, int]:
    """Compatibility wrapper: returns duplicates dict using single-pass scan."""
    return scan_imu_duplicates(imu_path)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_frames_by_time_window(video: Path, outdir: Path, start_s: float, end_s: float) -> list[Path]:
    ensure_dir(outdir)
    # Clean previous tmp_ files
    for p in outdir.glob("tmp_*.png"):
        p.unlink()
    # Use time-based select filter to avoid seeking issues
    ss = max(0.0, float(start_s))
    to = max(ss, float(end_s))
    vf = f"select='gte(t,{ss})*lte(t,{to})',setpts=N/FRAME_RATE/TB"
    cmd = (
        f"ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(str(video))} "
        f"-vsync 0 -vf {shlex.quote(vf)} {shlex.quote(str(outdir))}/tmp_%06d.png"
    )
    run(cmd)
    files = sorted(outdir.glob("tmp_*.png"))
    if not files:
        raise RuntimeError("No frames extracted")
    return files


def format_epoch(ts_micro: int) -> tuple[str, str]:
    s = str(ts_micro)
    if len(s) < 7:
        s = s.rjust(7, "0")
    sec = s[:-6] if len(s) > 6 else "0"
    usec = s[-6:]
    return sec, usec
                                                                                                                                          

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="/home/william/extdisk/data/motorEV/self-calib-analysis/indoor/",
        help="Base directory containing video, index, imu",
    )
    parser.add_argument(
        "--video",
        default="20250201_000004_main.h265",
        help="Video filename",
    )
    parser.add_argument(
        "--imu",
        default="20250201_000004_imu.txt",
        help="IMU source filename",
    )
    parser.add_argument(
        "--start",
        default=94.0,
        type=float,
        help="Start seconds offset from first IMU timestamp",
    )
    parser.add_argument(
        "--end",
        default=143.0,
        type=float,
        help="End seconds offset from first IMU timestamp",
    )
    args = parser.parse_args()

    base = Path(args.base)
    video = base / args.video
    imu_src = base / args.imu
    outdir = base / "color"

    ensure_dir(outdir)

    # Determine time window from IMU
    imu_first = read_first_imu_timestamp(imu_src)
    start_s = float(args.start)
    end_s = float(args.end)
    abs_start = imu_first + start_s * 1e6
    abs_end = imu_first + end_s * 1e6

    # Extract frames in the window
    frames = extract_frames_by_time_window(video, outdir, start_s, end_s)

    # Derive timestamps uniformly over [abs_start, abs_end]
    n = len(frames)
    color_lines = []
    if n == 1:
        ts_list = [abs_start]
    else:
        step = (abs_end - abs_start) / (n - 1)
        ts_list = [abs_start + i * step for i in range(n)]

    # Rename frames to sec.usec.png and build color.txt lines
    for img_path, ts in zip(frames, ts_list):
        ts_str = f"{ts:.6f}"
        new_path = outdir / f"{ts_str}.png"
        img_path.rename(new_path)
        color_lines.append(f"{ts_str} color/{ts_str}.png")

    ## Write color.txt (epoch and relative colors path)
    (base / "color.txt").write_text("\n".join(color_lines) + "\n")

    # Filter IMU within the selected ts range
    ts_lo = float(f"{ts_list[0]:.6f}")
    ts_hi = float(f"{ts_list[-1]:.6f}")

    print(ts_lo, ts_hi)

    # Build IMU output matching imu_eg.txt format: header + space-separated values
    imu_out_lines: list[str] = ["#Time Gx Gy Gz Ax Ay Ax "]
    import json as _json
    with imu_src.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = _json.loads(s)
            except Exception:
                # If not JSON, try space-split passthrough when first token is time
                parts = s.split()
                try:
                    t = float(parts[0])
                except Exception:
                    continue
                if ts_lo <= t <= ts_hi:
                    imu_out_lines.append(s)
                continue
            ts_field = obj.get("timestamp", None)
            if ts_field is None:
                continue
            tsec = float(ts_field)
            if tsec >= 1e9:
                tsec = tsec / 1_000_000.0
            if not (ts_lo <= tsec <= ts_hi):
                continue
            gx = obj.get("Gyro", {}).get("x", 0.0)
            gy = obj.get("Gyro", {}).get("y", 0.0)
            gz = obj.get("Gyro", {}).get("z", 0.0)
            ax = obj.get("Acce", {}).get("x", 0.0)
            ay = obj.get("Acce", {}).get("y", 0.0)
            az = obj.get("Acce", {}).get("z", 0.0)
            imu_out_lines.append(
                f"{tsec:.8f} {gx:.8f} {gy:.8f} {gz:.8f} {ax:.8f} {ay:.8f} {az:.8f}"
            )
    (base / "imu.txt").write_text("\n".join(imu_out_lines) + "\n")

    print(
        f"Done. Extracted {len(ts_list)} frames to {outdir}, wrote color.txt and imu.txt in {base}"
    )


if __name__ == "__main__":
    # main()
    imu_path = Path("/home/william/extdisk/data/motorEV/self-calib-analysis/indoor/ref1/imu.txt")
    dupes = scan_imu_duplicates(imu_path)
    if bool(dupes):
        for micro_ts, count in sorted(dupes.items()):
            sec = micro_ts // 1_000_000
            usec = micro_ts % 1_000_000
            print(f"{sec}.{usec:06d} -> {count} times")
    else:
        print("No duplicate timestamps.")
