#!/usr/bin/env python3
"""Monitor Branch G Stage 3B training logs for pre-registered hard pathologies."""

import argparse
import json
import math
import os
import re
import signal
import time
from pathlib import Path


LOG_RE = re.compile(r"Epoch \[(?P<epoch>\d+)/(?P<epochs>\d+)\]\[(?P<iter>\d+)/(?P<iters>\d+)\]")
TOTAL_LOSS_RE = re.compile(r"\] - Epoch \[[^\]]+\]\[[^\]]+\] loss: (?P<value>[-+0-9.eEinfnaINFNA]+)")
LOSS_VALUE_RE = re.compile(r"(?:^|, )(?P<name>[A-Za-z0-9_.]*loss[A-Za-z0-9_.]*): (?P<value>[-+0-9.eEinfnaINFNA]+)")
DIVERGENCE_MIN_DELTA = 10.0


def parse_float(raw):
    try:
        return float(raw)
    except ValueError:
        return math.nan


def find_train_log(run_root):
    candidates = sorted(Path(run_root).glob("*/*/train.log"))
    if not candidates:
        return None
    return candidates[-1]


def read_events(log_path):
    events = []
    if log_path is None or not log_path.exists():
        return events
    for line in log_path.read_text(errors="replace").splitlines():
        match = LOG_RE.search(line)
        total = TOTAL_LOSS_RE.search(line)
        if not match or not total:
            continue
        losses = {m.group("name"): parse_float(m.group("value")) for m in LOSS_VALUE_RE.finditer(line)}
        losses["loss"] = parse_float(total.group("value"))
        events.append(
            {
                "epoch": int(match.group("epoch")),
                "iter": int(match.group("iter")),
                "iters": int(match.group("iters")),
                "loss": losses["loss"],
                "losses": losses,
                "line": line,
            }
        )
    return events


def append_jsonl(path, payload):
    if not path:
        return
    with Path(path).open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def terminate_process_group(pid, reason, log_copy):
    payload = {"time": time.time(), "event": "terminate", "pid": pid, "reason": reason}
    append_jsonl(log_copy, payload)
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(30)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--log-copy")
    args = parser.parse_args()

    seen = 0
    nonfinite_logged_iters = 0
    append_jsonl(args.log_copy, {"time": time.time(), "event": "monitor_start", "pid": args.pid})
    while process_alive(args.pid):
        log_path = find_train_log(args.run_root)
        events = read_events(log_path)
        if len(events) > seen:
            for event in events[seen:]:
                values = event["losses"].values()
                if any(not math.isfinite(v) for v in values):
                    nonfinite_logged_iters += 50
                else:
                    nonfinite_logged_iters = 0
                append_jsonl(
                    args.log_copy,
                    {
                        "time": time.time(),
                        "event": "loss",
                        "epoch": event["epoch"],
                        "iter": event["iter"],
                        "loss": event["loss"],
                        "loss_dualview_distill": event["losses"].get("loss_dualview_distill"),
                    },
                )
            seen = len(events)

        if nonfinite_logged_iters >= 50:
            terminate_process_group(args.pid, "nonfinite_loss_ge_50_iters", args.log_copy)
            return

        if len(events) >= 5:
            window = [event["loss"] for event in events[-5:]]
            is_strictly_increasing = all(a < b for a, b in zip(window, window[1:]))
            is_material_rise = (window[-1] - window[0]) >= DIVERGENCE_MIN_DELTA
            if all(math.isfinite(v) for v in window) and is_strictly_increasing and is_material_rise:
                terminate_process_group(args.pid, "strictly_increasing_loss_ge_200_iters", args.log_copy)
                return

        if log_path is not None:
            text_tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-200:]).lower()
            if "out of memory" in text_tail or "cuda error: out of memory" in text_tail:
                terminate_process_group(args.pid, "oom_detected", args.log_copy)
                return

        time.sleep(args.poll_seconds)
    append_jsonl(args.log_copy, {"time": time.time(), "event": "monitor_stop", "pid": args.pid})


if __name__ == "__main__":
    main()
