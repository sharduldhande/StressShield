"""
lsl_probe.py — verify OpenSignals LSL stream is reachable and print live data.

Run this AFTER enabling the LSL plugin in OpenSignals and starting a recording.
It will:
  1. List every LSL stream visible on this machine
  2. Connect to the first one that looks like an OpenSignals / BITalino stream
  3. Print a live sample feed so you can confirm data is flowing

Usage:
    python lsl_probe.py
    python lsl_probe.py --stream "openSignals"   # match by partial name (case-insensitive)
    python lsl_probe.py --duration 30            # run for N seconds then exit (default: run forever)
    python lsl_probe.py --list-only              # only list streams, don't connect
"""

import argparse
import time
from pylsl import StreamInlet, resolve_streams, proc_clocksync


# ── helpers ──────────────────────────────────────────────────────────────────

def discover(timeout: float = 5.0):
    print(f"\nScanning for LSL streams ({timeout}s)...\n")
    streams = resolve_streams(wait_time=timeout)
    if not streams:
        print("No LSL streams found.")
        print("\nTroubleshooting:")
        print("  • Is OpenSignals running and connected to BITalino?")
        print("  • Is the LSL plugin enabled? (OpenSignals → Add-ons / Plugins → LSL)")
        print("  • Did you start an acquisition in OpenSignals?")
        return []

    print(f"Found {len(streams)} stream(s):\n")
    for i, s in enumerate(streams):
        print(f"  [{i}] name={s.name()!r:30s}  type={s.type()!r:10s}  "
              f"channels={s.channel_count()}  fs={s.nominal_srate():.0f} Hz  "
              f"source_id={s.source_id()!r}")
    return streams


def pick_stream(streams, name_filter: str | None):
    if not streams:
        return None

    if name_filter:
        needle = name_filter.lower()
        matches = [s for s in streams if needle in s.name().lower()
                                      or needle in s.type().lower()]
        if matches:
            return matches[0]
        print(f"\nNo stream matching {name_filter!r}. Using first available stream.")

    # heuristic: prefer streams whose name/type hints at BITalino / OpenSignals
    hints = ("opensignals", "bitalino", "exg", "eeg", "ecg", "biosignals")
    for h in hints:
        for s in streams:
            if h in s.name().lower() or h in s.type().lower():
                return s

    return streams[0]


def print_samples(inlet: StreamInlet, duration: float | None, n_header_lines: int = 1):
    info = inlet.info()
    n_ch = info.channel_count()
    fs   = info.nominal_srate()
    name = info.name()

    # Try to read channel labels from stream XML
    ch_names = []
    ch = info.desc().child("channels").child("channel")
    while ch.name() == "channel":
        label = ch.child_value("label") or ch.child_value("name") or f"ch{len(ch_names)}"
        ch_names.append(label)
        ch = ch.next_sibling()
    if len(ch_names) != n_ch:
        ch_names = [f"ch{i}" for i in range(n_ch)]

    print(f"\nConnected to {name!r}  |  {n_ch} channels @ {fs:.0f} Hz")
    print(f"Channel labels: {ch_names}")
    print(f"\n{'Timestamp':>14s}  " + "  ".join(f"{c:>10s}" for c in ch_names))
    print("-" * (16 + 12 * n_ch))

    start = time.time()
    sample_count = 0
    try:
        while True:
            sample, ts = inlet.pull_sample(timeout=2.0)
            if sample is None:
                print("  [timeout — no sample received]")
                continue

            sample_count += 1
            # Print every ~0.5 s worth of samples to avoid flooding the terminal
            if sample_count % max(1, int(fs * 0.5)) == 0:
                vals = "  ".join(f"{v:>10.4f}" for v in sample)
                print(f"{ts:>14.3f}  {vals}")

            if duration and (time.time() - start) >= duration:
                break

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start
    print(f"\nReceived {sample_count} samples in {elapsed:.1f}s "
          f"(effective rate: {sample_count/elapsed:.1f} Hz)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Probe LSL streams from OpenSignals/BITalino")
    parser.add_argument("--stream",    default=None,  help="Partial stream name to prefer")
    parser.add_argument("--duration",  type=float, default=None, help="Seconds to run (default: forever)")
    parser.add_argument("--timeout",   type=float, default=5.0,  help="Discovery timeout in seconds")
    parser.add_argument("--list-only", action="store_true",      help="List streams and exit")
    args = parser.parse_args()

    streams = discover(timeout=args.timeout)
    if not streams or args.list_only:
        return

    chosen = pick_stream(streams, args.stream)
    if chosen is None:
        return

    print(f"\nConnecting to {chosen.name()!r} ...")
    inlet = StreamInlet(chosen, processing_flags=proc_clocksync)
    print_samples(inlet, args.duration)


if __name__ == "__main__":
    main()
