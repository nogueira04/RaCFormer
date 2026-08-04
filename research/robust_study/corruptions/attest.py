"""Runtime attestation for the radar corruption families (b), (c) and (d2).

NEW FILE, stdlib only (no torch, no mmdet) so the runner can aggregate in the main process.

WHY A SIDECAR AND NOT A COUNTER
-------------------------------
The (a) family's attestation (research/robust_study/corruptions/cam_subset_removal.py) can hold
counters in memory because its intervention runs on the model, in the main process. The radar
families inject inside the DATASET pipeline, and the frozen driver builds its loader with
``workers_per_gpu=cfg.data.workers_per_gpu`` (research/night_gen_phase1/eval_by_condition.py:266,
config value 4 at configs/racformer_r50_nuimg_704x256_f8.py:251). The pipeline therefore runs in
4 subprocesses and an in-process counter in the parent would observe exactly zero corruption
applications — indistinguishable from the corruption never running, which is the hole this module
exists to close.

Each worker accumulates its own counters and flushes a JSON snapshot to
``$ROBUST_ATTEST_DIR/worker_<pid>_<uid>.json``; the runner unions the snapshots afterwards. The
snapshot is rewritten in place (not appended) so file size stays bounded regardless of run length.

FORK SAFETY
-----------
The sink object is created at import time in the parent and inherited by every worker at fork,
counters and all. Every record therefore checks ``os.getpid()`` first and resets on a pid change
(``_ensure_process``), so a worker starts from zero and cannot re-report anything the parent
happened to accumulate before the fork.

WHY APPEND-ONLY JSONL AND NOT A PERIODIC SNAPSHOT
-------------------------------------------------
The first version of this module rewrote a whole-state snapshot every N applications plus an
``atexit`` hook. It lost data, and the probe caught it: with 2 workers over 3 samples, a worker
that handled 2 samples flushed only its first 8 of 16 applications and the rest died with the
process — DataLoader workers are torn down by signal often enough that ``atexit`` cannot be
relied on. The attestation would then have reported partial coverage for a run that was actually
complete, which is the same failure mode in the opposite direction.

So each record is one appended JSON line, written and flushed as it happens. Whatever reached the
file stays in the file, no matter how the worker dies. Lines are deltas, so the aggregate is just
a sum over every line of every worker file. A full (b)/(c) cell writes 6,019 x 8 lines of roughly
150 bytes -- about 7 MB spread over 4 workers, which is not worth optimising away.

Evidence that is not itself an application (the (d2) calibrated_sensor lookups) accumulates in
``_pending`` and rides on the next application line, so a lookup-heavy family does not write a
line per lookup. Both (d2) loaders record their application in a ``finally`` AFTER the
perturbation context closes, so the lookups a sample caused are on that sample's own line and
nothing is left pending at the end of a worker's queue.

DISABLED BY DEFAULT
-------------------
With ``ROBUST_ATTEST_DIR`` unset the sink is a no-op, so the corruption classes stay usable
outside a runner. The runner always sets it AND fails the cell if no evidence file appears —
"the class was constructed but never invoked" surfaces as an empty directory, not as silence.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import uuid

ATTEST_DIR_ENV = "ROBUST_ATTEST_DIR"
ATTEST_PROBE_ENV = "ROBUST_ATTEST_PROBE"

RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)

# Sigma multiplier for the statistical bands below. 5 sigma puts the false-failure rate of each
# band near 6e-7 per cell, so a band firing is evidence about the mechanism, not about luck.
Z = 5.0


class EvidenceSink(object):
    """Per-process accumulator. One instance per interpreter; safe across fork."""

    def __init__(self):
        self._dir = os.environ.get(ATTEST_DIR_ENV) or None
        probe = os.environ.get(ATTEST_PROBE_ENV, "")
        self._probe_tokens = set(t for t in probe.split(",") if t)
        self._pid = None
        self._reset()
        atexit.register(self._at_exit)

    # -- lifecycle ----------------------------------------------------------------------------
    def _reset(self):
        self._pid = os.getpid()
        self._path = None
        self._handle = None
        self._identity_written = False
        self.identity = {}
        self.applications = 0
        self._pending = {"sums": {}, "tags": {}, "sets": {}, "probe": None}

    def _ensure_process(self):
        if os.getpid() != self._pid:
            self._reset()

    @property
    def enabled(self):
        return self._dir is not None

    def is_probe(self, sample_token):
        return str(sample_token) in self._probe_tokens

    def set_probe_tokens(self, tokens):
        """Set the probe set after import.

        The sink reads ``ROBUST_ATTEST_PROBE`` once, at construction, but a runner cannot know
        which sample tokens to probe until it has built the dataset -- which imports this module.
        Setting it here in the parent still reaches every worker, because workers are forked
        after this call and ``_reset`` (the fork guard) deliberately does not clear the probe set.
        """
        self._probe_tokens = set(str(t) for t in tokens)

    # -- recording ----------------------------------------------------------------------------
    def set_identity(self, **fields):
        if not self.enabled:
            return
        self._ensure_process()
        for key, value in fields.items():
            self.identity[key] = value

    def add(self, sums=None, tags=None, sets=None):
        """Evidence not tied to one corruption application (e.g. a table lookup).

        Held in ``_pending`` and written on the next application line, so a family that makes
        hundreds of lookups per sample does not write hundreds of lines.
        """
        if not self.enabled:
            return
        self._ensure_process()
        self._merge(self._pending, sums, tags, sets)

    def application(self, sample_token, sums=None, tags=None, sets=None, probe=None):
        """One corruption application. This is the unit the coverage check counts.

        Writes immediately. Nothing is buffered across applications, so however the process
        dies, every application that happened is already on disk.
        """
        if not self.enabled:
            return
        self._ensure_process()
        self._merge(self._pending, sums, tags, sets)
        token = str(sample_token)
        if probe is not None and self.is_probe(token):
            self._pending["probe"] = dict(probe, sample_token=token)
        self.applications += 1
        self._write_line(token)

    def _merge(self, into, sums, tags, sets):
        if sums:
            for key, value in sums.items():
                into["sums"][key] = into["sums"].get(key, 0.0) + float(value)
        if tags:
            for key, value in tags.items():
                name = "%s=%s" % (key, value)
                into["tags"][name] = into["tags"].get(name, 0) + 1
        if sets:
            for key, value in sets.items():
                into["sets"].setdefault(key, set()).add(str(value))

    # -- output -------------------------------------------------------------------------------
    def _open(self):
        if self._handle is None:
            os.makedirs(self._dir, exist_ok=True)
            self._path = os.path.join(
                self._dir, "worker_%d_%s.jsonl" % (self._pid, uuid.uuid4().hex[:8]))
            self._handle = open(self._path, "a")
        return self._handle

    def _write_line(self, token):
        pending = self._pending
        line = {}
        if token is not None:
            line["t"] = token
        if pending["sums"]:
            line["s"] = pending["sums"]
        if pending["tags"]:
            line["g"] = pending["tags"]
        if pending["sets"]:
            line["e"] = {k: sorted(v) for k, v in pending["sets"].items()}
        if pending["probe"] is not None:
            line["p"] = pending["probe"]
        if not self._identity_written and self.identity:
            line["i"] = dict(self.identity)
            self._identity_written = True
        if not line:
            return
        handle = self._open()
        handle.write(json.dumps(line, sort_keys=True) + "\n")
        handle.flush()
        self._pending = {"sums": {}, "tags": {}, "sets": {}, "probe": None}

    def flush(self):
        """Write any evidence still pending (lookups after the final application)."""
        if not self.enabled:
            return
        self._ensure_process()
        self._write_line(None)
        if self._handle is not None:
            self._handle.flush()

    def _at_exit(self):
        try:
            self.flush()
        except Exception:  # never let attestation break a run at interpreter shutdown
            pass


SINK = EvidenceSink()


def aggregate(attest_dir):
    """Union every worker snapshot in ``attest_dir``."""
    merged = {
        "files": [],
        "identity": {},
        "applications": 0,
        "per_sample": {},
        "sums": {},
        "tags": {},
        "sets": {},
        "probes": [],
        "identity_conflicts": [],
    }
    if not os.path.isdir(attest_dir):
        return merged
    for name in sorted(os.listdir(attest_dir)):
        if not (name.startswith("worker_") and name.endswith(".jsonl")):
            continue
        merged["files"].append(name)
        with open(os.path.join(attest_dir, name)) as handle:
            for lineno, raw in enumerate(handle, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = json.loads(raw)
                except ValueError:
                    # A worker killed mid-write leaves at most one truncated trailing line.
                    # Record it rather than dropping it: an incomplete file is evidence.
                    merged["identity_conflicts"].append(
                        "%s: unparsable line %d (worker died mid-write?)" % (name, lineno))
                    continue
                token = line.get("t")
                if token is not None:
                    merged["applications"] += 1
                    merged["per_sample"][token] = merged["per_sample"].get(token, 0) + 1
                for key, value in line.get("i", {}).items():
                    if key in merged["identity"] and merged["identity"][key] != value:
                        merged["identity_conflicts"].append(
                            "%s: %r vs %r" % (key, merged["identity"][key], value))
                    merged["identity"][key] = value
                for key, value in line.get("s", {}).items():
                    merged["sums"][key] = merged["sums"].get(key, 0.0) + float(value)
                for key, value in line.get("g", {}).items():
                    merged["tags"][key] = merged["tags"].get(key, 0) + int(value)
                for key, values in line.get("e", {}).items():
                    merged["sets"].setdefault(key, set()).update(values)
                if line.get("p") is not None:
                    merged["probes"].append(line["p"])
    merged["sets"] = {k: sorted(v) for k, v in merged["sets"].items()}
    return merged


class RadarAttestation(object):
    """Same report shape as gate_b_removal.Attestation, for a pipeline-side corruption.

    The runner constructs this in the main process from the aggregated worker snapshots, so the
    verdict is computed exactly once, from evidence written by the processes that did the work.
    """

    SCHEMA = "radar_corruption_attestation/1"
    FAMILY = None

    def __init__(self, cell, params, expect_samples, expect_apps_per_sample):
        self.cell = cell
        self.params = dict(params)
        self.expect_samples = int(expect_samples)
        self.expect_apps_per_sample = int(expect_apps_per_sample)
        self.merged = None

    def load(self, attest_dir):
        self.merged = aggregate(attest_dir)
        return self.merged

    # -- shared checks ------------------------------------------------------------------------
    def _base_failures(self):
        merged = self.merged
        failures = []
        expect_apps = self.expect_samples * self.expect_apps_per_sample

        if not merged["files"]:
            failures.append(
                "no attestation evidence was written: the corruption class was constructed "
                "but never invoked, or the workers never reached a corruption call")
            return failures, expect_apps

        if merged["applications"] == 0:
            failures.append(
                "0 corruption applications recorded across %d evidence file(s); a cell that "
                "loads samples must apply the corruption" % len(merged["files"]))

        distinct = len(merged["per_sample"])
        if distinct != self.expect_samples:
            failures.append(
                "corruption applied to %d distinct samples, expected %d"
                % (distinct, self.expect_samples))
        if merged["applications"] != expect_apps:
            failures.append(
                "total corruption applications %d != expected %d (= %d samples x %d per sample)"
                % (merged["applications"], expect_apps, self.expect_samples,
                   self.expect_apps_per_sample))

        wrong = [(t, c) for t, c in sorted(merged["per_sample"].items())
                 if c != self.expect_apps_per_sample]
        for token, count in wrong[:5]:
            failures.append(
                "sample %s had %d corruption applications, expected %d"
                % (token, count, self.expect_apps_per_sample))
        if len(wrong) > 5:
            failures.append("... and %d more samples with the wrong application count"
                            % (len(wrong) - 5))

        failures.extend("identity conflict across workers: %s" % c
                        for c in merged["identity_conflicts"])
        for key, value in self.params.items():
            recorded = merged["identity"].get(key)
            if recorded is not None and recorded != value:
                failures.append(
                    "workers recorded %s=%r but the cell config says %r" % (key, recorded, value))
        return failures, expect_apps

    def _family_report(self):
        raise NotImplementedError

    def report(self):
        failures, expect_apps = self._base_failures()
        observed, family_failures = self._family_report()
        failures.extend(family_failures)
        return {
            "schema": self.SCHEMA,
            "family": self.FAMILY,
            "cell": self.cell,
            "params": self.params,
            "expected": {
                "n_samples": self.expect_samples,
                "applications_per_sample": self.expect_apps_per_sample,
                "applications": expect_apps,
            },
            "observed": dict(
                observed,
                evidence_files=self.merged["files"],
                distinct_samples=len(self.merged["per_sample"]),
                applications=self.merged["applications"],
                worker_identity=self.merged["identity"],
            ),
            "probe_set": self.merged["probes"],
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        }


class DropoutAttestation(RadarAttestation):
    """(b): did the realised drop fraction match nominal p, over the whole run?"""

    FAMILY = "radar_dropout"

    def _family_report(self):
        sums = self.merged["sums"]
        tags = self.merged["tags"]
        failures = []

        n_before = sums.get("points_before", 0.0)
        n_after = sums.get("points_after", 0.0)
        realised = (1.0 - n_after / n_before) if n_before else float("nan")
        nominal = float(self.params.get("drop_p", float("nan")))

        # Band: each point is an independent Bernoulli(p) drop, so the realised fraction has
        # SE = sqrt(p(1-p)/N). Z sigma, floored at 0.002 so a huge N cannot make the band
        # narrower than the arithmetic is worth.
        band = float("nan")
        if n_before and nominal == nominal:
            se = math.sqrt(max(nominal * (1.0 - nominal), 0.0) / n_before)
            band = max(Z * se, 0.002)
            if abs(realised - nominal) > band:
                failures.append(
                    "realised drop fraction %.6f differs from nominal p=%.3f by more than the "
                    "%.6f band (N=%d points, SE=%.6f)"
                    % (realised, nominal, band, int(n_before), se))

        if tags.get("times_mismatch=1"):
            failures.append(
                "%d call(s) returned points and times with different column counts"
                % tags["times_mismatch=1"])
        if tags.get("grew=1"):
            failures.append("%d call(s) returned MORE points than they were given"
                            % tags["grew=1"])

        return {
            "points_before": int(n_before),
            "points_after": int(n_after),
            "realised_drop_fraction": realised,
            "nominal_drop_p": nominal,
            "band": band,
            "empty_cloud_calls": int(tags.get("empty=1", 0)),
            "tags": tags,
        }, failures


class NoiseAttestation(RadarAttestation):
    """(c): did the delta that LANDED in the tensor have SD sigma, on each of the 3 rows?

    This is the check that catches a dtype demotion at radar_noise.py:207
    (``.to(points.dtype)``). It does not inspect the drawn noise; it measures
    ``points_after - points_before`` element-wise, so a delta that was truncated on its way into
    the tensor reports a shrunken SD no matter how it was drawn. The recorded ``points_dtype``
    tag catches the same fault directly, from the other side.
    """

    FAMILY = "radar_doppler_rcs_noise"
    ROWS = ("rcs", "vx_comp", "vy_comp")
    FLOAT_DTYPES = ("torch.float16", "torch.float32", "torch.float64", "torch.bfloat16")

    def _family_report(self):
        sums = self.merged["sums"]
        tags = self.merged["tags"]
        failures = []
        sigma = float(self.params.get("sigma", float("nan")))

        rows = {}
        for row in self.ROWS:
            n = sums.get("%s_n" % row, 0.0)
            total = sums.get("%s_sum" % row, 0.0)
            sqsum = sums.get("%s_sqsum" % row, 0.0)
            if n <= 1:
                rows[row] = {"n": int(n), "realised_sd": float("nan"),
                             "realised_mean": float("nan"), "band": float("nan")}
                continue
            mean = total / n
            var = max(sqsum / n - mean * mean, 0.0)
            sd = math.sqrt(var)
            # SE(sd)/sd = 1/sqrt(2n) for a Gaussian; Z sigma, floored at 2% so the band never
            # tightens past the point where round-off in the accumulated sums matters.
            rel_band = max(Z / math.sqrt(2.0 * n), 0.02)
            mean_band = Z * sigma / math.sqrt(n) if sigma == sigma else float("nan")
            rows[row] = {"n": int(n), "realised_sd": sd, "realised_mean": mean,
                         "band_rel": rel_band, "sd_ratio": (sd / sigma) if sigma else float("nan")}
            if sigma and sigma == sigma:
                if abs(sd / sigma - 1.0) > rel_band:
                    failures.append(
                        "row %s: realised delta SD %.6f is %.2f%% off sigma=%.3f, outside the "
                        "%.2f%% band (n=%d) — the applied noise is not the noise the cell "
                        "declares" % (row, sd, 100.0 * (sd / sigma - 1.0), sigma,
                                      100.0 * rel_band, int(n)))
                if abs(mean) > mean_band:
                    failures.append(
                        "row %s: realised delta mean %.6f exceeds the %.6f band (n=%d); the "
                        "noise should be zero-mean" % (row, mean, mean_band, int(n)))

        dtype_tags = {k: v for k, v in tags.items() if k.startswith("points_dtype=")}
        for key, count in sorted(dtype_tags.items()):
            dtype = key.split("=", 1)[1]
            if dtype not in self.FLOAT_DTYPES:
                failures.append(
                    "%d call(s) applied the noise to a non-floating point tensor (dtype=%s); "
                    "the delta is truncated on assignment and the cell's sigma is not the "
                    "sigma that reached the model" % (count, dtype))

        changed = tags.get("untouched_rows=CHANGED", 0)
        if changed:
            failures.append(
                "%d call(s) changed a row outside {rcs, vx_comp, vy_comp}; (c) is a "
                "non-positional disturbance and must leave every other field alone" % changed)
        moved = tags.get("positions=CHANGED", 0)
        if moved:
            failures.append(
                "%d call(s) moved radar point positions (rows x,y,z); (c) must not" % moved)

        return {
            "sigma": sigma,
            "rows": rows,
            "positions_untouched_calls": int(tags.get("positions=IDENTICAL", 0)),
            "untouched_rows_identical_calls": int(tags.get("untouched_rows=IDENTICAL", 0)),
            "empty_cloud_calls": int(tags.get("empty=1", 0)),
            "tags": tags,
        }, failures


class MisalignAttestation(RadarAttestation):
    """(d2): were exactly the radar extrinsics perturbed, and every non-radar one passed through?"""

    FAMILY = "d2_extrinsic"

    def _family_report(self):
        tags = self.merged["tags"]
        sets = self.merged["sets"]
        failures = []

        # Tags are recorded as "cs=<channel>:<outcome>" (the sink encodes every tag as
        # "<key>=<value>", so the channel and outcome travel together on the value side).
        per_channel = {}
        for key, count in tags.items():
            if not key.startswith("cs="):
                continue
            payload = key[len("cs="):]
            if ":" not in payload:
                failures.append("malformed calibrated_sensor tag %r" % key)
                continue
            channel, outcome = payload.rsplit(":", 1)
            entry = per_channel.setdefault(
                channel, {"perturbed": 0, "passthrough_in_context": 0,
                          "passthrough_no_context": 0})
            if outcome in entry:
                entry[outcome] += count
            else:
                failures.append("unknown calibrated_sensor outcome %r in tag %r" % (outcome, key))

        for channel, counts in sorted(per_channel.items()):
            total = sum(counts.values())
            if channel in RADAR_CHANNELS:
                if counts["passthrough_in_context"]:
                    failures.append(
                        "radar channel %s was passed through UNPERTURBED on %d lookup(s) made "
                        "inside a (d2) context" % (channel, counts["passthrough_in_context"]))
                if counts["perturbed"] == 0:
                    failures.append(
                        "radar channel %s was never perturbed" % channel)
            else:
                if counts["perturbed"]:
                    failures.append(
                        "non-radar channel %s was perturbed on %d lookup(s); only the radar "
                        "extrinsics may move" % (channel, counts["perturbed"]))
            counts["total"] = total
            counts["passthrough_rate"] = (
                (counts["passthrough_in_context"] + counts["passthrough_no_context"]) / total
                if total else float("nan"))

        # A channel that never appears at all produces no per-channel row, so the loop above
        # cannot see it. Name it explicitly: "the hook was never installed" must read as a
        # missing perturbation, not as a missing LIDAR_TOP lookup.
        for channel in RADAR_CHANNELS:
            if channel not in per_channel:
                failures.append(
                    "radar channel %s never appeared in a calibrated_sensor lookup; the (d2) "
                    "hook was not active for it" % channel)

        lidar = per_channel.get("LIDAR_TOP")
        if lidar is None:
            failures.append(
                "no LIDAR_TOP calibrated_sensor lookup was observed; the reference-frame "
                "pass-through claim is unverified, not confirmed")
        elif lidar["perturbed"] != 0 or lidar["passthrough_rate"] != 1.0:
            failures.append(
                "LIDAR_TOP must be 100%% pass-through; observed perturbed=%d rate=%r"
                % (lidar["perturbed"], lidar["passthrough_rate"]))

        pairs = sets.get("perturbed_scene_channel", [])
        scenes = sets.get("scenes", [])
        expect_pairs = len(scenes) * len(RADAR_CHANNELS)
        if scenes and len(pairs) != expect_pairs:
            failures.append(
                "%d distinct (scene, channel) perturbations, expected %d (= %d scenes x %d "
                "radar channels)" % (len(pairs), expect_pairs, len(scenes), len(RADAR_CHANNELS)))

        return {
            "per_channel": per_channel,
            "distinct_scenes": len(scenes),
            "distinct_perturbed_scene_channel_pairs": len(pairs),
            "expected_scene_channel_pairs": expect_pairs,
            "lidar_top": lidar,
            "tags": tags,
        }, failures


class D1AsyncAttestation(RadarAttestation):
    """(d1): did every element really move OLDER along the sweep chain, by about k*77 ms?"""

    FAMILY = "d1_async"
    # Physical band for ONE sweep step. nuScenes radars run ~13 Hz; the nominal period is
    # 77 ms and measured spacing sits near 75 ms. A per-step lag outside this band means the
    # walk is not stepping the sweep chain, whatever the counters say. Registered here rather
    # than imported from a budget: it is a property of the dataset, not a tolerance we chose.
    STEP_BAND_S = (0.050, 0.110)

    def _family_report(self):
        sums = self.merged["sums"]
        tags = self.merged["tags"]
        failures = []

        walks = int(sums.get("d1_walks", 0))
        full = int(sums.get("d1_full_walks", 0))
        clamped = int(sums.get("d1_clamped_walks", 0))
        steps_taken = int(sums.get("d1_steps_taken", 0))
        steps_requested = int(sums.get("d1_steps_requested", 0))
        offset = self.merged["identity"].get("offset")

        newer = sum(c for k, c in tags.items() if k == "d1_dir=newer")
        zero = sum(c for k, c in tags.items() if k == "d1_dir=zero")

        per_step = (sums.get("d1_full_lag_per_step_sum_s", 0.0) / full) if full else float("nan")
        mean_full_lag = (sums.get("d1_full_lag_sum_s", 0.0) / full) if full else float("nan")

        if walks == 0:
            failures.append(
                "0 sweep-chain walks recorded: the (d1) offset context never wrapped a "
                "from_file_multisweep call, so no element was shifted")
        if newer:
            failures.append(
                "%d walk(s) produced NEWER radar than clean; (d1) must only ever move data "
                "older -- this is the exact failure of the superseded prev-index mechanism"
                % newer)
        if full == 0 and walks:
            failures.append(
                "every one of the %d walk(s) clamped at a scene boundary; no element actually "
                "moved" % walks)
        if zero > clamped:
            failures.append(
                "%d walk(s) produced a zero timestamp delta but only %d clamped; a walk that "
                "took steps must change the aggregation start" % (zero, clamped))
        # Partial clamped walks (chain ended after 1..k-1 steps) are LEGITIMATE under the
        # registered clamp rule ("the walk stops at the oldest sweep available") -- a mid-scene
        # sweep gap can produce one, and (d1) being deterministic, hard-failing here would
        # INVALIDate the same cell on every rerun. Measured 2026-08-03: 0 partial walks in
        # 12000; the counts stay in the evidence sums so a surprise is visible, not fatal.
        if steps_taken > steps_requested:
            failures.append("walks took %d steps but only %d were requested"
                            % (steps_taken, steps_requested))
        if offset is not None and full:
            expect = float(offset) * float(self.merged["identity"].get("sweep_period_s", 0.077))
            low, high = self.STEP_BAND_S
            if not (low <= per_step <= high):
                failures.append(
                    "mean lag per sweep step %.6f s is outside the physical band [%.3f, %.3f]; "
                    "the walk is not stepping the sweep chain" % (per_step, low, high))
            if not (offset * low <= mean_full_lag <= offset * high):
                failures.append(
                    "mean realised lag %.6f s on unclamped walks is not consistent with "
                    "offset=%s (nominal %.6f s)" % (mean_full_lag, offset, expect))

        return {
            "walks": walks,
            "full_walks": full,
            "clamped_walks": clamped,
            "steps_requested": steps_requested,
            "steps_taken": steps_taken,
            "mean_lag_per_step_s": per_step,
            "mean_realised_lag_s": mean_full_lag,
            "nominal_lag_s": (float(offset) * 0.077) if offset is not None else None,
            "walks_newer": newer,
            "walks_zero_delta": zero,
            "channels": self.merged["sets"].get("d1_channels", []),
            "step_band_s": list(self.STEP_BAND_S),
            "tags": tags,
        }, failures


ATTESTATIONS = {
    "radar_dropout": DropoutAttestation,
    "radar_doppler_rcs_noise": NoiseAttestation,
    "d2_extrinsic": MisalignAttestation,
    "d1_async": D1AsyncAttestation,
}
