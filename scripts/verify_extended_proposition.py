"""
verify_extended_proposition.py

Closes the loop on theory/joint_discovery_threshold_proposition.md Part 2.
Every rank-logging script built so far (severity_sweep_pygod_instrumented.py,
condition_comparison_pygod.py, real_data_experiment.py --log_ranks) produces
a rank-grid CSV and a matching trial-level CSV, but nothing so far actually
checks whether the extended proposition's predictions matched what happened.
This script does that check, for any pair of those output files, and reports
the specific case the extended proposition exists to explain: trials where
the FLOOR-only condition (r=1) fails but a LARGER r succeeds, matching the
Amazon/clean finding in PAPER_REFRAME_HANDOFF.md section 5.5 (required
clearance 218, observed 134, floor predicts no discovery; actual result was
3,420 discoveries via rejections at larger ranks).

WHAT "CORRECT" MEANS HERE, PRECISELY. The extended proposition
(theory/.../Part 2, Proposition "generalized discovery condition") is a
SUFFICIENT, IN-EXPECTATION condition: dagger_satisfied=True at some r is
supposed to predict n_discoveries > 0, but the converse is NOT claimed --
dagger_satisfied=False everywhere does NOT establish n_discoveries == 0
(see Remark "Why the floor-only condition does not predict absence of
discovery" in the theory doc, which applies equally to the extended
version's own limits). So this script reports two different things
separately, not one merged accuracy number:

  1. SOUNDNESS check: among trials where any_r_predicts=True, what fraction
     actually had n_discoveries > 0? This is the claim the proposition
     actually makes. A soundness failure here (predicts discovery, none
     observed) is a real falsification and must be reported as one, not
     explained away.
  2. COVERAGE check (informational, not a claim of the proposition): among
     trials where any_r_predicts=False, what fraction had n_discoveries > 0
     anyway? A high number here does NOT mean the proposition is wrong --
     it means, correctly, that the sufficient condition is not necessary,
     and there are discovery mechanisms this in-expectation argument does
     not capture. Do not report this as an error rate for the proposition.

Also separately reports the FLOOR-ONLY vs EXTENDED comparison directly: how
many trials does extending the condition beyond r=1 change the prediction
for, and in which direction. This is the number that says whether the
extension was worth deriving.

Usage:
  python3 scripts/verify_extended_proposition.py \
      --trial_csv results/logs/real_data_experiment_amazon_dominant_pygod.csv \
      --rank_csv results/logs/real_data_experiment_amazon_dominant_pygod_ranks.csv

  # also works on the severity sweep and condition comparison outputs:
  python3 scripts/verify_extended_proposition.py \
      --trial_csv results/logs/severity_sweep_pygod.csv \
      --rank_csv results/logs/severity_sweep_pygod_ranks.csv

  # check every produced pair at once:
  python3 scripts/verify_extended_proposition.py --all
"""

import argparse
import csv
import os


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def infer_join_keys(trial_rows, rank_rows):
    """Different scripts key trials differently: severity_sweep_pygod uses
    (p_an, seed), condition_comparison_pygod and real_data_experiment use
    (condition, seed). Detect which is present in both files rather than
    hardcoding one, so this script works unmodified against any of the
    three sources."""
    trial_cols = set(trial_rows[0].keys())
    rank_cols = set(rank_rows[0].keys())
    shared = trial_cols & rank_cols
    if {"condition", "seed"} <= shared:
        return ["condition", "seed"]
    if {"p_an", "seed"} <= shared:
        return ["p_an", "seed"]
    raise ValueError(
        f"Could not infer join keys. Trial columns: {sorted(trial_cols)}. "
        f"Rank columns: {sorted(rank_cols)}. Expected either "
        f"(condition, seed) or (p_an, seed) in both files."
    )


def verify(trial_csv_path, rank_csv_path):
    trial_rows = load_csv(trial_csv_path)
    rank_rows = load_csv(rank_csv_path)

    if not trial_rows or not rank_rows:
        print(f"  SKIP: empty file(s) -- {trial_csv_path} or {rank_csv_path}")
        return

    join_keys = infer_join_keys(trial_rows, rank_rows)

    # group rank rows by trial key, sorted by r so index 0 is always the
    # smallest r in the grid (closest to the floor, though not necessarily
    # exactly r=1 depending on n_calib and n_rank_points)
    from collections import defaultdict
    ranks_by_trial = defaultdict(list)
    for row in rank_rows:
        key = tuple(row[k] for k in join_keys)
        ranks_by_trial[key].append(row)
    for key in ranks_by_trial:
        ranks_by_trial[key].sort(key=lambda r: int(r["r"]))

    total = 0
    skipped_no_rank_data = 0

    # soundness: any_r_predicts=True -> observed=True ?
    predicted_yes = []  # list of (observed_bool,)
    predicted_no = []   # list of (observed_bool,)

    # floor-only vs extended disagreement
    floor_yes_extended_yes = 0
    floor_no_extended_yes = 0   # the case the extension exists for
    floor_yes_extended_no = 0   # should be impossible (floor is r=1, always in the any-r check)
    floor_no_extended_no = 0

    disagreement_examples = []

    for trial in trial_rows:
        key = tuple(trial[k] for k in join_keys)
        if key not in ranks_by_trial:
            skipped_no_rank_data += 1
            continue
        total += 1
        rows = ranks_by_trial[key]
        floor_row = rows[0]  # smallest r in the grid
        floor_predicts = floor_row["dagger_satisfied"] in ("True", "true", "1")
        any_predicts = any(r["dagger_satisfied"] in ("True", "true", "1") for r in rows)
        observed = int(trial["n_discoveries"]) > 0

        if any_predicts:
            predicted_yes.append(observed)
        else:
            predicted_no.append(observed)

        if floor_predicts and any_predicts:
            floor_yes_extended_yes += 1
        elif not floor_predicts and any_predicts:
            floor_no_extended_yes += 1
            if len(disagreement_examples) < 5:
                satisfying_r = next(r["r"] for r in rows if r["dagger_satisfied"] in ("True", "true", "1"))
                disagreement_examples.append(
                    f"    key={key}: floor (r={floor_row['r']}) fails, "
                    f"but r={satisfying_r} succeeds. n_discoveries={trial['n_discoveries']} "
                    f"(observed={'discovery' if observed else 'no discovery'})"
                )
        elif floor_predicts and not any_predicts:
            floor_yes_extended_no += 1  # should not happen; floor is always checked in any_predicts
        else:
            floor_no_extended_no += 1

    print(f"\n=== {os.path.basename(trial_csv_path)} vs {os.path.basename(rank_csv_path)} ===")
    print(f"Join keys: {join_keys}")
    print(f"Trials matched: {total} (skipped, no rank data: {skipped_no_rank_data})")

    n_yes = len(predicted_yes)
    n_no = len(predicted_no)
    if n_yes > 0:
        sound_frac = sum(predicted_yes) / n_yes
        print(f"\nSOUNDNESS (the actual claim): of {n_yes} trials where the extended "
              f"condition predicted discovery, {sum(predicted_yes)} actually had "
              f"discoveries ({sound_frac:.1%}).")
        if sound_frac < 1.0:
            n_fail = n_yes - sum(predicted_yes)
            print(f"  *** {n_fail} SOUNDNESS FAILURE(S): predicted discovery, none observed. ***")
            print(f"  This is a real falsification of the extended proposition on this data")
            print(f"  and must be reported as such, not explained away. Report it in the paper.")
        else:
            print(f"  No soundness failures on this data: every predicted discovery occurred.")
    else:
        print(f"\nSOUNDNESS: no trials where the extended condition predicted discovery -- "
              f"nothing to check here.")

    if n_no > 0:
        coverage_frac = sum(predicted_no) / n_no
        print(f"\nCOVERAGE (informational only, NOT a claim of the proposition): of {n_no} "
              f"trials where the extended condition predicted NO discovery, {sum(predicted_no)} "
              f"had discoveries anyway ({coverage_frac:.1%}). This does not falsify the "
              f"proposition -- it is a sufficient, not necessary, condition.")

    print(f"\nFLOOR-ONLY vs EXTENDED comparison:")
    print(f"  floor succeeds AND extended succeeds (extension made no difference): {floor_yes_extended_yes}")
    print(f"  floor FAILS but extended succeeds (this is the case the extension exists for): {floor_no_extended_yes}")
    print(f"  floor succeeds but extended fails (should be impossible -- investigate if nonzero): {floor_yes_extended_no}")
    print(f"  floor fails AND extended fails (neither predicts discovery): {floor_no_extended_no}")

    if floor_no_extended_yes > 0:
        print(f"\n  {floor_no_extended_yes} trial(s) where extending beyond the floor changed the")
        print(f"  prediction from 'no discovery' to 'discovery'. Examples:")
        for ex in disagreement_examples:
            print(ex)
    else:
        print(f"\n  No trials where the extension changed the prediction on this data.")
        print(f"  (This is expected if AUROC is very high across the board -- the floor")
        print(f"  condition alone already succeeds whenever detection is easy. The extension's")
        print(f"  value shows up specifically on harder, lower-power cases like Amazon/clean.)")

    if floor_yes_extended_no > 0:
        print(f"\n  WARNING: floor_yes_extended_no={floor_yes_extended_no} should be mathematically")
        print(f"  impossible (the extended check always includes r=1). This indicates a bug in")
        print(f"  either this script or the rank-logging code that produced the CSV -- investigate")
        print(f"  before trusting any other number in this report.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_csv", type=str, default=None)
    parser.add_argument("--rank_csv", type=str, default=None)
    parser.add_argument("--all", action="store_true",
                         help="Auto-discover every trial/rank CSV pair in results/logs/ "
                              "and check each one, instead of specifying a single pair.")
    args = parser.parse_args()

    if args.all:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
        if not os.path.isdir(log_dir):
            print(f"No results/logs directory found at {log_dir}. Run an experiment first.")
            return
        rank_files = [f for f in os.listdir(log_dir) if f.endswith("_ranks.csv")]
        if not rank_files:
            print(f"No *_ranks.csv files found in {log_dir}. Run a rank-logging experiment "
                  f"first (severity_sweep_pygod_instrumented.py, condition_comparison_pygod.py, "
                  f"or real_data_experiment.py --log_ranks).")
            return
        for rank_file in sorted(rank_files):
            trial_file = rank_file.replace("_ranks.csv", ".csv")
            trial_path = os.path.join(log_dir, trial_file)
            rank_path = os.path.join(log_dir, rank_file)
            if not os.path.exists(trial_path):
                print(f"\nSKIP {rank_file}: matching trial file {trial_file} not found.")
                continue
            verify(trial_path, rank_path)
        return

    if not args.trial_csv or not args.rank_csv:
        parser.error("Provide both --trial_csv and --rank_csv, or use --all.")

    verify(args.trial_csv, args.rank_csv)


if __name__ == "__main__":
    main()