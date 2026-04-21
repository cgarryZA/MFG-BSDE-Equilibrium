#!/usr/bin/env python
"""
Read results_paper_final/all_results.json and print the exact LaTeX
table rows + inline numbers for the paper. Copy-paste into main.tex.
"""

import json
import sys
import os

def main():
    path = os.path.join("results_paper_final", "all_results.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run run_all_for_paper.py first.")
        sys.exit(1)

    with open(path) as f:
        R = json.load(f)

    print("=" * 70)
    print("TABLE 1: Main sensitivity (MomentEncoder, phi=0.1)")
    print("=" * 70)
    ms = R["main_sensitivity"]["evaluations"]
    for label, key in [("Narrow (std 0.1)", "std=0.1"), ("Medium (std 1.0)", "std=1.0"), ("Wide (std 3.0)", "std=3.0")]:
        e = ms[key]
        zq_sign = "+" if e["z_q"] >= 0 else ""
        print(f"{label} & {e['h']:.3f} & ${zq_sign}{e['z_q']:.3f}$ & "
              f"{e['delta_a']:.3f} & {e['delta_b']:.3f} & "
              f"{e['nu_a']:.3f} & {e['nu_b']:.3f} \\\\")
    h_narrow = ms["std=0.1"]["h"]
    h_wide = ms["std=3.0"]["h"]
    print(f"\nh varies {h_narrow:.3f} -> {h_wide:.3f} ({h_narrow/max(h_wide,1e-6):.0f}x)")
    print(f"MV Y0 = {R['main_sensitivity']['y0']:.4f}")
    print(f"No-coupling Y0 = {R['no_coupling']['y0']:.4f}")

    # Full sweep for reference
    print(f"\nFull sweep (for abstract/inline numbers):")
    for key in ["std=0.1", "std=0.5", "std=1.0", "std=2.0", "std=3.0", "std=5.0"]:
        e = ms[key]
        print(f"  {key}: h={e['h']:.4f}, da={e['delta_a']:.4f}, db={e['delta_b']:.4f}, "
              f"da_clip={e['delta_a_clip']:.4f}, db_clip={e['delta_b_clip']:.4f}")

    print(f"\n{'='*70}")
    print("TABLE: Encoder ablation")
    print("=" * 70)
    ea = R["encoder_ablation"]
    for enc in ["moments", "quantiles", "histogram", "deepsets"]:
        ev = ea[enc]["evaluations"]
        h_vals = [ev[f"std={s}"]["h"] for s in [0.1, 1.0, 3.0]]
        print(f"{enc}: h = {h_vals[0]:.3f} / {h_vals[1]:.3f} / {h_vals[2]:.3f} "
              f"({h_vals[0]/max(h_vals[2],1e-3):.1f}x)")

    print(f"\n{'='*70}")
    print("TABLE: Multi-seed")
    print("=" * 70)
    ms2 = R["multi_seed"]
    print(f"Y0: {ms2['y0_mean']:.4f} +/- {ms2['y0_std']:.4f}")
    print(f"h-gap: {ms2['h_gap_mean']:.4f} +/- {ms2['h_gap_std']:.4f}")
    for i in range(len(ms2["y0s"])):
        print(f"  seed {i}: Y0={ms2['y0s'][i]:.4f}, h_n={ms2['h_narrow'][i]:.4f}, "
              f"h_w={ms2['h_wide'][i]:.4f}, gap={ms2['h_gaps'][i]:.4f}")

    print(f"\n{'='*70}")
    print("TABLE: Placebo")
    print("=" * 70)
    pl = R["placebo"]
    for key in ["std=0.1", "std=1.0", "std=3.0"]:
        p = pl[key]
        print(f"{key}: real={p['real_h']:.4f}, shuffled={p['shuffled_h']:.4f}, random={p['random_h']:.4f}")

    print(f"\n{'='*70}")
    print("TABLE: Disentanglement")
    print("=" * 70)
    di = R["disentanglement"]
    for label in ["narrow_both", "wide_both", "h_narrow_sub_wide", "h_wide_sub_narrow"]:
        d = di[label]
        print(f"{label}: h={d['h']:.4f}, da={d['delta_a']:.4f}, db={d['delta_b']:.4f}, z_q={d['z_q']:.6f}")

    print(f"\n{'='*70}")
    print("H-ONLY CONTROL")
    print("=" * 70)
    ho = R["h_only"]
    print(f"Y0 = {ho['y0']:.4f}")
    for key, v in ho["evaluations"].items():
        print(f"  {key}: h={v['h']:.4f}, z_q={v['z_q']:.6f}")

    print(f"\n{'='*70}")
    print("GENERALISATION")
    print("=" * 70)
    gen = R["generalisation"]
    for fam, v in gen.items():
        print(f"  {fam}: h={v['h']:.4f}, da={v['delta_a']:.4f}")

    print(f"\n{'='*70}")
    print("PENALTY SWEEP")
    print("=" * 70)
    ps = R["penalty_sweep"]
    for key, v in ps.items():
        print(f"  {key}: h_gap={v['h_gap']:.4f}, policy_gap={v['policy_gap']:.4f}")

    print(f"\n{'='*70}")
    print(f"Elapsed: {R['metadata']['elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
