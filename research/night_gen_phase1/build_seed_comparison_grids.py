"""Build N comparison grids over tokens that exist in both seed-1 and seed-2 ratio18p75 manifests.

Layout per token (PIL Image): 3 panoramic 3×2 panels stacked vertically.
  Row A: original day JPG (1600×900 each, downsampled to 683×384 for consistency)
  Row B: seed20260425 ratio18p75 generated PNG (683×384)
  Row C: seed20260502 ratio18p75 generated PNG (683×384)

Each panel: 3 cams across × 2 cams down (front-row, back-row).
"""

import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = "/srv/nfs/shared/gnmp/RaCFormer"
OUT_DIR = "/srv/nfs/shared/gnmp/RaCFormer/research/night_gen_phase1/reports/seed_compare_grids"

M1_PATH = "research/night_gen_phase1/manifests/phase1_t10_seed20260425_ratio18p75_manifest.json"
M2_PATH = "research/night_gen_phase1/manifests/phase1_t10_seed20260502_ratio18p75_generated.json"

CELL_W, CELL_H = 683, 384
LABEL_H = 32
PANEL_GAP = 8

CAM_LAYOUT = [
    ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"),
]


def _load_manifest(path):
    with open(os.path.join(REPO_ROOT, path)) as f:
        d = json.load(f)
    by_tok_cam = {}
    for e in d["entries"]:
        if e.get("status") != "ok":
            continue
        by_tok_cam[(e["sample_token"], e["camera"])] = (
            e["generated_path"],
            e["cluster_path"],  # original
        )
    return by_tok_cam


def _load_resize(p, cluster_or_repo):
    abs_p = p if p.startswith("/") else os.path.join(REPO_ROOT, p)
    im = Image.open(abs_p).convert("RGB")
    if im.size != (CELL_W, CELL_H):
        im = im.resize((CELL_W, CELL_H), Image.BICUBIC)
    return im


def _draw_label(panel: Image.Image, text: str):
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, panel.width, LABEL_H], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)


def build_panel(label: str, get_cell):
    panel_w = CELL_W * 3
    panel_h = CELL_H * 2 + LABEL_H
    panel = Image.new("RGB", (panel_w, panel_h), (40, 40, 40))
    for r, row in enumerate(CAM_LAYOUT):
        for c, cam in enumerate(row):
            cell = get_cell(cam)
            panel.paste(cell, (c * CELL_W, LABEL_H + r * CELL_H))
    _draw_label(panel, label)
    return panel


def build_grid_for_token(tok, m1, m2, out_path):
    cams = [c for row in CAM_LAYOUT for c in row]
    # Sanity
    for cam in cams:
        if (tok, cam) not in m1 or (tok, cam) not in m2:
            return False, f"missing cam {cam} in one of the manifests"

    def cell_orig(cam):
        return _load_resize(
            m1[(tok, cam)][1], "cluster"
        )  # seed-1 cluster_path == original

    def cell_seed1(cam):
        return _load_resize(m1[(tok, cam)][0], "repo")

    def cell_seed2(cam):
        return _load_resize(m2[(tok, cam)][0], "repo")

    p_orig = build_panel(f"original (day)  •  token={tok}", cell_orig)
    p1 = build_panel(f"seed20260425 ratio18p75 (n=375 partition)", cell_seed1)
    p2 = build_panel(f"seed20260502 ratio18p75 (n=374 direct)", cell_seed2)

    grid_w = p_orig.width
    grid_h = p_orig.height + p1.height + p2.height + 2 * PANEL_GAP
    grid = Image.new("RGB", (grid_w, grid_h), (10, 10, 10))
    y = 0
    for p in (p_orig, p1, p2):
        grid.paste(p, (0, y))
        y += p.height + PANEL_GAP

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path, "PNG", optimize=True)
    return True, None


def main():
    n_grids = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 13

    m1 = _load_manifest(M1_PATH)
    m2 = _load_manifest(M2_PATH)
    t1 = {tok for (tok, _) in m1.keys()}
    t2 = {tok for (tok, _) in m2.keys()}
    overlap = sorted(t1 & t2)
    print(f"[grids] overlap: {len(overlap)} tokens")

    rng = random.Random(seed)
    rng.shuffle(overlap)
    selected = overlap[:n_grids]

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for i, tok in enumerate(selected, 1):
        out_path = os.path.join(OUT_DIR, f"compare_seed1_seed2_{i:02d}_{tok[:12]}.png")
        ok, err = build_grid_for_token(tok, m1, m2, out_path)
        status = "OK" if ok else f"FAIL ({err})"
        print(f"[grids] {i}/{n_grids}  {tok}  -> {status}")
        if ok:
            rows.append((i, tok, os.path.basename(out_path)))

    # tiny HTML index
    html_path = os.path.join(OUT_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write("<title>seed-1 vs seed-2 ratio18p75 comparison</title>")
        f.write(
            "<style>body{background:#111;color:#ddd;font-family:system-ui;margin:20px}"
            "img{max-width:100%;display:block;margin:24px 0;border:1px solid #333}"
            "h2{margin-top:48px}</style></head><body>"
        )
        f.write(
            "<h1>RaCFormer Phase 1 T10 — seed-1 vs seed-2 ratio18p75 cohort comparison</h1>"
        )
        f.write(
            "<p>Same sample token across both seeds. Top: original day. "
            "Middle: seed20260425 generation. Bottom: seed20260502 generation.</p>"
        )
        for i, tok, fn in rows:
            f.write(f"<h2>#{i} — token <code>{tok}</code></h2>")
            f.write(f"<img src='{fn}' alt='compare {tok}'>")
        f.write("</body></html>")
    print(f"[grids] wrote {html_path} ({len(rows)} grids)")


if __name__ == "__main__":
    main()
