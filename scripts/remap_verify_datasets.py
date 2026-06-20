#!/usr/bin/env python
"""Remap label kelas dataset verifikasi ke skema model kita (helmet/license_plate/motorcyclist).

Model kita: 0=helmet, 1=license_plate, 2=motorcyclist

Membuat salinan split uji dengan indeks kelas yang sudah dipetakan + data.yaml baru,
agar `yolo val` membandingkan prediksi model (0,1,2) terhadap ground-truth yang konsisten.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUR_NAMES = ["helmet", "license_plate", "motorcyclist"]


def remap_split(src_root: Path, splits: list[str], mapping: dict[int, int | None], out_root: Path) -> int:
    """Salin gambar+label dari beberapa split sumber ke out_root/test, remap indeks kelas.

    mapping: indeks_sumber -> indeks_target (None = buang kelas itu).
    Return jumlah gambar yang disalin.
    """
    out_img = out_root / "test" / "images"
    out_lbl = out_root / "test" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_img = 0
    for split in splits:
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            # nama unik antar-split (hindari tabrakan)
            stem = f"{split}_{img.stem}"
            shutil.copy(img, out_img / f"{stem}{img.suffix}")
            n_img += 1

            src_lbl = lbl_dir / f"{img.stem}.txt"
            lines_out = []
            if src_lbl.exists():
                for line in src_lbl.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    tgt = mapping.get(cls, None)
                    if tgt is None:
                        continue
                    lines_out.append(" ".join([str(tgt)] + parts[1:]))
            (out_lbl / f"{stem}.txt").write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return n_img


def write_yaml(out_root: Path) -> None:
    cfg = {
        "path": str(out_root.resolve()),
        "train": "test/images",  # placeholder (Ultralytics wajib ada train+val)
        "test": "test/images",
        "val": "test/images",  # arahkan val ke test agar split='val' juga bekerja
        "names": OUR_NAMES,
        "nc": len(OUR_NAMES),
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    # === Dataset MIT (kelas cocok bersih) ===
    # names: 0=helmet 1='plate ' 2=plate 3='rider ' 4=rider
    mit_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
    mit_src = REPO / "data/verify-mit-helmetplate"
    mit_out = REPO / "data/verify-mit-remap"
    n = remap_split(mit_src, ["test"], mit_map, mit_out)
    write_yaml(mit_out)
    print(f"[MIT]  remap selesai -> {mit_out} ({n} gambar test)")

    # === Dataset ID (parsial; motorcycle->motorcyclist dgn catatan; nohelmet dibuang) ===
    # names: 0=Helmet 1=motorcycle 2=nohelmet 3=platenumber
    id_map = {0: 0, 3: 1, 1: 2, 2: None}
    id_src = REPO / "data/verify-id-nohelmet"
    id_out = REPO / "data/verify-id-remap"
    n = remap_split(id_src, ["valid", "test"], id_map, id_out)  # gabung valid+test
    write_yaml(id_out)
    print(f"[ID]   remap selesai -> {id_out} ({n} gambar valid+test)")


if __name__ == "__main__":
    main()
