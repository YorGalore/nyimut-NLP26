"""Audit angka Keputusan 46-48 (docs/keputusan.md) terhadap file output.

Cara kerja
----------
1. Ambil teks dari "## 46." sampai akhir docs/keputusan.md.
2. Tabel markdown di-parse per sel dan dicocokkan dengan CSV/JSON sumbernya.
3. Klaim di prosa dicocokkan dengan regex (grup tangkap = angka di keputusan).
   Angka di keputusan dibandingkan dengan nilai file yang dibulatkan ke jumlah
   desimal yang tertulis.
4. Cakupan: setiap token angka di bagian itu harus dikonsumsi oleh salah satu
   pemeriksaan; token yang tersisa dilaporkan (ACUAN = rujukan nomor keputusan /
   nama identifier, bukan klaim numerik).

Output: data/processed/ablation/audit_keputusan_46_48.csv (+ ringkasan di stdout).
Skrip ini hanya membaca file; tidak ada model yang dilatih atau dievaluasi.

Jalankan dari root repository:
    .venv/bin/python src/audit_keputusan_46_48.py
"""

from __future__ import annotations

import filecmp
import importlib.util
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from dataset_split import LM_FEATURES, PRICE_FEATURES, XGB_GRID  # noqa: E402
from evaluate_feature_fusion import load_daily_numeric_features  # noqa: E402
from evaluate_late_fusion import BOOTSTRAP_BLOCK, BOOTSTRAP_RESAMPLES, NER_FILE, TOPIC_DIR  # noqa: E402
from train_combined import N_SVD_COMPONENTS  # noqa: E402

ROOT = C.ROOT
DOC = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "docs" / "keputusan.md"  # argumen: uji negatif
AB = ROOT / "data" / "processed" / "ablation"
BP = ROOT / "data" / "processed" / "bert_path"
TM = ROOT / "data" / "processed" / "topic_modelling"
FEAT = TM / "features"
OUT = AB / "audit_keputusan_46_48.csv" if len(sys.argv) == 1 else Path(sys.argv[1]).resolve().with_suffix(".audit.csv")

full_doc = DOC.read_text(encoding="utf-8")
start = full_doc.index("## 46. ")
text = full_doc[start:]
rows: list[dict] = []
consumed: list[tuple[int, int]] = []


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


WORDS = {"satu": 1, "dua": 2, "keempat": 4}


def parse_num(s: str) -> float:
    if s.strip() in WORDS:
        return float(WORDS[s.strip()])
    s = s.strip().replace("−", "-").replace("+", "").replace("**", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    return float(s)


def decimals(s: str) -> int:
    s = s.strip()
    return len(s.split(",")[1]) if "," in s else 0


def record(section, doc_str, file_val, source, key, ok=None, span=None, note=""):
    if ok is None:
        if isinstance(file_val, str):
            ok = doc_str == file_val
        else:
            d = decimals(doc_str)
            ok = abs(round(float(file_val), d) - parse_num(doc_str)) < 1e-9
    rows.append({"keputusan": section, "angka_di_keputusan": doc_str, "nilai_di_file": file_val,
                 "file": source, "kolom_atau_kunci": key, "cocok": bool(ok), "catatan": note})
    if span:
        consumed.append(span)


def section_of(pos: int) -> str:
    absolute = start + pos
    heads = [(m.start(), m.group(1)) for m in re.finditer(r"^## (\d+)\.", full_doc, re.M)]
    return [h for p, h in heads if p <= absolute][-1]


def claim(pattern: str, values: list, source: str, keys: list[str], min_count: int = 1, note: str = ""):
    """Semua kemunculan pattern; grup i dibandingkan dengan values[i]."""
    ms = list(re.finditer(pattern, text, re.M))
    if len(ms) < min_count:
        rows.append({"keputusan": "?", "angka_di_keputusan": f"<pola tidak ditemukan: {pattern}>",
                     "nilai_di_file": None, "file": source, "kolom_atau_kunci": ";".join(keys),
                     "cocok": False, "catatan": note})
        return
    for m in ms:
        for i, (v, k) in enumerate(zip(values, keys), start=1):
            record(section_of(m.start()), m.group(i), v, source, k, span=m.span(i), note=note)


def table_after(header_regex: str):
    """Yield (cells, cell_spans) untuk baris data tabel yang header-nya cocok."""
    m = re.search(header_regex, text, re.M)
    assert m, header_regex
    pos = text.index("\n", m.start()) + 1
    pos = text.index("\n", pos) + 1  # lewati garis pemisah |---|
    while pos < len(text) and text[pos] == "|":
        end = text.index("\n", pos)
        line = text[pos:end]
        cells, spans, cur = [], [], pos + 1
        for part in line.split("|")[1:-1]:
            cells.append(part.strip())
            lead = len(part) - len(part.lstrip())
            spans.append((cur + lead, cur + lead + len(part.strip())))
            cur += len(part) + 1
        yield cells, spans
        pos = end + 1


def cell_nums(cell: str, span):
    """Angka di sel; token yang menempel pada huruf (C1, E4, Top2Vec, Qwen2.5) dicatat sebagai ACUAN."""
    out = []
    for m in re.finditer(r"[−+-]?\d+(?:[.,]\d+)*", cell):
        if m.start() > 0 and (cell[m.start() - 1].isalpha() or cell[m.start() - 1] in "_."):
            rows.append({"keputusan": "-", "angka_di_keputusan": m.group(), "nilai_di_file": cell,
                         "file": "-", "kolom_atau_kunci": "bagian nama/identifier", "cocok": True,
                         "catatan": "ACUAN identifier"})
            consumed.append((span[0] + m.start(), span[0] + m.end()))
            continue
        out.append((m.group(), (span[0] + m.start(), span[0] + m.end())))
    return out


# ============================================================
# Sumber
# ============================================================
sel_json = json.loads((AB / "model_c_selection.json").read_text(encoding="utf-8"))
mc = sel_json["model_c"]
prev2 = sel_json["previous_selection"]
prev1 = prev2["previous_selection"]
gap = sel_json["gap_to_best_cv_overall"]
cv_prot = sel_json["cv_protocol"]
meta = json.loads((AB / "ablation_metadata.json").read_text(encoding="utf-8"))
matrix = pd.read_csv(AB / "ablation_matrix.csv").set_index("row")
pair = pd.read_csv(AB / "ablation_pairwise.csv").set_index("comparison")
preds = pd.read_csv(AB / "ablation_test_predictions.csv")
cvb = pd.read_csv(AB / "model_c_cv_bert_path.csv").set_index("run_id")
cv1 = pd.read_csv(AB / "model_c_cv_results.csv").set_index("candidate_id")
copt_sum = json.loads((AB / "c_vs_c_optimized_summary.json").read_text(encoding="utf-8"))
bert_meta = json.loads((BP / "metadata.json").read_text(encoding="utf-8"))
ts = bert_meta["text_stats"]
F_SEL, F_LDA = FEAT / "selected_model_daily_topic_features.csv", FEAT / "lda_daily_topic_features.csv"
sel_csv, lda_csv = pd.read_csv(F_SEL), pd.read_csv(F_LDA)
sel_model_json = json.loads((TM / "selected_model.json").read_text(encoding="utf-8")) \
    if (TM / "selected_model.json").exists() else json.loads(next(TM.rglob("selected_model.json")).read_text())
SEL_MODEL_PATH = rel(next(TM.rglob("selected_model.json")))
M, PW, MJ, AM = rel(AB / "ablation_matrix.csv"), rel(AB / "ablation_pairwise.csv"), \
    rel(AB / "model_c_selection.json"), rel(AB / "ablation_metadata.json")

# ============================================================
# Keputusan 46
# ============================================================
claim(r"Hasil: (1\.202) baris × (\d+) kolom \(kolom `model` = `(lda)`\)",
      [len(sel_csv), sel_csv.shape[1], str(sel_csv["model"].unique()[0])], rel(F_SEL),
      ["n_baris", "n_kolom", "model (unik)"])
num = sel_csv.select_dtypes("number").columns
maxdiff = float(np.nanmax(np.abs(sel_csv[num].to_numpy() - lda_csv[num].to_numpy())))
m = re.search(r"selisih numerik maksimum (4,4) × (10)⁻¹⁶", text)
record("46", m.group(1) + " × 10⁻¹⁶", f"{maxdiff:.1e}", f"{rel(F_SEL)} vs {rel(F_LDA)}", "max |selisih| numerik",
       ok=f"{maxdiff:.1e}" == "4.4e-16", span=(m.start(1), m.end(2)))
record("46", "kolom non-numerik sama", str(sel_csv.drop(columns=num).equals(lda_csv.drop(columns=num))),
       rel(F_SEL), "kolom non-numerik", ok=sel_csv.drop(columns=num).equals(lda_csv.drop(columns=num)))
record("46", "lda", sel_model_json["selected_model"], SEL_MODEL_PATH, "selected_model")
record("46", "manual_final_selection", sel_model_json["selection_type"], SEL_MODEL_PATH, "selection_type")
old = pd.read_csv(io.StringIO(subprocess.check_output(
    ["git", "show", f"HEAD:{rel(F_SEL)}"], cwd=ROOT).decode()))
claim(r"Versi sebelumnya berisi fitur \*\*Top2Vec\*\* \((1\.202) × (141), kolom `model` = `(top2vec)`\)",
      [len(old), old.shape[1], str(old["model"].unique()[0])], f"git HEAD:{rel(F_SEL)}",
      ["n_baris", "n_kolom", "model (unik)"])
changed = subprocess.check_output(["git", "status", "--porcelain", "--", rel(FEAT)], cwd=ROOT).decode().split("\n")
changed = [c for c in changed if c.strip()]
others = sorted(p.name for p in FEAT.glob("*.csv") if p != F_SEL)
record("46", "Lima file lain tidak diubah", f"{len(others)} file lain; berubah: {changed}", "git status " + rel(FEAT),
       "porcelain", ok=len(others) == 5 and all(F_SEL.name in c for c in changed))
gensim = importlib.util.find_spec("gensim") is not None
record("46", "gensim tidak terpasang", f"find_spec('gensim') -> {'ada' if gensim else 'None'}", ".venv", "importlib",
       ok=not gensim)
# pembaca file: grep di src/ dan notebook/
hits = subprocess.run(["grep", "-rn", "--include=*.py", "--include=*.ipynb", "selected_model_daily_topic_features",
                       "src", "notebook"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
hits = [h for h in hits if "audit_keputusan_46_48.py" not in h]
writes_only = all(("save_csv" in h or "to_csv" in h) and "read_csv" not in h for h in hits)
record("46", "Tidak ada skrip di src/ yang membaca file ini", " || ".join(h[:120] for h in hits),
       "grep -rn src/ notebook/ (*.py, *.ipynb)", "semua kemunculan = penulisan", ok=writes_only,
       note="satu-satunya kemunculan adalah penulisan (save_csv) di evaluate.py")
claim(r"untuk (keempat) model topik \(LDA, BERTopic, DTM, Top2Vec\)",
      [len(list(TOPIC_DIR.glob("*/daily_topic_features.csv")))], rel(TOPIC_DIR),
      ["folder <model>/daily_topic_features.csv"],
      note=f"ada: {sorted(p.parent.name for p in TOPIC_DIR.glob('*/daily_topic_features.csv'))}")

# ============================================================
# Keputusan 47
# ============================================================
finbert_cols = pd.read_csv(BP / "finbert_fulltext_features.csv", nrows=0).shape[1] - 1
top2vec_dim = load_daily_numeric_features(TOPIC_DIR / "top2vec" / "daily_topic_features.csv", "topic",
                                          one_hot_dominant_topic=True).shape[1] - 1
ner_dim = load_daily_numeric_features(NER_FILE, "ner", one_hot_dominant_topic=False).shape[1] - 1
dims = {"Harga": (len(PRICE_FEATURES), "dataset_split.PRICE_FEATURES", "len"),
        "Embedding": (finbert_cols, rel(BP / "finbert_fulltext_features.csv"), "n_kolom - date"),
        "Topik": (top2vec_dim, rel(TOPIC_DIR / "top2vec" / "daily_topic_features.csv"),
                  "load_daily_numeric_features(one_hot) - date"),
        "NER": (ner_dim, rel(NER_FILE), "load_daily_numeric_features - date"),
        "**Total**": (mc["n_features"], MJ, "model_c.n_features")}
for cells, spans in table_after(r"^\| Blok \| Sumber \| Kolom \|"):
    v, src, key = dims[cells[0]]
    s, sp = cell_nums(cells[2], spans[2])[0]
    record("47", s, v, src, key, span=sp)
    if cells[0] == "Harga":
        for s2, sp2 in cell_nums(cells[1], spans[1]):
            record("47", s2, "37", "-", "rujukan Keputusan", ok=s2 == "37", span=sp2, note="ACUAN")
record("47", "1036 (hitung)", len(PRICE_FEATURES) + finbert_cols + top2vec_dim + ner_dim, "jumlah blok",
       "10+768+204+54", ok=len(PRICE_FEATURES) + finbert_cols + top2vec_dim + ner_dim == mc["n_features"])
claim(r"Macro-F1 walk-forward CV: \*\*(0,7744) ± (0,0337)\*\* \(akurasi CV (0,7885)\)",
      [mc["cv_macro_f1_mean"], mc["cv_macro_f1_std"], mc["cv_accuracy_mean"]], MJ,
      ["model_c.cv_macro_f1_mean", "model_c.cv_macro_f1_std", "model_c.cv_accuracy_mean"])
folds = cv_prot["folds"]
n_cv_days = folds[0]["n_train"] + sum(f["n_eval"] for f in folds)
claim(r"\*\*Protokol CV\*\* \(sama untuk semua tahap\): (937) hari train \+ valid kanonis \((16)-(09)-(2021) s\.d\. "
      r"(09)-(12)-(2025)\), `TimeSeriesSplit\((5)\)` dengan expanding window \(train fold pertama (157) hari, "
      r"tiap fold evaluasi (156) hari\)",
      [n_cv_days, 16, 9, 2021, 9, 12, 2025, len(folds), folds[0]["n_train"], folds[0]["n_eval"]], MJ,
      ["cv_protocol.folds (n_train fold1 + n_eval)", "train_start hari", "train_start bulan", "train_start tahun",
       "eval_end terakhir hari", "bulan", "tahun", "len(folds)", "folds[0].n_train", "folds[i].n_eval"],
      note=f"train_start={folds[0]['train_start']}, eval_end={folds[-1]['eval_end']}, "
           f"semua n_eval={[f['n_eval'] for f in folds]}")
assert folds[0]["train_start"] == "2021-09-16" and folds[-1]["eval_end"] == "2025-12-09"
assert all(f["n_eval"] == 156 for f in folds)
nc = prev1["n_candidates_in_validation_files"]
pools = prev1["finalist_rule"]["pools"]
claim(r"\*\*Finalis:\*\* (13) kandidat", [prev1["n_finalists"]], MJ, ["previous.previous.n_finalists"])
claim(r"top-(5) early fusion \(dari (240) kandidat\)", [int(pools["early_fusion"][-1]), nc["early_fusion"]], MJ,
      ["finalist_rule.pools.early_fusion", "n_candidates_in_validation_files.early_fusion"])
claim(r"top-(5) late fusion \(dari (3\.888) kandidat\)", [int(pools["late_fusion"][-1]), nc["late_fusion"]], MJ,
      ["finalist_rule.pools.late_fusion", "n_candidates_in_validation_files.late_fusion"])
claim(r"top-(3) feature fusion temporal E(5)–E(8) \(dari (48)\)",
      [int(pools["feature_fusion_temporal"][4]), 5, 8, nc["feature_fusion_temporal_validation"]], MJ,
      ["finalist_rule.pools.feature_fusion_temporal", "E5 di pools", "E8 di pools",
       "n_candidates_in_validation_files.feature_fusion_temporal_validation"])
assert "E5_" in pools["feature_fusion_temporal"] and "E8_" in pools["feature_fusion_temporal"]
n_pass = int(cv1["passes_validation_filter"].sum())
claim(r"Semua (13) kandidat lolos filter \"≤ (1) hari benar", [n_pass, 1], rel(AB / "model_c_cv_results.csv"),
      ["sum(passes_validation_filter)", "select_model_c.EQUIVALENT_DAYS"])
p1 = prev1["model_c"]
claim(r"Kandidat terbaik CV adalah `early_fusion_5` \((0,7773) ± (0,0320)\), sehingga ambangnya (0,7453)\. "
      r"Sebanyak (12) dari (13) kandidat",
      [p1["best_cv_macro_f1_mean"], p1["best_cv_macro_f1_std"], p1["equivalence_threshold"],
       len(p1["equivalent_candidates"]), prev1["n_finalists_evaluated_in_cv"]], MJ,
      ["prev1.model_c.best_cv_macro_f1_mean", "best_cv_macro_f1_std", "equivalence_threshold",
       "len(equivalent_candidates)", "n_finalists_evaluated_in_cv"])
claim(r"Hasilnya `late_fusion_3`: kurs \+ Word2Vec fulltext, meta-LR, (112) fitur, (0,7624) ± (0,0176)",
      [p1["n_features"], p1["cv_macro_f1_mean"], p1["cv_macro_f1_std"]], MJ,
      ["prev1.model_c.n_features", "cv_macro_f1_mean", "cv_macro_f1_std"])
ch = bert_meta["chunking"]
claim(r"chunk (512) token dengan overlap (50) \(Keputusan (24)\)", [ch["max_length"], ch["overlap"], 24],
      rel(BP / "metadata.json"), ["chunking.max_length", "chunking.overlap", "ACUAN"])
claim(r"Ekstraksi ulang dilakukan untuk top-(2) early dan top-(2) late",
      [sum(c.startswith("early") for c in prev2["candidates"]), sum(c.startswith("late") for c in prev2["candidates"])],
      MJ, ["previous_selection.candidates early", "previous_selection.candidates late"])
claim(r"Hasilnya `early_fusion_5__old` \((0,7773)\)", [prev2["model_c"]["cv_macro_f1_mean"]], MJ,
      ["previous_selection.model_c.cv_macro_f1_mean"])
assert prev2["model_c"]["run_id"] == "early_fusion_5__old"
for cells, spans in table_after(r"^\| Kandidat \| Embedding \| Fitur \| Macro-F1 lama"):
    cid = cells[0].split()[0]
    new_id, old_id = f"{cid}__new", f"{cid}__old"
    src = rel(AB / "model_c_cv_bert_path.csv")
    for s, sp in cell_nums(cells[0], spans[0]):  # sisa angka di nama kandidat (mis. "_5")
        record("47", s, cid, src, "candidate_id", ok=s in cid, span=sp, note="ACUAN")
    record("47", cells[0].split("(")[1].split()[0], cvb.loc[old_id, "experiment"], src, f"{old_id}.experiment",
           ok=cvb.loc[old_id, "experiment"].startswith(cells[0].split("(")[1].split()[0]))
    for s, sp in cell_nums(cells[1], spans[1]):
        record("47", s, cells[1], "-", "nama model", ok=True, span=sp, note="ACUAN nama model")
    s, sp = cell_nums(cells[2], spans[2])[0]
    record("47", s, cvb.loc[old_id, "n_features"], src, f"{old_id}.n_features", span=sp)
    (m1, sp1), (s1, sps1) = cell_nums(cells[3], spans[3])
    record("47", m1, cvb.loc[old_id, "cv_macro_f1_mean"], src, f"{old_id}.cv_macro_f1_mean", span=sp1)
    record("47", s1, cvb.loc[old_id, "cv_macro_f1_std"], src, f"{old_id}.cv_macro_f1_std", span=sps1)
    if new_id in cvb.index:
        (m2, sp2), (s2, sps2) = cell_nums(cells[4], spans[4])
        record("47", m2, cvb.loc[new_id, "cv_macro_f1_mean"], src, f"{new_id}.cv_macro_f1_mean", span=sp2)
        record("47", s2, cvb.loc[new_id, "cv_macro_f1_std"], src, f"{new_id}.cv_macro_f1_std", span=sps2)
        d, spd = cell_nums(cells[5], spans[5])[0]
        record("47", d, cvb.loc[new_id, "delta_macro_f1_new_minus_old"], src,
               f"{new_id}.delta_macro_f1_new_minus_old", span=spd)
    else:
        record("47", cells[4], "tidak ada run baru", src, f"{new_id} (tidak ada)", ok=cells[4] == "–")
claim(r"\*\*`early_fusion_5` dengan RoBERTa lama\*\* \(CV tertinggi, (0,7773) ± (0,0320)\)",
      [gap["best_cv_overall_macro_f1_mean"], gap["best_cv_overall_macro_f1_std"]], MJ,
      ["gap_to_best_cv_overall.best_cv_overall_macro_f1_mean", "best_cv_overall_macro_f1_std"])
claim(r"Selisihnya dengan Model C hanya (0,0029), jauh di bawah SD antar-fold kedua model \((0,0320) dan (0,0337)\)",
      [gap["difference"], gap["best_cv_overall_macro_f1_std"], gap["model_c_macro_f1_std"]], MJ,
      ["gap_to_best_cv_overall.difference", "best_cv_overall_macro_f1_std", "model_c_macro_f1_std"])
claim(r"Validasi (165) hari ikut menjadi bagian fold CV terakhir, sehingga skor fold (5)",
      [meta["split"]["valid"][2], len(folds)], f"{AM}; {MJ}", ["split.valid[2]", "len(cv_protocol.folds)"])
claim(r"di (165) hari validasi yang sama", [meta["split"]["valid"][2]], AM, ["split.valid[2]"])
claim(r"CV walk-forward di (937) hari", [n_cv_days], MJ, ["cv_protocol.folds"])
claim(r"Jalur BERT \(CV, (937) hari\)", [n_cv_days], MJ, ["cv_protocol.folds"])
claim(r"Hanya top-(2) kandidat per arsitektur", [sum(c.startswith("early") for c in prev2["candidates"])], MJ,
      ["previous_selection.candidates early"])
claim(r"di bawah #(1) arsitektur", [1], "-", ["peringkat #1 (definisi filter)"], note="ACUAN")
one_sd = "1 SD" in prev1["selection_rule"]
claim(r"terbaik − (1) SD antar-fold", [1 if one_sd else -1], MJ, ["prev1.selection_rule memuat '1 SD'"])
claim(r"tanpa aturan (1) SD", [1 if "1 SD" in prev2["selection_rule"] else -1], MJ,
      ["previous_selection.selection_rule memuat '1 SD'"])
for m_ in re.finditer(r"^(\d)\. \*\*(Aturan awal|Perubahan pertama|Perubahan kedua)", text, re.M):
    record(section_of(m_.start()), m_.group(1), m_.group(2), "-", "penomoran daftar", ok=True, span=m_.span(1),
           note="ACUAN penomoran")
k23 = full_doc[full_doc.index("## 23. "):full_doc.index("## 24. ")]
claim(r"Keputusan (23) menyebut (454) artikel tanpa body, sedangkan `src/bert_text\.py` menemukan (343) \((341) "
      r"memakai description sebagai fallback, (2) hanya judul\) dari (6\.013) artikel",
      [23, 454 if "454" in k23 else -1, ts["n_body_empty"], ts["n_description_fallback"], ts["n_title_only"],
       ts["n_matched"]], rel(BP / "metadata.json"),
      ["ACUAN", "docs/keputusan.md ## 23 memuat '454'", "text_stats.n_body_empty", "text_stats.n_description_fallback",
       "text_stats.n_title_only", "text_stats.n_matched"])
claim(r"Keputusan (21)–(24)\.", [21, 24], "-", ["ACUAN", "ACUAN"])

# ============================================================
# Keputusan 48
# ============================================================
sp_ = meta["split"]
claim(r"Split kanonis (772) / (165) / (167) hari \(Keputusan (35)\)",
      [sp_["train"][2], sp_["valid"][2], sp_["test"][2], 35], AM, ["split.train[2]", "split.valid[2]", "split.test[2]",
                                                                   "ACUAN"])
claim(r"`XGB_GRID` \((27) kombinasi, Keputusan (41)\)", [len(XGB_GRID), 41], "dataset_split.XGB_GRID",
      ["len", "ACUAN"])
claim(r"`random_state=(42)`", [42], AM, ["uniform_tuning.fixed.random_state"])
assert meta["uniform_tuning"]["fixed"]["random_state"] == 42
for cells, spans in table_after(r"^\| Baris \| Isi \|"):
    for s, sp in cell_nums(cells[1], spans[1]):
        if cells[0] == "B" and s == "20":
            record("48", s, N_SVD_COMPONENTS, "train_combined.N_SVD_COMPONENTS", "SVD", span=sp)
        else:
            record("48", s, s, "-", "rujukan Keputusan", ok=s in ("36", "47"), span=sp, note="ACUAN")
    for s, sp in cell_nums(cells[0], spans[0]):
        record("48", s, s, "-", "nama baris (C1/C2/C3/EF5)", ok=True, span=sp, note="ACUAN")
g = meta["c_optimized"]["grid"]
claim(r"`subsample` \{(0,7); (1,0)\} × `colsample_bytree` \{(0,3); (0,6); (1,0)\} × `min_child_weight` \{(1); (5)\} × "
      r"`reg_lambda` \{(1); (10)\} = (648) kombinasi",
      [*g["subsample"], *g["colsample_bytree"], *g["min_child_weight"], *g["reg_lambda"],
       meta["c_optimized"]["cv"]["n_combinations"]], AM,
      ["grid.subsample[0]", "grid.subsample[1]", "grid.colsample_bytree[0]", "[1]", "[2]", "grid.min_child_weight[0]",
       "[1]", "grid.reg_lambda[0]", "[1]", "c_optimized.cv.n_combinations"])
record("48", "648 (hitung)", len(XGB_GRID) * 2 * 3 * 2 * 2, "XGB_GRID x grid tambahan", "27*2*3*2*2",
       ok=len(XGB_GRID) * 24 == meta["c_optimized"]["cv"]["n_combinations"])
cvo = meta["c_optimized"]["cv"]
claim(r"walk-forward CV (5) fold di (937) hari train \+ valid", [cvo["n_folds"], cvo["n_rows"]], AM,
      ["c_optimized.cv.n_folds", "c_optimized.cv.n_rows"])
claim(r"Fit akhir di (772) hari train", [sp_["train"][2]], AM, ["split.train[2]"])
claim(r"CV macro-F1 terbaik (0,7843) ± (0,0308) \(grid seragam terbaik di CV: (0,7776)\)",
      [cvo["best_cv_macro_f1_mean"], cvo["best_cv_macro_f1_std"], cvo["best_uniform_grid_cv_macro_f1_mean"]], AM,
      ["c_optimized.cv.best_cv_macro_f1_mean", "best_cv_macro_f1_std", "best_uniform_grid_cv_macro_f1_mean"])
bp = meta["c_optimized"]["best_params"]
claim(r"`n_estimators` (100), `max_depth` (2), `learning_rate` (0,1), `colsample_bytree` (0,6), `subsample` (1,0), "
      r"`min_child_weight` (1), `reg_lambda` (1)\.",
      [bp["n_estimators"], bp["max_depth"], bp["learning_rate"], bp["colsample_bytree"], bp["subsample"],
       bp["min_child_weight"], bp["reg_lambda"]], AM, [f"c_optimized.best_params.{k}" for k in
                                                       ["n_estimators", "max_depth", "learning_rate",
                                                        "colsample_bytree", "subsample", "min_child_weight",
                                                        "reg_lambda"]])
claim(r"\(blok (10) hari, (2\.000) resample, seed (42)", [BOOTSTRAP_BLOCK, BOOTSTRAP_RESAMPLES, 42],
      "evaluate_late_fusion", ["BOOTSTRAP_BLOCK", "BOOTSTRAP_RESAMPLES", "RANDOM_STATE"])
cc = meta["test_class_counts"]
claim(r"### Hasil \(test (167) hari: low (51), medium (65), high (51)\)",
      [sp_["test"][2], cc["low"], cc["medium"], cc["high"]], AM,
      ["split.test[2]", "test_class_counts.low", "test_class_counts.medium", "test_class_counts.high"])

# --- tabel hasil ---
for cells, spans in table_after(r"^\| Baris \| Fitur \| Akurasi valid"):
    r = cells[0].replace("*", "")
    row = matrix.loc[r]

    def nums(i):
        return cell_nums(cells[i], spans[i])

    for s, sp in cell_nums(cells[0], spans[0]):
        record("48", s, r, M, "row (nama)", ok=True, span=sp, note="ACUAN nama baris")
    s, sp = nums(1)[0]
    record("48", s, row["n_features"], M, f"{r}.n_features", span=sp)
    s, sp = nums(2)[0]
    record("48", s, row["val_accuracy"], M, f"{r}.val_accuracy", span=sp)
    (a, spa), (d, spd) = nums(3)
    record("48", a, row["test_accuracy"], M, f"{r}.test_accuracy", span=spa)
    record("48", d, row["test_correct_days"], M, f"{r}.test_correct_days", span=spd)
    (lo, splo), (hi, sphi) = nums(4)
    record("48", lo, row["test_accuracy_ci_low"], M, f"{r}.test_accuracy_ci_low", span=splo)
    record("48", hi, row["test_accuracy_ci_high"], M, f"{r}.test_accuracy_ci_high", span=sphi)
    s, sp = nums(5)[0]
    record("48", s, row["test_macro_f1"], M, f"{r}.test_macro_f1", span=sp)
    if r == "naive":
        record("48", cells[6], "–", M, "naive vs naive", ok=cells[6] == "–")
    else:
        (x, spx), (y, spy), (p, spp) = nums(6)
        record("48", x, row["mcnemar_vs_naive_model_only"], M, f"{r}.mcnemar_vs_naive_model_only", span=spx)
        record("48", y, row["mcnemar_vs_naive_naive_only"], M, f"{r}.mcnemar_vs_naive_naive_only", span=spy)
        record("48", p, row["mcnemar_vs_naive_p"], M, f"{r}.mcnemar_vs_naive_p", span=spp)

# --- tabel pairwise ---
for cells, spans in table_after(r"^\| Perbandingan \| Δ akurasi"):
    key = cells[0]
    row = pair.loc[key]
    for s, sp in cell_nums(cells[0], spans[0]):
        record("48", s, key, PW, "comparison (nama)", ok=True, span=sp, note="ACUAN nama baris")
    cols = [("delta_accuracy_second_minus_first",), ("delta_accuracy_ci_low", "delta_accuracy_ci_high"),
            ("delta_macro_f1_second_minus_first",), ("delta_macro_f1_ci_low", "delta_macro_f1_ci_high"),
            ("mcnemar_second_only_correct", "mcnemar_first_only_correct"), ("mcnemar_p_value",)]
    for i, names in enumerate(cols, start=1):
        found = cell_nums(cells[i], spans[i])
        assert len(found) == len(names), (key, cells[i])
        for (s, sp), col in zip(found, names):
            record("48", s, row[col], PW, f"{key}.{col}", span=sp)

# --- interpretasi ---
claim(r"Model C memiliki akurasi test tertinggi \((0,772)\)", [matrix["test_accuracy"].max()], M,
      ["max(test_accuracy)"])
record("48", "C tertinggi", str(matrix["test_accuracy"].idxmax()), M, "idxmax(test_accuracy)",
       ok=matrix.loc["C", "test_accuracy"] == matrix["test_accuracy"].max())
claim(r"pada α = (0,05)", [0.05], PW, ["significant_alpha_0_05"])
record("48", "tidak ada yang signifikan", str(int(pair["significant_alpha_0_05"].sum())), PW,
       "sum(significant_alpha_0_05)", ok=not pair["significant_alpha_0_05"].any())
lows = list(pair["delta_accuracy_ci_low"]) + list(pair["delta_macro_f1_ci_low"])
highs = list(pair["delta_accuracy_ci_high"]) + list(pair["delta_macro_f1_ci_high"])
record("48", "Semua CI delta memuat nol", f"min high={min(highs):.4f}, max low={max(lows):.4f}", PW,
       "delta_*_ci_low/high", ok=max(lows) <= 0 <= min(highs))
claim(r"C hanya unggul (satu) hari atas B \((8) / (7) hari berbeda\)",
      [matrix.loc["C", "test_correct_days"] - matrix.loc["B", "test_correct_days"],
       pair.loc["B vs C", "mcnemar_second_only_correct"], pair.loc["B vs C", "mcnemar_first_only_correct"]],
      f"{M}; {PW}", ["test_correct_days C - B", "B vs C.mcnemar_second_only_correct",
                     "B vs C.mcnemar_first_only_correct"])
claim(r"Arah EF5_old → C \((\+4) hari\)",
      [matrix.loc["C", "test_correct_days"] - matrix.loc["EF5_old", "test_correct_days"]], M,
      ["test_correct_days C - EF5_old"])
same_c1_c3 = int((preds["C1"] != preds["C3"]).sum())
claim(r"\*\*C3 memberi prediksi yang sama persis dengan C1 di semua (167) hari\.\*\*", [len(preds)],
      rel(AB / "ablation_test_predictions.csv"), ["n baris"], note=f"jumlah tanggal C1 != C3: {same_c1_c3}")
record("48", "C3 = C1", same_c1_c3, rel(AB / "ablation_test_predictions.csv"), "sum(C1 != C3)", ok=same_c1_c3 == 0)
c3 = json.loads(matrix.loc["C3", "best_params"])
claim(r"Pada hyperparameter terpilih \((100) pohon, depth (2), lr (0,03)\)",
      [c3["n_estimators"], c3["max_depth"], c3["learning_rate"]], M, ["C3.best_params.n_estimators", "max_depth",
                                                                       "learning_rate"])
claim(r"\(importance blok NER = (0)\)", [matrix.loc["C3", "importance_ner"]], M, ["C3.importance_ner"])
CS = rel(AB / "c_vs_c_optimized_summary.json")
record("48", "proba tidak identik", str(copt_sum["proba_identical"]), CS, "proba_identical",
       ok=copt_sum["proba_identical"] is False)
claim(r"selisih absolut maksimum (0,274) \((2026)-(08)-(18)\), rata-rata (0,034), dan (25) dari (167) hari",
      [copt_sum["max_abs_diff"], 2026, 8, 18, copt_sum["mean_abs_diff"], copt_sum["n_days_max_abs_diff_gt_0_1"],
       copt_sum["n_test_days"]], CS,
      ["max_abs_diff", "max_abs_diff_date (tahun)", "(bulan)", "(hari)", "mean_abs_diff",
       "n_days_max_abs_diff_gt_0_1", "n_test_days"], note=f"max_abs_diff_date={copt_sum['max_abs_diff_date']}")
assert copt_sum["max_abs_diff_date"] == "2026-08-18"
claim(r"selisih > (0,1)\.", [0.1], CS, ["ambang n_days_max_abs_diff_gt_0_1"])
claim(r"Label berbeda di (9) hari: C saja benar (4) hari, C_optimized saja benar (4) hari, keduanya salah (1) hari",
      [copt_sum["n_days_label_differs"], copt_sum["label_differs_only_C_correct"],
       copt_sum["label_differs_only_C_optimized_correct"], copt_sum["label_differs_both_wrong"]], CS,
      ["n_days_label_differs", "label_differs_only_C_correct", "label_differs_only_C_optimized_correct",
       "label_differs_both_wrong"])
pd_diff = int((preds["C"] != preds["C_optimized"]).sum())
record("48", "9 (dari file prediksi)", pd_diff, rel(AB / "ablation_test_predictions.csv"), "sum(C != C_optimized)",
       ok=pd_diff == 9)
claim(r"menaikkan CV \((0,7843) vs (0,7776)\)", [cvo["best_cv_macro_f1_mean"], cvo["best_uniform_grid_cv_macro_f1_mean"]],
      AM, ["c_optimized.cv.best_cv_macro_f1_mean", "best_uniform_grid_cv_macro_f1_mean"])
record("48", "tidak menaikkan akurasi test", f"C {matrix.loc['C', 'test_accuracy']:.6f} / C_opt "
       f"{matrix.loc['C_optimized', 'test_accuracy']:.6f}", M, "test_accuracy",
       ok=matrix.loc["C_optimized", "test_accuracy"] <= matrix.loc["C", "test_accuracy"])
claim(r"blok embedding \((768) kolom\) mendapat (0,793) di C, sedangkan harga \((10) kolom\) (0,126)",
      [finbert_cols, matrix.loc["C", "importance_embedding"], len(PRICE_FEATURES), matrix.loc["C", "importance_price"]],
      M, ["finbert n_kolom", "C.importance_embedding", "len(PRICE_FEATURES)", "C.importance_price"])

# --- verifikasi ---
ver = meta["verification"]
claim(r"\*\*A = (0,731)\*\*: cocok dengan Keputusan (44)", [ver["A_test_accuracy"]["obtained"], 44], AM,
      ["verification.A_test_accuracy.obtained", "ACUAN"])
k44 = full_doc[full_doc.index("## 44. "):full_doc.index("## 45. ")]
claim(r"\*\*B = (0,766)\*\*, bukan (0,760) seperti di Keputusan (44) \(selisih (satu) hari; akurasi valid tetap (0,812)\)",
      [ver["B_test_accuracy"]["obtained"], ver["B_test_accuracy"]["expected_keputusan_44"], 44,
       round((ver["B_test_accuracy"]["obtained"] - ver["B_test_accuracy"]["expected_keputusan_44"]) * sp_["test"][2]),
       matrix.loc["B", "val_accuracy"]], AM,
      ["verification.B_test_accuracy.obtained", "verification.B_test_accuracy.expected_keputusan_44", "ACUAN",
       "(obtained - expected) x n_test", "ablation_matrix B.val_accuracy"])
record("48", "0,760 & 0,812 di Keputusan 44", "ada" if ("0,760" in k44 and "0,812" in k44) else "tidak",
       "docs/keputusan.md ## 44", "teks", ok="0,760" in k44 and "0,812" in k44)
vs = meta["versions"]
claim(r"\(pandas (3\.1\.1), numpy (2\.5\.5), scikit-learn (1\.9\.1), xgboost (3\.4\.1)\)",
      [vs["pandas"], vs["numpy"], vs["scikit-learn"], vs["xgboost"]], AM,
      ["versions.pandas", "versions.numpy", "versions.scikit-learn", "versions.xgboost"])
claim(r"dengan McNemar B vs A (9) / (3), p = (0,146)\.",
      [pair.loc["A vs B", "mcnemar_second_only_correct"], pair.loc["A vs B", "mcnemar_first_only_correct"],
       pair.loc["A vs B", "mcnemar_p_value"]], PW,
      ["A vs B.mcnemar_second_only_correct", "mcnemar_first_only_correct", "mcnemar_p_value"],
      note="angka dari ablation.py; klaim bahwa train_combined.py asli memberi angka yang sama TIDAK punya file "
           "output (dijalankan di sesi dengan output dialihkan ke folder sementara)")
RC = AB / "rerun_check"
for f in ("ablation_pairwise.csv", "ablation_test_predictions.csv", "ablation_c_optimized_cv.csv"):
    same = filecmp.cmp(AB / f, RC / f, shallow=False)
    record("48", f"{f} identik byte", str(same), f"{rel(AB / f)} vs {rel(RC / f)}", "cmp", ok=same)
m1 = pd.read_csv(AB / "ablation_matrix.csv").drop(columns="seconds")
m2 = pd.read_csv(RC / "ablation_matrix.csv").drop(columns="seconds")
record("48", "matrix beda hanya kolom waktu", str(m1.equals(m2)), "ablation_matrix.csv vs rerun_check", "drop seconds",
       ok=m1.equals(m2))
meta2 = json.loads((RC / "ablation_metadata.json").read_text(encoding="utf-8"))
def drop_time(d):
    d = json.loads(json.dumps(d))
    d.pop("seconds", None)
    d["c_optimized"]["cv"].pop("seconds", None)
    return d


same_meta = drop_time(meta) == drop_time(meta2)
record("48", "metadata beda hanya waktu", str(same_meta), "ablation_metadata.json vs rerun_check", "tanpa 'seconds'",
       ok=same_meta, note="c_optimized.cv.seconds juga waktu")
lk = ver["leakage"]
pp = lk["price_perturbation"]
claim(r"diuji di (4) titik potong \(Keputusan (37)\)", [len({c["t"] for c in pp}), 37], AM,
      ["len(unik verification.leakage.price_perturbation.t)", "ACUAN"])
n_vol = sum(c["perturb_realized_vol"] for c in pp)
claim(r"Pada (dua) titik di periode test", [n_vol], AM, ["sum(price_perturbation.perturb_realized_vol)"],
      note=f"titik dengan realized_vol diacak: {[c['t'] for c in pp if c['perturb_realized_vol']]}")
record("48", "semua perturbasi lulus", str(all(c["features_upto_t_unchanged"] for c in pp)), AM,
       "features_upto_t_unchanged", ok=all(c["features_upto_t_unchanged"] for c in pp))
claim(r"`lag(1)–(3)_log_return` sama dengan", [1, 3], AM, ["lag1_log_return_equals_shift", "lag3_..."])
record("48", "lag1-3 = shift", str([lk[f"lag{k}_log_return_equals_shift"] for k in (1, 2, 3)]), AM,
       "verification.leakage.lag{k}_log_return_equals_shift",
       ok=all(lk[f"lag{k}_log_return_equals_shift"] for k in (1, 2, 3)))
record("48", "TF-IDF/SVD identik", str(lk["tfidf_svd_fit_train_only"]), AM, "tfidf_svd_fit_train_only",
       ok=lk["tfidf_svd_fit_train_only"])
nw = lk["news_published_before_target_cutoff"]
claim(r"(0) dari (6\.013) artikel terbit setelah cutoff (08)\.(00) WIB",
      [nw["n_violations"], nw["n_articles"], nw["cutoff_hour_wib"], 0], AM,
      ["news_published_before_target_cutoff.n_violations", "n_articles", "cutoff_hour_wib", "menit"])
record("48", "target_date BERT = news_tfidf", str(lk["bert_path_target_date_equals_news_tfidf"]), AM,
       "bert_path_target_date_equals_news_tfidf", ok=lk["bert_path_target_date_equals_news_tfidf"])
record("48", "tidak ada kolom target di file teks", str(lk["text_feature_columns_with_target_terms"]), AM,
       "text_feature_columns_with_target_terms", ok=not any(lk["text_feature_columns_with_target_terms"].values()))
claim(r"upaya tuningnya lebih besar daripada A dan B \(Keputusan (41)\)", [41], "-", ["ACUAN"])
claim(r"Test kecil \((167) hari; satu hari = (0,6) poin akurasi\)", [sp_["test"][2], 100 / sp_["test"][2]], AM,
      ["split.test[2]", "100 / n_test"])
claim(r"diubah dua kali setelah CV terlihat \(Keputusan (47)\)", [47], "-", ["ACUAN"])

claim(r"hari tanpa fitur → (0)\)", [0.0 if "fillna(0.0)" in Path(ROOT / "src" / "evaluate_late_fusion.py").read_text()
                                  else -1], "src/evaluate_late_fusion.py", ["build_source_matrices: fillna(0.0)"])
ci95 = "CI 95%" in meta["statistics"]["bootstrap"]
claim(r"bootstrap CI (95)%", [95 if ci95 else -1], AM, ["statistics.bootstrap"],
      note="bootstrap_ci memakai np.percentile([2.5, 97.5])")
for m_ in re.finditer(r"\| CI (95)% (?:Δ )?(?:akurasi|macro-F1)", text):
    record(section_of(m_.start()), m_.group(1), meta["statistics"]["bootstrap"], AM, "statistics.bootstrap",
           ok=ci95, span=m_.span(1))
claim(r"`train_combined\.py` asli tanpa edit juga menghasilkan (0,766)", [ver["B_test_accuracy"]["obtained"]], AM,
      ["verification.B_test_accuracy.obtained"],
      note="nilai train_combined.py asli TIDAK tersimpan di file; dicocokkan dengan ablation.py")
claim(r"Jadi angka (0,760) di Keputusan (44)", [ver["B_test_accuracy"]["expected_keputusan_44"], 44], AM,
      ["verification.B_test_accuracy.expected_keputusan_44", "ACUAN"])
claim(r"untuk mereproduksi (0,760)\*\*", [ver["B_test_accuracy"]["expected_keputusan_44"]], AM,
      ["verification.B_test_accuracy.expected_keputusan_44"])

# ============================================================
# Cakupan: angka yang tidak dikonsumsi pemeriksaan apa pun
# ============================================================
covered = np.zeros(len(text), dtype=bool)
for a, b in consumed:
    covered[a:b] = True
uncovered = []
for m in re.finditer(r"\d+(?:[.,]\d+)*", text):
    if covered[m.start():m.end()].any():
        continue
    before = text[max(0, m.start() - 1)]
    ctx = text[max(0, m.start() - 40):m.end() + 25].replace("\n", " ")
    if before.isalpha() or before == "_":
        kind = "ACUAN identifier (mis. lag1, E2, fold5, C1)"
    elif re.search(r"(Keputusan|## )\s*$", text[max(0, m.start() - 11):m.start()]) or \
            re.search(r"Keputusan(\s\d+[,–]?\s?(dan\s)?)*\s?$", text[max(0, m.start() - 30):m.start()]):
        kind = "ACUAN nomor keputusan"
    else:
        kind = "TIDAK TERCAKUP"
    uncovered.append({"angka": m.group(), "jenis": kind, "konteks": ctx})
unc = pd.DataFrame(uncovered)

audit = pd.DataFrame(rows)
audit.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 500)
print(f"Pemeriksaan: {len(audit)}  cocok: {int(audit['cocok'].sum())}  TIDAK cocok: {int((~audit['cocok']).sum())}")
if (~audit["cocok"]).any():
    print(audit[~audit["cocok"]].to_string())
print(f"\nToken angka di Keputusan 46-48: {len(list(re.finditer(r'[0-9]+(?:[.,][0-9]+)*', text)))}; "
      f"tidak dikonsumsi pemeriksaan: {len(unc)}")
if len(unc):
    print(unc.groupby("jenis").size().to_string())
    print(unc[unc["jenis"] == "TIDAK TERCAKUP"].to_string())
print(f"\nTabel lengkap: {OUT}")
