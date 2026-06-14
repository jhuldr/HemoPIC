from pathlib import Path
from typing import Optional

HEMOPIC_SEG4_DIRNAME = "hemopic_seg4"
SEG4_LABEL_FILENAME = "seg4_label_ref24.nii.gz"
SEG4_GRAY_FILENAME = "seg4_gray_ref24.nii.gz"


def hemopic_map_stem(tag: str) -> str:
    return "HemoPIC_" + str(tag).upper()


def hemopic_map_path(fit_dir: Path, tag: str) -> Path:
    return Path(fit_dir) / (hemopic_map_stem(tag) + ".nii.gz")


def pick_hemopic_map(fit_dir: Path, tag: str) -> Optional[Path]:
    fit_dir = Path(fit_dir)
    tag_u = str(tag).upper()
    cand = [
        fit_dir / ("HemoPIC_" + tag_u + ".nii.gz"),
        fit_dir / ("HemoPIC_" + tag_u + ".nii"),
        fit_dir / ("Model_" + tag_u + "_raw_clamped.nii.gz"),
        fit_dir / ("Model_" + tag_u + "_raw.nii.gz"),
        fit_dir / ("Model_" + tag_u + "_scaled.nii.gz"),
        fit_dir / ("Model_" + tag_u + "_raw_clamped.nii"),
        fit_dir / ("Model_" + tag_u + "_raw.nii"),
        fit_dir / ("Model_" + tag_u + "_scaled.nii"),
    ]
    for p in cand:
        if p.exists():
            return p
    found = sorted(fit_dir.glob("*" + tag_u + "*.nii*"))
    found = [q for q in found if q.is_file()]
    if found:
        return found[0]
    return None


def require_hemopic_map(fit_dir: Path, tag: str) -> Path:
    p = pick_hemopic_map(fit_dir, tag)
    if p is None:
        raise FileNotFoundError("missing map HemoPIC_" + str(tag).upper() + " in " + str(fit_dir))
    return p


def training_subject_id(patient_num: int) -> str:
    return "training_" + str(int(patient_num))


def patient_num_from_dirname(name: str) -> Optional[int]:
    if name.startswith("training_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    if name.startswith("p"):
        digits = []
        for ch in name[1:]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            return int("".join(digits))
    return None


def list_fit_patient_ids(fit_root: Path) -> list[int]:
    fit_root = Path(fit_root)
    out = []
    for d in sorted(fit_root.iterdir()):
        if not d.is_dir():
            continue
        pid = patient_num_from_dirname(d.name)
        if pid is not None:
            out.append(int(pid))
    return sorted(set(out))


def resolve_fit_dir(fit_root: Path, patient_num: int) -> Optional[Path]:
    fit_root = Path(fit_root)
    pid = int(patient_num)
    subj = training_subject_id(pid)
    legacy_p = "p" + str(pid)
    cand = [
        fit_root / subj,
        fit_root / legacy_p / "global" / "core",
        fit_root / legacy_p / "global",
        fit_root / (legacy_p + "_kmeans") / "global" / "core",
        fit_root / (legacy_p + "_kmeans") / "global",
    ]
    for d in cand:
        if d.is_dir():
            return d
    return None


def require_fit_dir(fit_root: Path, patient_num: int) -> Path:
    fit_dir = resolve_fit_dir(fit_root, patient_num)
    if fit_dir is None:
        raise FileNotFoundError("missing fit dir for patient " + str(int(patient_num)))
    return fit_dir


def hemopic_seg4_dir(patient_dir: Path) -> Path:
    return Path(patient_dir) / HEMOPIC_SEG4_DIRNAME


def hemopic_seg4_label_path(patient_dir: Path) -> Path:
    return hemopic_seg4_dir(patient_dir) / SEG4_LABEL_FILENAME


def hemopic_seg4_gray_path(patient_dir: Path) -> Path:
    return hemopic_seg4_dir(patient_dir) / SEG4_GRAY_FILENAME
