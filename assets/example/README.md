# Demo (patient 22)

This folder contains a minimal ISLES 2017 subset for **training_22**, the manuscript example patient, plus precomputed HemoPIC fitting outputs. It is ready for **Step 2** (evaluation) out of the box, or **Step 1** can be rerun with `--fit`.

## Layout

```
ISLES2017_Training/
  training_22/
    MR_rCBF.nii.gz
    MR_rCBV.nii.gz
    MR_MTT.nii.gz
    CTC_from_MR_4DPWI.nii.gz
    OT.nii.gz
    hemopic_seg4/
      seg4_label_ref24.nii.gz
      seg4_gray_ref24.nii.gz

HemoPIC_outputs/
  training_22/
    HemoPIC_CBF.nii.gz
    HemoPIC_CBV.nii.gz
    HemoPIC_MTT.nii.gz
    roi_labels_seg4_within_kmeans.nii.gz
    roi_fit_summary.csv
    run_info_core.json
```

## Quick start

From the repository root:

```bash
python scripts/run_demo.py --eval          # Step 2: slice maps (z = 13, 14, 15)
python scripts/run_demo.py --fit           # Step 1: rerun fitting
python scripts/run_demo.py --fit --eval    # both steps
```

Evaluation figures are written to `assets/example/HemoPIC_eval/` by default.

## Notes

This demo skips **Preprocessing 1** (HemoPIC partition) and **Preprocessing 2** (PWI → tracer). For the full pipeline on all 48 patients, download [ISLES 2017 Training](https://zenodo.org/records/17736412) and follow the main README.