# HemoPIC

This repository contains a complete pipeline for running HemoPIC and evaluation scripts for cerebral perfusion processing.

## Execution order

Step 1 Run the pipeline for HemoPIC to generate fitting outputs  
Step 2 Run evaluation scripts in the Eval folder

## Data

This repository uses the [ISLES 2017](https://www.kaggle.com/datasets/ahmedaabdullah/isles-2017) Training set. Provide the path to the `ISLES2017_Training` folder as `DATASET_ROOT` in all commands. 

## Preprocessing 1 (SynthSeg and HemoPIC Partition)

This step runs SynthSeg on ISLES 2017 data and converts labels to the HemoPIC's partition on the perfusion grid.
 
Set up FreeSurfer environment

```bash
export FREESURFER_HOME=<path to your freesurfer install>
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FS_LICENSE=<path to your license.txt>

python /path/to/run_synthseg.py \
  /path/to/ISLES2017_Training \
  /path/to/output_synthseg
```

Run SynthSeg 

```bash
python /path/to/run_synthseg.py \
  /path/to/ISLES2017_Training \
  /path/to/output_synthseg \
  training_36
```

Convert SynthSeg labels to HemoPIC segmentation

```bash
python /path/to/hemopic_seg4.py \
  /path/to/ISLES2017_Training \
  /path/to/output_synthseg \
  /path/to/output_hemopic_seg4
```

## Preprocessing 2 (PWI to Tracer Concentration)

This step converts the raw ISLES 2017 Perfusion-Weighted Imaging signal into a tracer time series used by downstream scripts.

Run for all patients

```bash
python /path/to/pwi_to_tracer.py \
  /path/to/ISLES2017_Training \
  1 48
```

## Step 1 HemoPIC Pipeline

```bash
DATASET_ROOT=/path/to/ISLES2017_Training
OUT_ROOT=/path/to/HemoPIC_outputs
PATIENT_NUM=<PATIENT_NUM>
K_GM=8
K_WM=6
SEED=0

python scripts/run_hemopic.py "$DATASET_ROOT" "$OUT_ROOT" "$PATIENT_NUM" "$K_GM" "$K_WM" "$SEED"
```

Notes:  
- OUT_ROOT is the output folder produced by HemoPIC  
- Use this output folder as FIT_ROOT in the evaluation commands below

## Step 2 Evaluation

Shared arguments

```bash
DATASET_ROOT=/path/to/ISLES2017_Training
FIT_ROOT=/path/to/HemoPIC_outputs
OUT_DIR=/path/to/HemoPIC_eval
```

### Central Volume Theorem Verification

```bash
python Eval/cvt_check/run_eval_cvt.py "$DATASET_ROOT" "$FIT_ROOT" "$OUT_DIR"
```

### Slice Maps for Visualization

```bash
python Eval/summary_maps/run_eval_slice_maps.py \
  "$DATASET_ROOT" \
  "$FIT_ROOT" \
  "$OUT_DIR" \
  <patient_id> <slice_z> <cbf_max> <cbv_max> <mtt_max>
```

Notes:  
- Manuscript example used patient_id 22 with slice_z 13 14 15 and scales cbf_max 70 cbv_max 7 mtt_max 15

### Summary Statistics

```bash
python Eval/summary_maps/run_eval_summary_stats.py "$DATASET_ROOT" "$FIT_ROOT" "$OUT_DIR"
```

### Tracer Reconstruction Results

```bash
python Eval/tracer_recon/run_eval_tracer_estimation.py "$DATASET_ROOT" "$FIT_ROOT" "$OUT_DIR"
```

### Windkessel Parameters Report

```bash
python3 Eval/Windkessel/run_eval_tau_report.py "$DATASET_ROOT" "$FIT_ROOT" "$OUT_DIR"
```


## Link to Manuscript
- **`Eval/summary_maps/run_eval_summary_stats.py`** generates patient cohort summary statistics for lesion and gray/white matter
- **`Eval/summary_maps/run_eval_slice_maps.py`** generates qualitative slice map figures for selected patients and slices
- **`Eval/tracer_recon/run_eval_tracer_estimation.py`** generates tracer reconstruction evaluation reports and plots
- **`Eval/cvt_check/run_eval_cvt.py`** generates central volume theorem consistency summaries and plots  
- **`Eval/Windkessel/run_eval_tau_report.py`** generates Windkessel parameter reports


## Copyright
"HemoPIC: A Physics-Informed Cerebral Hemodynamics Digital Twin for Brain Perfusion" is a publication of The Johns Hopkins University and copyright © 2026 The Johns Hopkins University. All rights reserved.
 


