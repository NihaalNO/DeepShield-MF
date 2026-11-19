# Final evaluation protocol (Phase 8)

Written on 17 November 2026, **before** any model was scored on the FaceForensics++ test split or on Celeb-DF.
Everything below was decided on validation evidence only. The evaluation is run once with
`scripts/final_eval.py`, which refuses to overwrite existing results.

## Models

| Name | Run | Why it is included |
| --- | --- | --- |
| `final` (primary) | `results/ffpp_c23/e2e_finetune_aug-frame_final/seed0` | Best validation result of the project |
| `C` | `results/ffpp_c23/C_faces_T8/seed{0,1,2}` | Simplest reliable baseline (best frozen-feature model) |
| `F` | `results/ffpp_c23/F_faces_T8/seed{0,1,2}` | The original full hybrid model of the project |

**Choice of the final model.** On the validation split (120 training families, one seed each), fine-tuning
ResNet50 `layer4` and the last two ViT blocks gave 0.774 to 0.796 video AUC against 0.670 for the same model
with frozen backbones and 0.754 for row C. Per-frame augmentation had the highest validation AUC (0.796) and
the highest AUC under every degradation tested, so it was chosen over sequence-consistent augmentation (0.781)
and no augmentation (0.774).

**Final training run.** Same script and settings as the 120-family run (`scripts/finetune_e2e.py --mode finetune
--aug frame`), but on all 360 training families, seed 0, up to 8 epochs with early stopping patience 3. The
best epoch and the decision threshold are chosen on the validation split, as for every other run. The run is
used whatever its validation result, so that the choice cannot depend on the test data.

No other model is scored. In particular, the head-level variants of Phase 7 (I, J, J0, K, L, M1 to M3), the
other ablation rows and the leave-one-method-out runs are not evaluated on test data.

## Data

| Key | Data | Videos used |
| --- | --- | --- |
| `ffpp_test` | FaceForensics++ c23, official test split | 839 (one video has no usable face track) |
| `celebdf_v1` | Celeb-DF v1, `List_of_testing_videos.txt` | 100 (62 fake, 38 real) |
| `celebdf_v2` | Celeb-DF v2, `List_of_testing_videos.txt` | 516 of 518 (339 fake, 177 real; 2 without face track) |

Face crops and features were prepared with the same pipeline as for training (YuNet, 32 frames, 224x224).
No Celeb-DF video has been used for training, selection, thresholds or calibration.

## Scoring

- Video score: mean P(fake) over 7 windows of 8 frames (50 % overlap).
- Threshold: the one each model chose on the FF++ validation split, unchanged.
- Reported: ROC AUC (threshold-free, primary), balanced accuracy, fake recall and specificity at the frozen
  threshold, mean P(fake), per-method AUC on FF++, 95 % CIs from resampling source families
  (FF++) or target identities (Celeb-DF).
- C and F: mean ± std over 3 seeds. `final`: one model, with its bootstrap CI.

All results are reported, including unfavourable ones. Cross-dataset thresholds are expected to transfer
badly; that is reported, not corrected.

## Amendment: second-domain model (written 19 November 2026, before it was trained)

**Why.** Served on uploads, the final model called most real Celeb-DF interviews fake (on the Celeb-DF v2
test list it put 45 % of the real videos above its threshold, and 60 % of 25 Celeb-real videos outside
the list). The serving code reproduces the evaluation exactly, so the cause is the training data: its
only real videos are 720 FaceForensics++ news clips.

**Model `final_cdf`.** The same recipe as `final` (`scripts/finetune_e2e.py --mode finetune --aug frame`,
all 360 FF++ training families, seed 0, up to 8 epochs, patience 3), plus `--extra-dataset
celebdf_v2_train`: Celeb-DF v2 videos outside the v1 and v2 official test lists
(`manifests/celebdf_v2_train.csv`), split by identity (6 validation celebrities, no identity shared
between train and validation), all reals kept, fakes subsampled to 1,500 (train) and 300 (validation).
Model selection and the threshold use the combined validation set (FF++ val + Celeb-DF val).

**Which model the app serves** is decided on validation only, before any test score: `final_cdf`
replaces `final` if its Celeb-DF validation specificity at its own threshold is higher than `final`'s
on the same videos, and its FF++ validation AUC is at most 0.02 below `final`'s (0.867).

**Evaluation.** Scored once, like `final`, on the FF++ test split and both Celeb-DF test lists.
After this fine-tuning the Celeb-DF scores are **same-dataset** results: the test videos are unseen, but
54 of the 59 test celebrities also appear in training. No unseen-dataset score exists for `final_cdf`;
that would need a third dataset. The earlier results of `final`, `C` and `F` stay as they are.
