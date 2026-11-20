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

## Amendment: unseen-dataset test on DeepFakeDetection (written 20 November 2026, before any DFD score)

**Why.** After `final_cdf` trained on Celeb-DF, no test set was unseen by the served model. FaceForensics++'s
DeepFakeDetection set (DFD, c23) is: 363 real videos of 28 paid actors in scripted scenes and 3,068
deepfakes between pairs of them. None of its actors or videos are in FF++'s YouTube videos or in Celeb-DF.

**Data.** `manifests/dfd_c23.csv` (`scripts/build_manifest_dfd.py`): every DFD video, split `test`; five
`tmp*` leftovers of an interrupted download are skipped. Face crops with the unchanged
`scripts/extract_faces.py` settings (32 frames, YuNet track, 1.3x crops); a video without a usable
face track is excluded, as for every other dataset. DFD is used for nothing but this test.

**Models.** `final_cdf` (served), `final` (FF++ only), and the frozen-feature rows `C` and `F` (3 seeds
each), all exactly as already trained, each with its own frozen validation threshold.

**Reported.** ROC AUC (primary), balanced accuracy, fake recall, specificity, mean P(fake), with 95 %
CIs from resampling the 28 actors. Scored once; all results are reported, including unfavourable ones.
No threshold, calibration or model choice is changed because of DFD results.

*Note added after face extraction, before any DFD score:* 13 DFD fakes (short `secret_conversation` clips)
give fewer than the 32 face crops every model reads; they are left out for all models, like videos
without a face track (43). DFD test set: 3,375 videos.

## Amendment: calibration of the served model (written 20 November 2026, before fitting)

**Why.** The served model `final_cdf` outputs a mean window P(fake) that is not a probability, and its
single threshold (0.816, chosen on pooled FF++ + Celeb-DF validation videos, which are 80 % fake) called
31 % of real FF++ test videos fake. That test result is already known; the change below is therefore
motivated by it, but every choice is made on validation data only.

**Calibration.** Platt scaling, p = sigmoid(a * logit(score) + b), fitted on `final_cdf`'s validation
scores (FF++ val 840 videos + Celeb-DF v2 val 377 videos), with sample weights that give each of the four
cells (dataset x real/fake) the same total weight. The calibrated value therefore reads as "probability
of fake if real and fake were equally likely, averaged over the two datasets". Its quality is estimated
on validation by 5-fold cross-fitting over families (ECE with 10 bins, Brier, NLL, same weights).

**Decision rule.** Fake if the calibrated probability is at least 0.5. It replaces the 0.816 threshold
only if it gives a higher mean of the two datasets' validation balanced accuracies; otherwise the
threshold stays and only the calibrated probability is shown.

**Test report.** The fitted map is applied once to the stored test scores of `final_cdf` (FF++ test,
Celeb-DF v1/v2 test lists, DFD); nothing is refitted afterwards. Reported: ECE, Brier and NLL with the
same class-balanced weights, and specificity, fake recall and balanced accuracy at the adopted rule.
AUC does not change: the map is monotone.

## Amendment: whole-video reading (written 20 November 2026, before any segment was scored)

**Why.** The served model reads one 32-frame segment from the middle of a video (about 8.5 s at stride 8),
so a manipulated passage elsewhere is missed.

**Segments** (`deepshield/segments.py`). The evaluated middle segment is repeated back to back over the
whole video (starts at middle_start + k x 32 x stride), plus a segment flush with the first and with the
last frame when they would otherwise leave at least half a segment unread; at most 40 segments, evenly
spaced, always including the middle one. Each segment gets its own face track and crops with the
training pipeline; a segment with a face in fewer than half of its frames is not scored.

**Rules for one video score.** R1 `middle`: the middle segment's score (the evaluated reading; if it has
no face, the nearest segment that has one). R2 `mean`: the mean over scored segments. R3 `max`: the
highest segment score. All use the served model `final_cdf` unchanged.

**Choice, on validation only** (FF++ val + Celeb-DF v2 val): R2 or R3 replaces R1 for the verdict only if
its mean validation AUC over the two datasets is higher than R1's; between R2 and R3 the higher wins.
R1 keeps its threshold (0.816); an adopted R2 or R3 gets the threshold that maximises the mean of the two
datasets' validation balanced accuracies. Whatever the choice, the app shows every segment's score.

**Test report.** Scored once on the FF++ test split, the Celeb-DF v2 test list and DFD (the longest
videos, median 25 s), for R1 and the adopted rule: AUC, balanced accuracy, fake recall, specificity.
Check: R1 must reproduce the stored test scores of `final_cdf` wherever the middle segment has a face.
