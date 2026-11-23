# Research plan: modality-aware feature upsampling for frozen EO encoders

We build a feature upsampler for frozen Earth-observation (EO) encoders. It takes the encoder's coarse feature map and the input image, and returns one feature vector per image pixel. It is trained once without labels and stays frozen while task heads are trained.

The work has three parts. Each ends in a gate: a list of conditions that must hold before the next part starts. Part 1 runs checks before any upsampler is trained (gate G0). Part 2 builds the upsampler for TerraMind (gates G1, G2). Part 3 tests one upsampler across several encoders (gate G3). This document is the single, self-contained plan for the research.

## How to use this plan

This plan is written for the person or agent who carries out the research in a new repository.

1. Work through the steps in order. Steps that do not depend on each other may run in parallel.
2. At each gate (G0–G3), write a gate report: the results the gate's conditions refer to, the decisions taken, and anything unexpected. Then stop and wait for the team's approval before starting the next part.
3. Never score a test split before gate G2 (Part 2) or G3 (Part 3).
4. Values marked "proposed" are defaults. Use them unless the team changes them, and record every change in the decision log.
5. If a step cannot be done as written (missing data, code that does not run, a result that contradicts an assumption), record it in the decision log, take the closest option that keeps the step's purpose, and report it at the next gate.
6. Keep the records listed under "Repository and records".

## Why the plan is shaped this way

Four risks shaped the plan. Each is tested by a step before effort is spent on it.

- **Training targets.** AnyUp is trained by comparing its output with the encoder's features of magnified crops. TerraMind was pretrained on 224–256 px inputs and saw S2 only at 10–20 m ([TerraMind/THOR comparison](https://arxiv.org/pdf/2607.18504)), so those features may not be valid targets. Steps 1.5 and 1.6 test this, and Part 2 compares two training objectives.
- **What averaging can recover.** AnyUp-type upsamplers output weighted averages of nearby patch features. With a linear head they cannot recover detail that no nearby patch encodes. Step 1.8 measures this limit.
- **Room for improvement.** Some tasks are already close to their best score at patch resolution: on Sen1Floods11, frozen TerraMind-tiny with UPerNet reaches 0.897 mIoU ([same source](https://arxiv.org/pdf/2607.18504)). Step 1.7 measures each task's resolution-limited error.
- **Source of any gain.** A guided upsampler also gives the head access to full-resolution pixels. Steps 1.11 and 1.12 test whether simpler ways of giving that access do as well.

## Terms used in this plan

| Term | Meaning |
| --- | --- |
| Encoder | A pretrained EO model that turns an image into features. Always frozen. Parts 1 and 2 use TerraMind-tiny (TerraMind v1) |
| Patch feature map | The encoder output. TerraMind cuts the image into 16×16-pixel patches and outputs one feature vector per patch: a 224×224 input gives a 14×14 map |
| Guidance image | The full-resolution image the upsampler uses to place edges: S2 RGB for released upsamplers, S1 and S2 bands for ours |
| Upsampler | A network that takes the patch feature map and the guidance image and outputs one feature vector per pixel (224×224 here) |
| AnyUp | Published upsampler (Wimmer et al., ICLR 2026). Each output vector is a weighted average of the patch vectors in a window around the pixel; attention on the guidance image and patch features sets the weights |
| E0 | AnyUp with its authors' published weights, used unchanged (step 1.10) |
| NAF | Published upsampler (Chambon et al., CVPR 2026) whose weights depend only on the guidance image |
| Bilinear upsampling | Bilinear interpolation of the patch feature map to pixel resolution; the main baseline |
| Task head | The only part trained on labels during evaluation. A linear head is a 1×1 convolution from each pixel's feature vector to class scores or a regression value |
| Modality subset | The inputs a sample has: S1 only, S2 only, or S1+S2. The encoder and the upsampler always get the same subset |
| Gain | Score of a method minus the score of the method it is compared with, computed on the same chips |

## Shared setup

| Task | PANGAEA dataset | Type | Inputs | Role |
| --- | --- | --- | --- | --- |
| Flood mapping | Sen1Floods11 | Segmentation | S1, S2 | Primary candidate |
| Field boundaries | AI4SmallFarms | Segmentation | S2 | Primary candidate |
| Marine pollution | MADOS | Segmentation | S2 | Primary candidate |
| Burn scars | HLS Burn Scars | Segmentation | HLS at 30 m | Secondary: different resolution |
| Biomass | BioMassters, single date | Regression | S1, S2 | Regression task |
| Crop types | PASTIS-R, CropTypeMapping-SS | Segmentation, time series | S1, S2 | Step 2.13 only |

Primary tasks are fixed at gate G0 using step 1.7.

**Head training.** PANGAEA splits and settings: Adam, learning rate 1e-4, batch size 8, 80 epochs, learning rate divided by 10 at 60% and 90% of training. Every method in a comparison gets the same head architecture and settings. Inputs stay at native resolution in 224×224 windows, with PANGAEA's sliding-window inference.

**Metrics.** Computed for every run and saved as CSV.

- Segmentation: mIoU; per-class precision, recall, IoU and F-score; unnormalized confusion matrix; IoU of the task's main class (water, field boundary, burn scar); boundary F1 at 1 and 2 px tolerance.
- Regression: RMSE, MAE, bias, and error by target quantile.
- Cost: maximum memory, training time, inference time, GFLOPs, and GPUs used (type and count).

**Statistics.** One head seed for debugging, three for any reported number. To compare two methods, resample the chips with replacement 1,000 times and recompute both scores on each resample. Report the mean gain and its 2.5–97.5 percentile range (the 95% interval); a method beats another when that interval lies above zero.

**Validation and test.** Every choice is made on validation splits. Test splits are scored once per frozen model, at G2 and G3.

## Baseline upsamplers from the literature

Every method below is compared with our upsampler under the same protocol: frozen encoder, frozen upsampler, a task head trained per task. Methods tied to one encoder are retrained for TerraMind with their own code; the others are used as released.

| Method | How it upsamples | Tied to one encoder | Code and weights | Used in |
| --- | --- | --- | --- | --- |
| Bilinear interpolation | Fixed interpolation | No | Standard | Every step |
| Joint bilateral upsampling (Kopf et al., 2007) | Fixed filter weighted by spatial distance and guidance-image similarity | No | Implement | Step 1.11 |
| Guided filter (He et al., 2013) | Fixed local linear model of the guidance image | No | Implement | Step 1.11 |
| [Upsample Anything](https://arxiv.org/pdf/2511.16301) (2025) | Fits per-pixel Gaussian kernels to each image with 50 Adam steps; needs no training set | No | Paper lists a project page but no code link | Step 1.13 |
| [AnyUp](https://arxiv.org/html/2510.12764v2) (ICLR 2026) | Window attention; output is a weighted average of patch features | No | [Code](https://github.com/wimmerth/anyup) with paper and multi-encoder weights | Step 1.10 (E0); basis of our model |
| [NAF](https://openaccess.thecvf.com/content/CVPR2026/html/Chambon_NAF_Zero-Shot_Feature_Upsampling_via_Neighborhood_Attention_Filtering_CVPR_2026_paper.html) (CVPR 2026) | Neighbourhood attention with weights computed from the guidance image only | No | [Code](https://github.com/valeoai/NAF), Apache-2.0, weights and training code | Step 1.13; architecture of step 3.4 |
| [FeatUp](https://arxiv.org/pdf/2403.10516), JBU variant (ICLR 2024) | Stack of learned joint bilateral upsamplers, trained by multi-view reconstruction | Yes | [Code](https://github.com/mhamilton723/FeatUp) | Step 2.7 |
| [JAFAR](https://arxiv.org/pdf/2506.11136) (NeurIPS 2025) | Global attention, trained on multi-resolution views at low upsampling factors | Yes | [Code](https://github.com/PaulCouairon/JAFAR) | Steps 2.7, 3.6 |
| [UPLiFT](https://arxiv.org/pdf/2601.17950) (CVPR 2026) | Iterative upsampling with local attention | Yes; weights released for DINOv2-S/14 and DINOv3-S+/16 | [Code](https://github.com/mwalmer-umd/UPLiFT), MIT, training code | Step 2.7 |
| [LoftUp](https://arxiv.org/html/2504.14032v1) (2025) | Coordinate-based cross-attention, trained with SAM masks, then self-distillation | Yes | [Code](https://github.com/andrehuang/loftup) | Optional in step 2.7 |

All methods except ours take an RGB guidance image. They get the S2 RGB image of step 1.9 and cannot run on S1-only inputs. LoftUp is optional because SAM masks computed on 10 m S2 RGB are of uncertain quality.

LiFT is not run: UPLiFT revisits its iterative design. Newer methods (ViT-Up, PixelUp, DiveUp, FeatSharp) join step 2.7 only if their training code is public when Part 2 starts.

## At a glance

&#91;embedded content: three parts · gates G0 to G3\]

Part 1 is where you start: it needs only the PANGAEA pipeline and released checkpoints, and its results decide how Part 2 is trained.

## Part 1: checks before training any upsampler

Part 1 trains task heads only, never an upsampler. It sets up evaluation, measures how much released upsamplers help, and answers four questions that decide how Part 2 is trained.

### 1.1 Evaluation pipeline

Add TerraMind-tiny and TerraMind-base to PANGAEA as frozen encoders that return the last layer's patch feature map. Set each task's inputs as in the Shared setup table, keep native resolution, and log every image PANGAEA resizes. Record which S1 product each dataset uses (GRD or RTC).

**Check.** TerraMind-base with PANGAEA's UPerNet decoder should land near the published 66.6% average mIoU over six PANGAEA segmentation tasks. If it does not, fix the pipeline before continuing.

### 1.2 Splits and overlap with training data

Fix PANGAEA's splits and all random seeds. Record each chip's geographic footprint and list every TerraMesh tile that intersects a footprint or lies within 5 km of one; Part 2 never trains on these tiles. Note datasets whose chips are not georeferenced.

### 1.3 Metrics code

Implement the metrics and the bootstrap from Shared setup, writing CSV files. Validate them on the outputs of step 1.4.

### 1.4 Bilinear baseline

Bilinear upsampling plus a linear head, on every task, three seeds. Every later method is compared with this. Because the head is linear, applying it to upsampled features equals applying it to each patch and upsampling the scores; steps 1.8 and 1.11 use the head this way.

### 1.5 Does TerraMind describe a magnified crop consistently?

**Why.** Training objective A (Part 2) compares the upsampler's output for a crop with TerraMind's features of that crop magnified 2×. That target only makes sense if TerraMind describes a magnified crop consistently with the original image. TerraMind was pretrained at a fixed resolution, so this may fail.

**Procedure.**

1. Take 500 TerraMesh validation tiles and their central 224×224 window x. Run once with S2 only and once with S1 only.
2. Compute F = TerraMind(x), a 14×14 map.
3. Take the central 112×112 crop of x, resize it to 224×224 with bilinear interpolation, and compute G = TerraMind(resized crop). G is 14×14 but covers only the central 7×7 patches of x.
4. Average G over 2×2 blocks to get 7×7, and compute the cosine similarity with the central 7×7 vectors of F at each position.
5. Reference low value: cosine similarity between F vectors at random positions in different tiles.
6. Reference high value: repeat steps 1–5 with DINOv2 ViT-S/14 on 500 ImageNet validation images (14 px patches, so 8×8 instead of 7×7). AnyUp's objective is known to work for this encoder.

**Result and decision.** For each encoder, normalize the mean similarity as (similarity − low value) / (1 − low value). If TerraMind scores more than 0.2 below DINOv2 (proposed threshold), Part 2 starts with objective B; otherwise with objective A. Step 2.5 compares both either way.

### 1.6 Do TerraMind features change under shifts smaller than one patch?

**Why.** Training objective B (Part 2) uses features of the image shifted by 1–15 pixels as targets. If such shifts barely change the features, these targets carry no information beyond the unshifted map.

**Procedure.**

1. From each of 500 TerraMesh validation tiles (264×264), take 224×224 windows shifted horizontally by d = 0, 2, 4, …, 16 pixels. Run with S2 only and S1 only.
2. Compute the 14×14 map of each window.
3. For each d, compute the mean cosine similarity s(d) between patch (i, j) of the unshifted window and patch (i, j) of the shifted one. s(16) compares neighbouring patches.

**Result and decision.** Plot s(d). Compute r = (1 − s(8)) / (1 − s(16)): the change from a half-patch shift relative to a full-patch shift. If r < 0.1 (proposed threshold), sub-patch shifts barely change the features and objective B is ranked below objective A.

### 1.7 How much can better upsampling improve each task at most?

**Why.** If a task already scores close to perfect at patch resolution, no upsampler can improve it much.

**Procedure,** on validation labels:

1. Reduce each label map to the patch grid: per 16×16 patch, the fraction of pixels of each class (segmentation) or the mean value (regression).
2. Upsample these fractions or means bilinearly to full resolution. For segmentation, assign each pixel the class with the largest fraction.
3. Score this map against the original labels with the task metrics.

**Result and decision.** The score is what a perfect patch-level prediction reaches after bilinear upsampling. Its distance to a perfect score (IoU 100, RMSE 0) is the error caused by coarse resolution alone, the most a better upsampler can recover. Rank tasks by this distance; tasks below 3 IoU points (proposed) become secondary. The minimum gain required in Part 2 is set per task at G0 as a fraction of this distance.

### 1.8 Best score an averaging upsampler can reach with a linear head

**Why.** AnyUp, JAFAR and NAF output at each pixel a weighted average of the patch vectors in a window around it. With a linear head, the pixel's scores are the same weighted average of those patches' scores. A pixel can only be classified correctly if a nearby patch already carries the right answer.

**Procedure,** on validation:

1. Apply the step 1.4 head to every patch to get patch-level class scores or values.
2. For each pixel, take the k×k patches centred on its own patch; k is the attention window size, in patches, of the AnyUp configuration used.
3. Segmentation: mark the pixel reachable if at least one of these patches gives the true class the highest score. This is exact for two-class tasks and can slightly underestimate for more classes.
4. Build a map in which reachable pixels get their true class and the others keep the bilinear baseline's prediction; score it.
5. Regression: clip each pixel's true value to the range \[min, max\] of the window patches' predictions and score the clipped map.

**Result and decision.** The best score any averaging upsampler can reach with this head. If it is close to the bilinear baseline on a task, such upsamplers cannot help there with a linear head; that task is then judged on the small convolutional head (step 1.12).

### 1.9 RGB guidance image

Released upsamplers expect an 8-bit RGB image normalized with ImageNet mean and standard deviation. Build it from S2 bands B4, B3, B2 (for HLS, its red, green and blue bands) with the conversion TerraMind uses for its S2 RGB input. If that conversion is unavailable, clip reflectance to \[0, 0.3\] and scale linearly to \[0, 255\]. Use one conversion for all tasks and inspect a few images per task.

### 1.10 E0: released AnyUp with S2 RGB guidance

**Procedure.**

1. Load both published AnyUp weight files: the paper's (trained on DINOv2 ViT-S features) and the multi-backbone one. No training or fine-tuning.
2. Inputs: TerraMind-tiny's 14×14 patch feature map and the step 1.9 RGB image; output a 224×224 feature map. Use AnyUp's q\_chunk\_size option if memory runs out.
3. Train a linear head on every task, three seeds.
4. Control: repeat with the guidance image of a randomly chosen other chip from the same dataset. If scores match the correct-guidance run, the gain does not come from the image content.

**Result and decision.** Gain over bilinear per task, with 95% intervals and cost metrics. If E0 beats bilinear on at least one task: E0's scores become the minimum our upsampler must reach, and step 2.5 adds a run initialized from the published AnyUp weights.

### 1.11 Classical edge-aware baselines

Apply the step 1.4 head to every patch and upsample the scores bilinearly. Then refine them with (a) a guided filter (He et al., 2013) and (b) joint bilateral upsampling (Kopf et al., 2007). Both use S2 RGB as guidance, or S1 VV and VH when only S1 is present. Tune the guided filter's radius and regularization and the bilateral filter's two kernel widths on validation. This shows whether a learned upsampler beats classical, non-learned edge-preserving methods.

### 1.12 Raw-band concatenation baseline

Concatenate the bilinearly upsampled features with the raw input bands at each pixel. Train a linear head and a small convolutional head: two 3×3 convolutions with 64 channels and ReLU, then a 1×1 output layer. If this matches our upsampler, the upsampler's gain comes only from letting the head see full-resolution pixels. The same small convolutional head is used in steps 1.8 and 2.12.

### 1.13 Released NAF and Upsample Anything

(a) NAF: as step 1.10, with NAF's published weights. (b) Upsample Anything: for each image, fit the method's per-pixel kernels to the S2 RGB image (50 Adam steps at learning rate 1e-3, as in the paper), then apply them to the patch features. Use the authors' code if released; otherwise implement it from the paper and label it as our reimplementation in all results. The paper reports sensitivity to noisy inputs, so it runs with S2 RGB guidance only.

### 1.14 Standard decoders

Frozen TerraMind-tiny with PANGAEA's UPerNet decoder, and PANGAEA's UNet trained from scratch on the raw bands. They show where a full decoder and a fully supervised model land.

### Gate G0, on validation

Record these decisions before Part 2 starts:

1. Which training objective runs first (steps 1.5, 1.6).
2. Primary and secondary tasks (step 1.7).
3. Minimum gain per task for the Part 2 conditions (step 1.7).
4. Whether step 2.5 adds a run initialized from published AnyUp weights (step 1.10).
5. S1 product, GRD or RTC, per dataset (step 1.1).
6. Per task, the best validation score among E0, NAF, Upsample Anything, the two classical filters and raw-band concatenation. Runs in steps 2.4–2.6 are judged against it.

## Part 2: feature upsampler for TerraMind

Part 2 trains one upsampler for frozen TerraMind-tiny on unlabeled TerraMesh tiles. It chooses the training objective and input layers on validation, freezes the upsampler, and scores it once on test.

### Model

- **Inputs.** TerraMind-tiny's last-layer 14×14 patch feature map for a 224×224 window (S1 and S2 features merged by mean, TerraTorch's default), and the guidance image: S1 (VV, VH) and/or S2 bands at 224×224.
- **Input layers.** One for S1, one for S2. Each is a small stride-1 convolutional block with the same number of output channels as AnyUp's RGB input layer. With both modalities present, their outputs are averaged per pixel, leaving out cloudy or missing pixels.
- **Rest of the network.** AnyUp unchanged: guidance-image encoder, the layer that maps any feature dimension to a fixed size, positional encoding, window attention. Output: a 224×224 map with TerraMind's feature dimension, each vector a weighted average of patch vectors.
- **Initialization and settings.** Random initialization, except run C in step 2.5. Optimizer, learning rate and schedule from AnyUp's released training configuration; change batch size only if memory requires.

### Training objectives

Both use only unlabeled TerraMesh images; TerraMind's own features are the targets.

**Objective A, crop reconstruction (AnyUp's objective, adapted).**

1. Take a 224×224 window x from a tile and compute F = TerraMind(x).
2. Pick a random 112×112 crop of x whose corner lies on an 8-pixel grid, resize it to 224×224, and compute the target T = TerraMind(resized crop), a 14×14 map.
3. Run the upsampler on (F, x) at 28×28 output size, so the crop region spans 14×14 output positions; cut out that region.
4. Loss: 1 − cosine similarity plus mean squared error between the region and T, averaged over positions, plus AnyUp's consistency terms as in its code.

AnyUp itself downsamples the whole image and keeps the crop at native resolution. TerraMesh tiles are only 264 px and TerraMind was pretrained on 224–256 px inputs, so we keep x at native resolution and magnify the crop instead.

**Objective B, shifted-window reconstruction.**

1. From a 264×264 tile take the 224×224 window x at offset (0, 0) and compute F = TerraMind(x).
2. Pick two random offsets (dx, dy), each between 0 and 15 pixels, not both 0. Compute T = TerraMind(window at that offset).
3. Run the upsampler on (F, x) at 224×224. For each patch of the shifted window that lies fully inside x, average the output vectors over that patch's 16×16 pixels.
4. Loss: 1 − cosine similarity plus mean squared error between these averages and the matching vectors of T, plus AnyUp's input-consistency term (the same comparison at offset (0, 0)).

### 2.1 Training data

- About 50,000 TerraMesh training tiles, minus the tiles listed in step 1.2. S1 product as fixed at G0; S1 GRD exists only in TerraMesh's SSL4EO-S12 shards, S1 RTC in all.
- Balance land cover: give each tile the most frequent class of TerraMesh's land-cover layer and sample about equal numbers per class, water included.
- Clouds: use TerraMesh's SEnSeIv2 cloud masks. If more than 20% (proposed) of a tile's S2 pixels are cloud or shadow, use the tile as S1 only; otherwise mark those pixels invalid in the guidance image and the loss.
- Dates: if a tile's S1 and S2 dates differ by more than 7 days, never use them together.
- Output: a manifest with tile, modalities, cloud fraction, date difference and land cover.

### 2.2 Model code

Implement the model above. Unit tests: output shape, every modality subset, invalid pixels excluded from the averaging.

### 2.3 Objective code

Implement objectives A and B. Train each on 64 tiles until the loss stops falling, to confirm the model can fit them. Check that output vectors have norms similar to TerraMind's patch vectors.

### 2.4 Short run

One seed, a short schedule, the objective chosen at G0. Train linear heads on validation of the two primary tasks with the largest step 1.7 distance and compare with bilinear and E0, to catch errors early.

### 2.5 Choose objective and initialization

Full runs with the same number of iterations:

- Run A: objective A, random initialization.
- Run B: objective B, random initialization.
- Run C, only if G0 says so: the objective chosen at G0, initialized from the published AnyUp weights. The S2 input layer copies AnyUp's RGB input weights for B4, B3, B2 and is zero for other bands; the S1 input layer is random.

For each training sample, pick one modality subset uniformly from those the tile allows; TerraMind and the upsampler get the same subset. Evaluate each run with linear heads on validation of the primary tasks, every subset. Keep the run with the highest mean score; if runs fall within each other's 95% intervals, keep A, then B, then C.

### 2.6 Input-layer variants

With the objective and initialization from step 2.5, train:

- V1: separate S1 and S2 input layers, outputs averaged (the model above).
- V2: one input layer on all S1 and S2 bands stacked, a missing modality filled with zeros, plus two channels marking which modalities are present.
- V3: one input layer on S2 RGB only, evaluated only on subsets with S2.

Keep the simplest variant (V3, then V2, then V1) whose validation scores lie within the best variant's 95% interval.

### Gate G1, on validation

The chosen model beats bilinear and both classical filters of step 1.11 on every primary task. If it does not, train one upsampler on S1-only samples and one on S2-only samples. If these beat the joint model, training on both modalities hurts each, and step 2.5 is revisited.

### 2.7 Literature upsamplers trained for TerraMind

Train FeatUp (JBU variant), JAFAR and UPLiFT for frozen TerraMind-tiny with each method's own training code and default settings, on the step 2.1 tiles. Their image encoders take RGB, so the guidance is the step 1.9 S2 RGB image; TerraMind gets S2 or S1+S2 inputs, chosen at random per tile. Cap each method's training compute (GPU hours) at that of our final model. LoftUp is optional. Freeze all of them before step 2.8; they are evaluated only on subsets that include S2.

### 2.8 Freeze

Save the chosen weights with a checksum. Nothing changes after this point.

### 2.9 Segmentation on test

For every segmentation task and every subset its data allows (S1, S2 and S1+S2 for Sen1Floods11; S2 for the others), three seeds, linear heads: our upsampler, bilinear, the two classical filters, raw-band concatenation (linear and small convolutional heads), E0 with both weight files, NAF, Upsample Anything, and the step 2.7 upsamplers (subsets with S2 only). Add the UPerNet and UNet references. Report HLS Burn Scars separately, since it is at 30 m.

### 2.10 Regression on test

BioMassters single date, every subset, the same methods. RMSE, MAE, bias and error by target quantile.

### 2.11 Controls on test

(a) Guidance image from a random other chip of the same dataset. (b) Sen1Floods11 results split by the days between S1 and S2 acquisition, if the dates are available.

### 2.12 Comparison with full decoders

(a) Our upsampler with the small convolutional head vs UPerNet on frozen TerraMind-tiny. (b) UPerNet with our upsampled features concatenated to its last full-resolution feature map before the classifier. Report scores and cost metrics.

### Gate G2, on test

All five conditions must hold:

1. On at least two primary segmentation tasks and on BioMassters, our upsampler with a linear head beats bilinear and raw-band concatenation (linear head); the 95% interval of the gain lies above the minimum gain fixed at G0.
2. No task and subset has a gain over bilinear whose 95% interval lies entirely below zero.
3. On the S2-only subset of each primary task, it is not worse than the best of E0, NAF, Upsample Anything and the step 2.7 upsamplers: no 95% interval of the gain lies entirely below zero.
4. On at least one subset that includes S1, it beats every method named in condition 3.
5. Our upsampler with the small convolutional head comes within a margin fixed at G0 of UPerNet at lower memory or inference time, or step 2.12(b) improves UPerNet.

### 2.13 Confirmation, after G2

(a) Repeat step 2.9 with 10% of training labels, using PANGAEA's stratified subset. (b) PASTIS-R and CropTypeMapping-SS: apply the upsampler to every time step, then PANGAEA's L-TAE temporal aggregation.

## Part 3: one frozen upsampler for several EO encoders

Part 3 starts after G2. It tests whether one frozen upsampler improves other frozen EO encoders with no retraining and no added parameters, and whether using encoder features to compute the attention weights matters.

### 3.1 Encoders

From PANGAEA's supported models, pick one encoder to train with TerraMind and at least two held-out encoders from other model families. Proposed: CROMA for training (S1 and S2 inputs); DOFA and Prithvi held out. Fix the list before any Part 3 training.

### 3.2 Reference scores per encoder

For each encoder, on the primary tasks: bilinear with a linear head, E0, NAF, Upsample Anything, and UPerNet. Record its input preprocessing, patch size, feature dimension and mean feature-vector norm.

### 3.3 Direct transfer

Apply the frozen Part 2 upsampler to each encoder's patch feature map unchanged; AnyUp's input layer accepts any feature dimension. Train linear heads and compare with bilinear, E0 and NAF. If it beats all three on every held-out encoder, skip step 3.5.

### 3.4 Image-only guidance model

Train an upsampler with NAF's architecture and our S1 and S2 input layers, on the same data, with the same objective and number of iterations as Part 2. Its weights depend only on the guidance image, so it works with any encoder by construction. Our upsampler must beat it on held-out encoders; otherwise, using encoder features in the attention adds nothing.

### 3.5 Two-encoder training, only if step 3.3 falls short

Train on TerraMind and CROMA with AnyUp's multi-encoder training code. Each batch uses one encoder, chosen at random, with its own preprocessing and its own features as targets. Average the loss over feature elements so encoders with more channels do not dominate, and log each encoder's mean feature norm.

### 3.6 Held-out test

Freeze the upsampler. Score DOFA and Prithvi with no new parameters and no tuning, against bilinear, E0, NAF, Upsample Anything, the step 3.4 model and the Part 2 upsampler. Also train JAFAR for DOFA alone, as in step 2.7, to show how close the shared upsampler comes to one trained for that encoder.

### 3.7 Image resolution

HLS Burn Scars (30 m) is never used for training. Compare (a) training with several upsampling factors and with TerraMesh tiles resampled to 20 m and 30 m, and (b) the same plus FiLM conditioning: a small network that maps ground sampling distance and upsampling factor to a per-channel scale and shift of the guidance features. Keep (b) only if it improves HLS Burn Scars.

### 3.8 Other sensors, optional

xView2, Five Billion Pixels, DynamicEarthNet and SpaceNet 7 use non-Sentinel imagery. Run them only after adding an input layer for those sensors, and report them outside the gate conditions.

### Gate G3, on test

One frozen upsampler, with no new parameters or tuning for held-out encoders, beats bilinear, NAF and the step 3.4 model on the primary tasks for every held-out encoder, and is not worse than bilinear on HLS Burn Scars. Claims stay limited to the encoders and sensors tested.

## Repository and records

Suggested layout for the new repository:

```
configs/           one YAML file per run
src/upsampler/     our model and training objectives A and B
src/baselines/     classical filters, raw-band concatenation, wrappers for literature upsamplers
src/eval/          PANGAEA integration, task heads, metrics, bootstrap
scripts/           one entry point per plan step, e.g. step_1_5_magnification.py
results/           CSV files
docs/decisions.md  decision log
docs/gates/        one report per gate
```

- **Every run:** config file, git commit, random seed, checkpoint checksum, start and end time, GPU type and count.
- **Results CSV:** one row per run, task, modality subset, seed and metric, with columns run\_id, step, method, encoder, task, split, modality\_subset, seed, metric, value. Keep per-chip scores too; the bootstrap needs them.
- **Decision log:** date, step, decision, reason, and who approved it.
- **Gate reports:** one per gate, in docs/gates/.

## Inputs needed from the team

Ask for these before or during Part 1:

1. Compute: GPU type, count and hours available. This sets the training schedules in steps 2.5–2.7.
2. Storage for the TerraMesh subset and the PANGAEA datasets.
3. Approval of every value marked "proposed", at G0 at the latest.
4. The minimum gain per task and the UPerNet margin used at G2, fixed at G0.
5. Approval at each gate before the next part starts.

## Resources

| Resource | Links | Used for |
| --- | --- | --- |
| PANGAEA benchmark | [Paper](https://arxiv.org/html/2412.04204v2), [code](https://github.com/VMarsocci/pangaea-bench) | Tasks, splits, head training, UPerNet and UNet |
| TerraMind v1 in TerraTorch | [Guide](https://torchgeo.org/terratorch/1.2.1/guide/terramind/) | Encoders terramind\_v1\_tiny and terramind\_v1\_base, modality merging |
| TerraMind-1.0-base weights | [Hugging Face](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base) | Reproduction check, step 1.1 |
| TerraMesh | [Hugging Face](https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh) | Unlabeled training tiles, cloud masks, land cover |
| TerraMind and THOR comparison | [Paper](https://arxiv.org/pdf/2607.18504) | TerraMind pretraining range; Sen1Floods11 reference scores |
| AnyUp | [Paper](https://arxiv.org/html/2510.12764v2), [code](https://github.com/wimmerth/anyup) | E0, basis of our model, multi-encoder training |
| NAF | [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Chambon_NAF_Zero-Shot_Feature_Upsampling_via_Neighborhood_Attention_Filtering_CVPR_2026_paper.html), [code](https://github.com/valeoai/NAF) | Baseline; architecture of step 3.4 |
| FeatUp | [Paper](https://arxiv.org/pdf/2403.10516), [code](https://github.com/mhamilton723/FeatUp) | Step 2.7 |
| JAFAR | [Paper](https://arxiv.org/pdf/2506.11136), [code](https://github.com/PaulCouairon/JAFAR) | Steps 2.7 and 3.6 |
| UPLiFT | [Paper](https://arxiv.org/pdf/2601.17950), [code](https://github.com/mwalmer-umd/UPLiFT) | Step 2.7 |
| LoftUp | [Paper](https://arxiv.org/html/2504.14032v1), [code](https://github.com/andrehuang/loftup) | Optional, step 2.7 |
| Upsample Anything | [Paper](https://arxiv.org/pdf/2511.16301) | Step 1.13 |
