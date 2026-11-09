# Version 2 plan: পরিষ্কার 4-class fundus classification (Cataract / DR / Glaucoma / Normal)

**ফাইলগুলো কোথায়:**

| Folder | কী আছে |
|---|---|
| `resolved/Version 1/` | আপনার `class-7-correction.ipynb` আর তার ফলাফল `eye7_results.zip`। এখানে কোনো পরিবর্তন করা হয়নি। |
| `resolved/Version 2/class-4-v2.ipynb` | নতুন notebook। Version 1-এর একই ১৬টা Part, শুধু নিচে লেখা জায়গাগুলোয় বদল। |
| `resolved/Version 2/plan.md` | এই plan |

Dataset: **শুধু একটা**, `manan1717/ocular-disease-dataset`। অন্য কোনো dataset ব্যবহার হয় না।

---

## ০. সংক্ষেপে

1. **Version 1-এর ফলাফল report করার মতো ছিল না।**
   - ১৮টা training run-এর মাত্র ৬টা শেষ হয়েছিল (শুধু fold 0)। Report নিজেই লিখেছে: *"INTERIM, do not report"*।
   - কারণ: ৬টা model × ৩ fold-এ ~৭.৩ ঘণ্টা লাগত, অথচ time budget ছিল ২.৬৫ ঘণ্টা।
2. **Version 1-এ যেসব ভুল আর গোলমাল পেয়েছি:**
   - **DR-কে Normal বলা:** ১,০৩৮টা DR-এর ১৯৬টা (১৮.৯%), বেশিরভাগ ODIR-এর মৃদু DR।
   - **Glaucoma আর Myopia একে অপরকে:** ১৮ + ১৬টা ভুল।
   - **একই ছবি দুই class-এ:** Glaucoma আর Myopia folder-এ একই ছবির ১৪৫টা group। একটা AMD "ODIR-copy" ছবি আসলে একটা ODIR DR ছবি।
   - **খারাপ ছবিও training-এ ঢুকেছে:** পুরো ঝাপসা DR ছবি, বড় কালো দাগওয়ালা ছবি, অতিরিক্ত উজ্জ্বল ছবি (`fig_sources`)।
3. **সিদ্ধান্ত:**
   - **৪টা class রাখা হলো: C, D, G, N।** আপনার paper-এর ৪টা class-ই এগুলো।
   - **A (AMD), H (Hypertensive), M (Myopia) বাদ।** কারণ Part ২-এ।
   - তবে duplicate খোঁজার সময় **৭টা folder-ই স্ক্যান করা হয়।** ফলে একই ছবি Glaucoma আর Myopia, দুই folder-এ থাকলে ধরা পড়ে, আর দুটো কপিই বাদ যায়।
4. **নতুন cleaning (training-এর আগে):**
   - একই ছবি দুই class-এ থাকলে বাদ
   - Fundus না এমন ছবি বাদ
   - **Ungradable ছবি বাদ** (নতুন): খুব অন্ধকার, খুব উজ্জ্বল বা ঝাপসা
   - **প্রায়-একই কপির মধ্যে শুধু একটা রাখা** (নতুন)
   - বাদ পড়া ছবি train, validation বা test কোথাও থাকে না।
5. **Model:**
   - দুর্বল ৩টা বাদ: MobileNetV4-M (৭৫.৩%), ConvNeXt-T DINOv3 (৮২.১%), optic-disc ConvNeXt-S (৭৫.৮%)।
   - রইল: ResNet50, ConvNeXt-T, DINOv2 ViT-B/14।
   - **নতুন transformer: DINOv3 ViT-B/16 (2025)।** DINOv2-এর হালনাগাদ version।
6. **সময়:**
   - Version 1-এর মাপা epoch-সময় দিয়ে হিসাব করে ঠিক করা, যাতে **সব fold শেষ হয়।**
   - Default `fast` preset: ~৩.৫–৪ ঘণ্টা। `lite`: ~২.৭ ঘণ্টা। `full`: ~৬.৫ ঘণ্টা।
7. **Figure:** সব 600 dpi PNG আর vector PDF-এ। শেষে একটা zip-এর download link।

---

## ১. Version 1 বিশ্লেষণ: কী কাজ করেছে, কী করেনি

### ১.১ যা ভালো ছিল (Version 2-এ রাখা হয়েছে)

- 512 px + JPEG harmonisation। এতে উৎস চেনার ছাপ মুছে যায়: আলাদা করার accuracy মাত্র ৫৩.৩%, ৫০% মানে একদম চেনা যায় না।
- সব উৎসে retina একই মাপে আনা (round FOV crop) আর CLAHE।
- ODIR রোগীর দুই চোখ সবসময় একই fold-এ। প্রায়-একই ছবিও একই fold-এ।
- Stratified group K-fold, যেখানে strata = class × উৎস।
- GPU-তে augmentation, EMA, validation NLL দেখে early stopping, flip TTA।
- Stacking ensemble আর temperature scaling।
- Statistics: McNemar/Holm, corrected t-test, Friedman, cluster bootstrap CI।
- Shortcut baseline, প্রতি উৎসের আলাদা result, source-shift test, মাপা Grad-CAM।

### ১.২ যা সমস্যা ছিল

| সমস্যা | প্রমাণ (Version 1 run) | Version 2-এ সমাধান |
|---|---|---|
| Run শেষ হয়নি | ৬/১৮ run, "INTERIM" | Model কমানো, ConvNeXt-T 224 px-এ, preset আর budget মাপা সময় অনুযায়ী |
| ধীর model | Epoch প্রতি ConvNeXt-T 208 s, MobileNetV4 173 s, disc model 158 s; ResNet50 আর DINOv2 ~80 s | MobileNetV4 আর disc model বাদ; T4-এ depthwise convolution ধীর, তাই ConvNeXt-T 224 px-এ |
| দুর্বল model | MobileNetV4-M ৭৫.৩%, ConvNeXt-T (DINOv3) ৮২.১%, disc ৭৫.৮% | বাদ |
| Glaucoma↔Myopia গোলমাল | ১৪৫টা duplicate group, ১৫৮টা Myopia ছবি বাদ পড়েছিল; CM-এ G→M ১৮, M→G ১৬ | Myopia বাদ, conflict সব folder জুড়ে খোঁজা |
| AMD উৎস দেখেই চেনা যায় | AMD-এর ১০০% এমন set থেকে যেখানে আর কোনো class নেই; ৫৪% ODIR ছবির কপি; ODIR camera-য় F1 মাত্র 0.62 | AMD বাদ |
| খারাপ মানের ছবি | `fig_sources`-এ ঝাপসা Mendeley DR, কালো দাগওয়ালা ODIR Glaucoma, অতিরিক্ত উজ্জ্বল Myopia | Ungradable filter |
| Duplicate শুধু group করা | Cataract-এর ৩৯৬টা ছবি duplicate cluster-এ (একই চোখের augmented কপি) | প্রতি cluster-এ একটা রাখা |
| Normal-এর ১০০% ODIR থেকে | Shortcut: শুধু উৎস জেনেই macro-F1 50.8%, metadata দিয়ে 62.8% | এক dataset-এ এটা এড়ানো যায় না, তাই মেপে report করা হয়: shortcut baseline, ODIR-only result আর source-shift test (Part ৩-এর শেষ অংশ) |
| মূল ভুল DR→Normal | ODIR-এর ৫৪২টা DR-এর ১৯৯টা Normal বলা হয়েছে | পরিষ্কার data, DINOv3; per-source table-এ আলাদা করে report |

---

## ২. Class নির্বাচন: কেন C, D, G, N

| Class | মূল তথ্য (Version 1 audit) | সিদ্ধান্ত |
|---|---|---|
| **A** AMD | সব ছবি এমন set থেকে (ARMD: RFMiD/JSIEC/ARIA) যেখানে অন্য কোনো class নেই। ৫৪% ODIR ছবির re-save। DR আর Glaucoma-র সাথে ৮টা conflict। ৯৭% sensitivity আসতে পারে শুধু উৎস চিনে। | **বাদ** |
| **H** Hypertensive | ৭০% এমন set থেকে যেখানে শুধু H আছে। ODIR-এর ভেতরে F1 মাত্র 0.08–0.12। | **বাদ** (Version 1-এও বাদ ছিল) |
| **M** Myopia | Glaucoma-র সাথে ১৪৫টা duplicate group, dataset-এর সবচেয়ে বড় label-noise। Myopic disc দেখতে glaucoma-র মতো। Paper-এর ৪ class-এর বাইরে। | **বাদ** |
| **C** Cataract | চোখে আলাদা করা সহজ। ODIR camera-র ভেতরেও F1 0.85। | রাখা |
| **D** DR | মূল class, ODIR + Mendeley। | রাখা |
| **G** Glaucoma | মূল class, ODIR + Mendeley। | রাখা |
| **N** Normal | মূল class, সব ছবি ODIR থেকে। | রাখা |

বাদ পড়া class-এর ছবি **শুধু audit-এ** ব্যবহার হয়: কোনো task-ছবি ওদের কপি কিনা দেখার জন্য। Training, validation বা test-এ ওরা কখনো থাকে না। তাই ওদের সমস্যা আপনার research-এ প্রভাব ফেলে না। CELL-এ `classes`-এ `"M"` যোগ করলে 5-class-ও চালানো যায়।

---

## ৩. Data cleaning: কোন ছবি বাদ, কোনগুলো থাকে

সব নিয়ম training-এর আগে প্রয়োগ হয়, শুধু ছবি, ফাইলের নাম আর label দেখে। Model-এর prediction দেখে কিছু বাদ দেওয়া হয় না।

| নিয়ম | কী করে | Part |
|---|---|---|
| Label conflict | MD5, pHash (flip/rotation-সহ), deep-feature (cos ≥ 0.99) বা একই ODIR চোখ দুই **folder**-এ থাকলে (A/H/M সহ) পুরো cluster বাদ | 4 |
| Non-fundus | ছবির কোণা কালো না হলে, অর্থাৎ চোখের বাইরের ছবি | 4 |
| **Ungradable (নতুন)** | (ক) Retina-র গড় উজ্জ্বলতা < 20 বা > 235। (খ) ঝাপসা: সব non-cataract ছবির তুলনায় Laplacian variance-এর robust z < −3, আর নিজের উৎস × class group-এর তুলনায় z < −3.5। Cataract প্রথম নিয়ম থেকে ছাড় পায়, কারণ ঝাপসা দেখানোই cataract-এর লক্ষণ। | 3, 4 |
| **Duplicate কপি (নতুন)** | প্রতি প্রায়-একই cluster থেকে একটাই ছবি থাকে (সবচেয়ে বড় original) | 4 |
| রোগী grouping | একই ODIR রোগীর দুই চোখ সবসময় একই fold-এ | 4–5 |

### আনুমানিক সংখ্যা

Version 1-এর audit থেকে হিসাব (ungradable filter-এর আগে):

| Class | মোট | Conflict বাদ | Duplicate কপি বাদ | **থাকবে** |
|---|---|---|---|---|
| Cataract | 1,008 | 15 | 212 | **781** |
| DR | 3,128 | 30 | 63 | **3,035** |
| Glaucoma | 1,646 | 165 | 69 | **1,412** |
| Normal | 2,997 | 4 | 3 | **2,990** |
| **মোট** | 8,779 | 214 | 347 | **8,218** |

- Ungradable filter আরও আনুমানিক ১–৩% বাদ দেবে। আসল সংখ্যা notebook print করবে।
- বাদ পড়া ছবিগুলো `fig_removed_ungradable`-এ দেখা যাবে। ভালো ছবি দেখলে CELL 0-তে `blur_z_global=-3.5` করুন।
- কোনো উৎস × class-এর ১৫%-এর বেশি বাদ পড়লে notebook WARNING দেবে।

### একটাই dataset: উৎস-shortcut কীভাবে সামলানো হয়

- শুধু `manan1717` ব্যবহার হয়। Notebook অন্য কোনো dataset খোঁজে না, attach করা থাকলেও না।
- এই dataset-এ Normal-এর সব ছবি ODIR থেকে। Mendeley থেকে আসে শুধু DR আর Glaucoma। ফলে model ছবির উৎস দেখে কিছুটা আন্দাজ করতে পারে।
- এটা লুকানো হয় না, তিনভাবে মাপা হয়:
  - **Harmonisation:** সব ছবি একই 512 px + JPEG-এ আনা হয়, যাতে ফাইল বা resolution দেখে উৎস চেনা না যায়।
  - **Shortcut baseline** (Part 12): retina না দেখে, শুধু ফাইলের তথ্য দিয়ে কতটা ঠিক হয়।
  - **ODIR-only result** (Part 12): একই camera, চারটা class-ই আছে, তাই এখানে উৎস দিয়ে কিছু বোঝা যায় না। Paper-এর সবচেয়ে সৎ সংখ্যা এটাই।
  - **Source-shift test** (Part 14): ODIR-এর কোনো রোগের ছবি training-এ থাকে না। এতে মাপা যায় model রোগ চেনে নাকি উৎস চেনে।
- Paper-এ এই তিনটা সংখ্যা মূল result-এর পাশে দিতে হবে। এটাই Reviewer #4-এর "external/multi-center" প্রশ্নের উত্তর।

---

## ৪. Model

| Model | Version 1 (fold 0, 6 class) | Epoch-সময় (T4 x2) | Version 2 |
|---|---|---|---|
| ResNet50 | 83.62% | 78 s @256 | রাখা (CNN baseline, দ্রুত) |
| ConvNeXt-T (IN-1k) | **85.73%** (সেরা single model) | 208 s @256 | রাখা, **224 px-এ** (~1.3 গুণ দ্রুত) |
| DINOv2 ViT-B/14 | 84.71% (ECE সবচেয়ে কম, 1.99%) | 81 s @224 | রাখা |
| **DINOv3 ViT-B/16** | নতুন | আনুমানিক ~70–80 s @224 | **নতুন transformer** |
| MobileNetV4-M | 75.25% | 173–201 s | বাদ |
| ConvNeXt-T (DINOv3) | 82.07% | 208 s | বাদ |
| ConvNeXt-S [disc] | 75.78% | 158 s | বাদ |

**কেন DINOv3 ViT-B/16:**
- DINOv2 (Version 1-এর দ্বিতীয় সেরা) আর ODIR-এর ভেতরে সেরা probe ছিল। DINOv3 তার ২০২৫ সালের উত্তরসূরি।
- ১.৭ বিলিয়ন ছবিতে self-supervised training, RoPE position encoding, ৪টা register token।
- Version 1-এ DINOv3-এর ConvNeXt রূপটা খারাপ করেছিল, কিন্তু ওটা ছোট distilled CNN। DINOv3-এর মূল ক্ষমতা ViT রূপে।
- `timm`-এ নাম `vit_base_patch16_dinov3.lvd1689m`। এখানে model বানিয়ে যাচাই করেছি: ১২টা block, 224/256 px-এ চলে।
- Weight download কোনো কারণে ব্যর্থ হলে Part 8 নিজে থেকে DINOv2-reg ViT-B/14-এ চলে যায় আর log-এ লিখে দেয়। Paper-এ তখন সেই নামটাই লিখবেন।

**Proposed method:**
- ৪টা model-এর stacking (logistic regression, validation-এ fit), সাথে temperature scaling। Version 1-এর DA-MVE থেকে disc view বাদ।
- Ensemble এখন আর paper-এর মূল novelty না (Part ৭)।

---

## ৫. সময় (Kaggle GPU T4 x2)

Version 1-এর মাপা সময় (৫,৬২৭টা training ছবি, ৮ epoch) থেকে Version 2-এর ছবির সংখ্যা অনুযায়ী হিসাব। এগুলো **অনুমান, মাপা নয়।**

| Preset | Fold | Model | আনুমানিক মোট সময় | কখন |
|---|---|---|---|---|
| `fast` (default) | 3 | 4টা | ~৩.৫–৪ ঘণ্টা | প্রথম run, আর paper-এর জন্যও যথেষ্ট |
| `lite` | 3 | 3টা (ConvNeXt বাদ) | ~২.৫–২.৭ ঘণ্টা | সময় খুব কম থাকলে |
| `full` | 5 | 4টা | ~৬.৫ ঘণ্টা | Camera-ready (5-fold CV) |

- **Budget guard:** `time_budget_h` (fast 4.6 h) পার হবে মনে হলে notebook নতুন run শুরু করে না। তাই report সবসময় তৈরি হয়।
- প্রতিটা fold শেষে notebook আনুমানিক শেষ সময় print করে।
- কোনো run বাদ পড়লে report-এ "INTERIM" লেখা থাকবে। তখন output attach করে আবার চালালে যেখানে থেমেছিল সেখান থেকে চলবে।

---

## ৬. Reviewer-এর আপত্তি → Version 2

| আপত্তি | সমাধান | কোথায় |
|---|---|---|
| R1: contribution বোঝা যায় না, technical content কম | Data-centric contribution: (১) multi-source dataset-এর audit: conflict, duplicate, patient leakage, উৎস-shortcut, ungradable; (২) leakage-মুক্ত, উৎস-সচেতন 4-class benchmark; (৩) source-shift test; (৪) supervised CNN বনাম self-supervised ViT (DINOv2 → DINOv3) | Part 4, 12, 14 |
| R1: figure-এর মান | 600 dpi PNG আর vector PDF, IEEE column width | সব figure |
| R4: novelty কম (soft voting) | Ensemble মূল দাবি না। Stacking + calibration একটা অংশ মাত্র। | Part 11 |
| R4: repeated CV | Grouped stratified 3-fold (fast) বা 5-fold (full), mean ± SD | Part 5, 12 |
| R4: statistics | Cluster bootstrap 95% CI, McNemar + Holm, corrected t-test, Friedman | Part 12 |
| R4: external multi-center validation | Source-shift test: ODIR-এর কোনো রোগের ছবি training-এ নেই, test হয় ODIR রোগীদের ছবিতে (চীন বনাম বাংলাদেশ)। প্রতি উৎসের আলাদা result। আরেকটা dataset থাকলে Part 6-এ। | Part 12, 14 |
| R4: patient-level validation | ODIR-এ রোগী ধরে split (~৩,০০০ রোগী)। Random split হলে test ছবির ৪০% leak করত, সেটাও report হয়। | Part 4–5 |
| R4: Grad-CAM শুধু ছবি দিয়ে | Deletion/insertion AUC, drop/increase, optic-disc pointing game, random baseline-এর সাথে তুলনা | Part 15 |
| R4: Normal→Glaucoma ভুল | আলাদা table। Version 1-এ ৯৯৬টা Normal-এর মাত্র ৩৩টা (৩.৩%), পুরনো paper-এ ৫৫/২২৬ (২৪%)। | Part 12 |
| R4: clinical claim | Referral curve, referable sensitivity/specificity। Paper-এ claim নরম করে লেখা। | Part 13 |

---

## ৭. Result কেমন আশা করবেন (সৎ হিসাব)

- **ভিত্তি:** Version 1-এর fold 0-এর prediction শুধু ৪ class-এ সীমিত করলে (২,৮৫৫টা test ছবি):
  - সব মিলিয়ে: **accuracy 86.48%, macro-F1 88.63%**
  - Mendeley অংশে: 97.5%
  - ODIR অংশে: 79.1%
- **Version 2-এ যা সাহায্য করবে:**
  - ৪ class-এ training, তাই Myopia আর AMD-এর সাথে গোলমাল আর নেই
  - পরিষ্কার data
  - DINOv3
  - ConvNeXt-T আর দুটো ViT-এর ensemble
- **আমার অনুমান ~৮৭–৯১% accuracy।** এটা মাপা নয়।
- পুরো fold আর 3-fold CV-তে সংখ্যা fold 0-এর থেকে একটু আলাদা হবে।
- সবচেয়ে বড় বাধা ODIR-এর মৃদু DR। Paper-এ এটা per-source table দিয়ে সৎভাবে দেখাতে হবে।

---

## ৮. Kaggle-এ চালানো

1. **File → Import Notebook** → `Version 2/class-4-v2.ipynb`।
2. **Add Input:**
   - শুধু `manan1717/ocular-disease-dataset`
3. **Settings:** GPU **T4 x2**, Internet **ON**।
4. প্রথমে Part 1–8 interactive চালিয়ে মিলিয়ে নিন:
   - Part 3: `data root: …/manan1717/ocular-disease-dataset/preprocessed`।
   - Part 4: `DATA-QUALITY TABLE`, `Removed as ungradable`, আর `fig_removed_ungradable` চোখে দেখুন।
   - Part 8: ৪টা model-এর পাশে `ok` (DINOv3 সহ)।
5. তারপর **Save Version → Save & Run All (Commit)**।
6. **Output:**
   - `eyev2/v2_results.zip`, বা `/kaggle/working/v2_results.zip`। শেষ cell-এ download link-ও আসে।
   - Zip-এ থাকবে: `REPORT.md`, `paper_numbers.tex`, `summary.json`, `audit.json`, `data_quality.csv`, `splits.csv`, সব table (CSV + LaTeX) আর সব figure (**600 dpi PNG + vector PDF**)।
7. Zip আমাকে দিন। ওটা দেখে paper-এর সংখ্যা আর লেখা ঠিক করে দেব।

**যাচাই:**
- আপনার PC-তে শুধু নকল ছবি দিয়ে (A–N folder, ODIR রোগী, Mendeley আর cataract উৎসের নকল কাঠামো) ছোট CPU test চালিয়েছি। শেষ test শুধু একটা dataset দিয়ে, ~১.৫ মিনিট।
- Part 2–16 শেষ পর্যন্ত চলেছে, DINOv3 সহ ৪টা model train হয়েছে, ১৫টা figure তৈরি হয়েছে।
- ইচ্ছে করে রাখা Glaucoma+Myopia conflict, duplicate কপি, non-fundus ছবি আর ungradable ছবি, সব ধরা পড়েছে।
- এই PC-তে scikit-learn আটকানো আর grad-cam নেই। তাই test-এ ওদের জায়গায় ছোট বিকল্প ব্যবহার করেছি। Kaggle-এ আসলগুলো চলবে।
- আসল dataset বা GPU-তে এখনো চালানো হয়নি। সময়ের সংখ্যাগুলোও অনুমান।

---

## ৯. Paper-এ যা যা বদলাতে হবে

1. **Title:** উদাহরণ: *"Leakage-Free, Source-Aware Four-Class Fundus Disease Classification: Data Auditing, Cross-Source Validation and Self-Supervised Vision Transformers"*।
2. **Abstract:**
   - "6,086 images from Sylhet Eye Hospital" আর "93.77%" সরাতে হবে।
   - নতুন: multi-source dataset (ODIR-2019, Mendeley, Kaggle cataract), audit-এর সংখ্যা (`\ConflictImages`, `\UngradableImages`, `\DupCopiesRemoved`, `\UsedImages`), CV result (`\PropAcc` ± `\PropAccSD`), source-shift result, DINOv3-এর তুলনা।
3. **Introduction → contribution (৪টা):**
   - (১) Public multi-source fundus data-র audit: label conflict, duplicate, patient leakage, উৎস-shortcut, ungradable ছবি
   - (২) Leakage-মুক্ত, patient- আর উৎস-সচেতন benchmark
   - (৩) Cross-source (source-shift) validation
   - (৪) Supervised CNN বনাম self-supervised ViT (DINOv2/DINOv3), মাপা XAI সহ
4. **Dataset section:**
   - Manan1717-এর ৫টা মূল উৎস cite করতে হবে (ODIR-2019, Mendeley Eye Disease Dataset, Kaggle Eye Diseases Classification ইত্যাদি)। মূল উৎসের DOI বা link তাদের নিজেদের page থেকে নিন।
   - স্পষ্ট লিখুন যে Normal-এর সব ছবি ODIR থেকে, আর এই উৎস-পার্থক্য কীভাবে মাপা হয়েছে (shortcut baseline, ODIR-only, source-shift)।
   - A/H/M কেন বাদ (Part ২-এর table)।
   - পুরনো "Sylhet Eye Hospital" দাবি সরাতে হবে। ওই Kaggle release-এর DR আর Glaucoma ফাইল Mendeley-র ফাইলের সাথে হুবহু এক।
5. **Method section:** হুবহু CELL 0 (Part 2)-এর `CFG` অনুযায়ী লিখতে হবে:
   - Harmonisation, round FOV crop, CLAHE
   - Ungradable আর duplicate নিয়ম
   - Grouped stratified K-fold
   - AdamW, MixUp, label smoothing, EMA, early stopping, ৮ epoch
   - Image size: ResNet50 256 px, বাকিগুলো 224 px
   - Flip TTA, stacking + temperature scaling
   - DINOv2/DINOv3-এ layer-wise LR decay 0.75
6. **Table:** `tab_data_quality`, `tab_sources`, `tab_main_results`, `tab_statistics`, `tab_per_class`, `tab_per_source`, `tab_shortcut`, `tab_source_shift_models`, `tab_normal_glaucoma`, `tab_xai`, `tab_complexity`। সংখ্যা হাতে টাইপ না করে `\input{paper_numbers}` দিয়ে macro ব্যবহার করুন।
7. **Figure:** `fig_sources`, `fig_data_quality`, `fig_removed_ungradable`, `fig_preprocessing`, `fig_confusion`, `fig_roc`, `fig_learning_curves` (আসল log থেকে), `fig_source_shift_models`, `fig_gradcam`, `fig_xai_faithfulness`, `fig_benchmark`।
8. **Discussion:**
   - ODIR-এর মৃদু DR-কে Normal বলা হয়, এটাই মূল সীমা।
   - Mendeley-তে result উঁচু কেন (একই হাসপাতাল)।
   - Source-shift-এ performance কতটা কমে।
   - Shortcut baseline থেকে বোঝা যায় retina না দেখে কতটা পারা যেত।
9. **Limitations:**
   - Mendeley আর বাকি উৎসে patient ID নেই (duplicate cluster দিয়ে proxy)
   - ODIR-এর label রোগী-ভিত্তিক keyword থেকে
   - Optic disc-এর অবস্থান heuristic দিয়ে বের করা
   - Ungradable filter স্বয়ংক্রিয় (চোখে যাচাই করা figure দেওয়া আছে)
10. **Comparison table:** অন্য paper-গুলোর dataset আর split আলাদা, অনেকগুলোতে leakage সম্ভব। তাই সরাসরি তুলনা করা যায় না, এটা স্পষ্ট লিখতে হবে।
11. **পুরনো অমিলগুলো যেন আর না থাকে:** হাতে টাইপ করা loss curve, অন্য model-এর confusion matrix-কে ensemble-এর বলা, method আর code-এর অমিল। নতুন সব সংখ্যা আর figure এক run থেকে আসে।
12. **ভাষা:** পুরো paper আবার লেখা (Reviewer #1)। Run শেষ হলে zip পেলে আমি করে দিতে পারি।
