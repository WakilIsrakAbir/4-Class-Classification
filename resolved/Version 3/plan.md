# Version 3 plan: ODIR-5K, 4 class (Cataract / referable DR / Glaucoma / Normal), 384 px, রোগী-ভিত্তিক মূল্যায়ন

**ফাইলগুলো কোথায়:**

| Folder | কী আছে |
|---|---|
| `resolved/Version 1/` | আপনার `class-7-correction.ipynb` আর `eye7_results.zip` (বদলানো হয়নি) |
| `resolved/Version 2/` | V2-এর plan, notebook আর `v2_results.zip` (বদলানো হয়নি) |
| `resolved/Version 3/version-3-class-4.ipynb` | নতুন notebook। V2-এর একই ১৬টা Part, শুধু নিচে লেখা জায়গাগুলোয় বদল। |
| `resolved/Version 3/plan.md` | এই plan |

**Dataset (শুধু একটা):** official ODIR-5K, Kaggle-এ `andrewmvd/ocular-disease-recognition-odir5k`।
- এর `full_df.csv`-এ প্রতিটা চোখের ছবির জন্য একটা সারি: label, রোগী ID, আর ওই চোখের diagnosis keyword।
- এর `preprocessed_images/` = 512×512 px ছবি।
- `tanjemahamed/odir5k-classification` আর লাগবে না। ওটা এই একই ছবি, তবে 224 px-এ ছোট করা, রং উল্টানো, রোগী ID মুছে দেওয়া।

---

## ০. সংক্ষেপে (আপনার তিনটা সিদ্ধান্ত)

1. **Official 512 px ছবি।** Training 384 px-এ (ResNet50) আর 336 px-এ (DINOv2)। V2 ছিল 224–256 px।
2. **Class ৪টা: C, D, G, N।** A/H/M/O শুধু duplicate আর label conflict খোঁজায় স্ক্যান হয়।
3. **Referable DR:**
   - DR হিসেবে রাখা হয় শুধু মাঝারি NPDR বা তার বেশি (ICDR অনুযায়ী)।
   - মৃদু NPDR আর মাত্রা-না-লেখা DR (৫০৯টা চোখ) dataset থেকেই বাদ।
   - কারণ V2-এর মোট ভুলের ৬৮% ছিল ODIR-এর DR↔Normal, যার বড় অংশ এই মৃদু DR।
4. **সাথে নতুন:**
   - **রোগী-ভিত্তিক 5-fold CV:** একই রোগীর দুই চোখ সবসময় একই fold-এ। রোগী ID সরাসরি official table থেকে।
   - **Single-diagnosis cohort:** যে চোখের keyword-এ দ্বিতীয় কোনো ODIR রোগ আছে সেটা বাদ। "suspected" label-ও বাদ।
   - **Balancing:** sampler আর loss weight মিলিয়ে training-এ সব class-এর ওজন সমান। শেষে প্রতি class-এ কতটা ঝোঁক থাকবে (decision prior) তা ঠিক হয় post-hoc logit adjustment দিয়ে।
   - **Protocol তুলনা:** একই model একবার random image-level split-এ train হয়, যেভাবে অনেক paper করে। ফলে দেখা যায় ওই পদ্ধতিতে সংখ্যা কতটা কৃত্রিমভাবে বাড়ে। এটা আলাদা table, মূল result নয়।
5. **সময়:** `fast` ~৩.৭ ঘণ্টা, `lite` ~১.৫ ঘণ্টা, `full` ~৫ ঘণ্টা (অনুমান)।

---

## ১. অন্য paper-এ এত বেশি result কেন: আমাদের জবাব

| অন্যদের উপায় | আমরা কী করছি |
|---|---|
| **Random image-level split, duplicate পরীক্ষা নেই।** উদাহরণ: Kaggle 4-class dataset-এ EfficientNetB3, 95.12%। Test accuracy validation-এর চেয়ে বেশি, confusion matrix test-এর আকারের সাথে মেলে না। | রোগী-ভিত্তিক CV মূল result। একই model random split-এ কত পায়, সেটা **আলাদা table-এ** (`tab_protocol`) দেখানো হয়। এতে literature-এর সাথে ন্যায্য তুলনা হয়, আর ফাঁকটা নিজেই একটা contribution। |
| **Multi-source dataset, যেখানে class মানেই উৎস।** V2-এ মেপেছি: Mendeley অংশ 97%, ODIR অংশ 78%। | এক উৎস (ODIR-5K), তাই উৎস দেখে class বোঝার shortcut নেই। |
| **বড় resolution (448–512 px)** | 384 আর 336 px |
| **সহজ task সংজ্ঞা** | Referable DR (clinically স্বীকৃত সংজ্ঞা, paper-এ স্পষ্ট লিখতে হবে) |
| **অন্য বড় DR dataset-এ pretraining** | করা হয়নি, কারণ আপনি এক dataset চান |

---

## ২. Data cleaning: হিসাব (official table থেকে গোনা)

| Class | Official ছবি | DR মাত্রা কম | একাধিক রোগ | Suspected | থাকে |
|---|---|---|---|---|---|
| Cataract | 293 | – | 13 | 0 | **280** |
| DR | 1,608 | 509 | 82 | 1 | **1,016** |
| Glaucoma | 284 | – | 33 | 32 | **219** |
| Normal | 2,873 | – | – | – | **2,873** |
| মোট | 5,058 | 509 | 128 | 33 | **4,388** |

- **যে DR থাকে:** মাঝারি ৮৪৮, গুরুতর ১৪৬, PDR ২২।
- **রোগী:** ২,৬৭৬ জন, যাদের ১,৭১২ জনের দুই চোখই data-য় আছে।
- **Imbalance:** Normal : Glaucoma ≈ **১৩ : ১**।
- **ছবি-ভিত্তিক cleaning আরও আছে** (সংখ্যা ছোট হবে, Kaggle run-এ জানা যাবে):
  - Ungradable: খুব অন্ধকার, ঝলসানো বা ঝাপসা; cataract-এ blur নিয়ম নেই
  - প্রায়-একই কপি আর label conflict
- **"একাধিক রোগ" কী:** চোখের keyword-এ নিজের রোগ ছাড়া AMD, cataract, DR, glaucoma, hypertensive বা myopia-র যেকোনোটা থাকলে বাদ। Drusen, ERM, laser spot-এর মতো "other" finding এই নিয়মে পড়ে না।
- **ছবির কোণার সাদা letterbox** কালো করা হয়। আগের নমুনায় এটা cataract-এ বেশি ছিল, ফলে class-এর একটা shortcut হতে পারত।

---

## ৩. Preprocessing, resolution, balancing

- **Preprocessing:**
  - রং-ক্রম পরীক্ষা (official ফাইলে স্বাভাবিক)
  - সাদা কোণা কালো করা
  - গোল FOV crop (ছোট বাহুর 98%)
  - CLAHE
  - Cache 384 px-এ
- **Augmentation:** flip, rotation, colour jitter, blur, noise, MixUp। V2-এর random-resolution আর JPEG augmentation বন্ধ, কারণ এগুলো ছোট lesion ঝাপসা করে দিত।
- **Balancing (training):** ছবি বাছাইয়ের সম্ভাবনা ∝ 1/√n, loss weight ∝ 1/√n। দুটো মিলিয়ে সব class-এর ওজন সমান। কপি বা কৃত্রিম ছবি নেই।
- **Decision prior (Part 11):**
  - প্রতিটা model-এর output-কে post-hoc logit adjustment দিয়ে class frequency^0.5 prior-এ আনা হয়। Stacker-ও একই prior ব্যবহার করে।
  - এই 0.5 মান **আগে থেকে ঠিক করা।** Test result দেখে বাছাই করা হয়নি।
  - Part 12-এর `tab_decision_prior`-এ একই prediction তিনভাবে দেখানো হয়:
    - natural (সবচেয়ে বেশি accuracy)
    - square root (মূল)
    - balanced (সবচেয়ে বেশি balanced accuracy)

---

## ৪. Model আর সময়

| Model | Input | কেন |
|---|---|---|
| ResNet50 | 384 px | T4-এ দ্রুত, বড় ছবিতে ছোট lesion ধরতে পারে |
| DINOv2 ViT-B/14 | 336 px (24×24 token) | Transformer। V2-এ ConvNeXt-এর সমান। |
| Stacking + temperature scaling | – | Proposed |

**বাদ:**
- **ConvNeXt-T:** T4-এ ধীর। V2-তে 224 px-এই প্রতি fold-এ ২৪ মিনিট লাগত।
- **DINOv3:** V2-এ প্রতিটা fold-এ সবার শেষে ছিল।

**সময়** (V2-এর মাপা epoch-সময় থেকে হিসাব; অনুমান):

| Preset | কী | মোট |
|---|---|---|
| `fast` (default) | 5 fold × 10 epoch + protocol run | ~৩.৭ ঘণ্টা |
| `lite` | 3 fold, DINOv2 224 px | ~১.৫ ঘণ্টা |
| `full` | 5 fold × 14 epoch | ~৫ ঘণ্টা |

`time_budget_h` পার হতে যাবে এমন run শুরুই হয় না, ফলে report সবসময় তৈরি হয়।

---

## ৫. Reviewer-এর আপত্তি → Version 3

| আপত্তি | সমাধান | কোথায় |
|---|---|---|
| R1: contribution কম | (১) ODIR-5K label audit: single-diagnosis cohort, suspected label, referable DR। (২) রোগী-ভিত্তিক, leakage-মুক্ত benchmark। (৩) **Protocol তুলনা:** image-level split-এ সংখ্যা কতটা বাড়ে, মাপা। (৪) 384 px CNN বনাম self-supervised ViT, stacking। (৫) DR মাত্রা, image-quality shortcut, decision prior বিশ্লেষণ। | Part 4, 12, 14 |
| R1: figure-এর মান | 600 dpi PNG আর vector PDF | সব figure |
| R4: novelty (soft voting) | Ensemble মূল দাবি না | Part 11 |
| R4: k-fold CV | রোগী-ভিত্তিক stratified 5-fold, mean ± SD | Part 5, 12 |
| R4: statistics | Bootstrap CI, McNemar + Holm, corrected t-test, Friedman | Part 12 |
| R4: patient-level validation | রোগী ধরে fold, আর random split-এর সাথে সরাসরি তুলনা | Part 5, 14 |
| R4: external validation | **খোলা থাকছে।** এক dataset চাওয়া হয়েছে। Limitation-এ স্পষ্ট লিখতে হবে। Part 6-এ folder দিলে চলবে। | Part 6 |
| R4: Grad-CAM | Deletion/insertion, pointing game, random baseline | Part 15 |
| R4: Normal→Glaucoma | আলাদা table | Part 12 |

---

## ৬. Result কেমন আশা করবেন (সৎ হিসাব)

- **ভিত্তি:**
  - V2-এ শুধু ODIR অংশে (4 class, 224–256 px, সব DR সহ) accuracy ছিল 78%।
  - সেখানে DR recall ~56%, glaucoma ~45%। DR↔Normal ভুল ছিল মোট ভুলের 68%।
- **V3-এ যা সাহায্য করবে:**
  - মৃদু DR বাদ (সবচেয়ে বড় প্রভাব)
  - 384 px
  - পরিষ্কার label
  - 5 fold
- **আমার অনুমান:** মূল (রোগী-ভিত্তিক) accuracy **~৮৪–৮৮%**। এটা মাপা নয়, Kaggle run-এর পরেই জানা যাবে।
- **Random split-এর সংখ্যা (`tab_protocol`)** এর চেয়ে বেশি হবে বলে আশা করছি। ওটা paper-এ "literature protocol" হিসেবে আলাদা দেখাতে হবে, মূল result হিসেবে নয়।
- **সবচেয়ে কঠিন class:** Glaucoma, কারণ ছোট class আর ছবিগুলো বেশি অন্ধকার।

---

## ৭. Kaggle-এ চালানো

1. **File → Import Notebook** → `Version 3/version-3-class-4.ipynb`।
2. **Add Input:** শুধু `andrewmvd/ocular-disease-recognition-odir5k`।
3. **Settings:** GPU **T4 x2**, Internet **ON**।
4. প্রথমে Part 1–8 interactive চালিয়ে মিলিয়ে নিন:
   - **Part 3:**
     - `table: …/full_df.csv`
     - `Images per ODIR label` (মোট ৬,৩৯২)
     - `Colour channels: … normal order`
   - **Part 4:**
     - `DATA-QUALITY TABLE` (Used: C ~280, D ~1,016, G ~219, N ~2,873)
     - `IMAGE QUALITY PER CLASS`
     - `fig_removed_ungradable`
   - **Part 5:** `Protocol-comparison split … % of its test images have the patient's other eye in training`।
   - **Part 8:** দুটো model-এর পাশে `ok`।
5. তারপর **Save Version → Save & Run All (Commit)**।
6. **Output:** `eyev3/v3_results.zip`, বা `/kaggle/working/v3_results.zip` (শেষ cell-এ download link-ও আসে)। ভেতরে থাকবে:
   - REPORT.md আর paper_numbers.tex
   - সব table (CSV + LaTeX)
   - সব figure (600 dpi PNG + PDF)
7. Zip আমাকে দিন। ওটা দেখে paper-এর সংখ্যা আর লেখা ঠিক করে দেব।

**যাচাই:**
- **ছোট CPU test:** নকল official ODIR কাঠামো দিয়ে চালিয়েছি (১৫৯টা আসল নমুনা ছবি 512 px-এ, নকল `full_df.csv`)।
  - Part 2–16 শেষ পর্যন্ত চলেছে (~১.২ মিনিট)।
  - ইচ্ছে করে রাখা মৃদু DR, "glaucoma + DR", "myopia + DR" আর "suspected glaucoma" চোখ সব বাদ পড়েছে।
  - Protocol run আর `tab_protocol` তৈরি হয়েছে।
  - Decision prior table আর নতুন LaTeX macro এসেছে।
- **এখনো হয়নি:** আসল GPU-তে চালানো। সময় আর result অনুমান।

---

## ৮. Paper-এ যা যা বদলাতে হবে

1. **Title:** উদাহরণ: *"Patient-Level Four-Class Fundus Disease Classification on ODIR-5K: Label Auditing, Referable-DR Definition, and the Optimism of Image-Level Evaluation"*।
2. **Abstract:**
   - ODIR-5K per-eye, ৪ class, referable DR।
   - Cleaning-এর সংখ্যা: `\MildDRRemoved`, `\MultiDiseaseImages`, `\UncertainImages`, `\UsedImages`।
   - রোগী-ভিত্তিক 5-fold result: `\PropAcc` ± `\PropAccSD`, `\PropFOne`, `\PropBalAcc`, `\PropAUC`।
   - Protocol তুলনা: `\CVAccSameModel` বনাম `\NaiveSplitAcc`, আর random split-এ `\NaiveSplitLeakPct`% test ছবির অন্য চোখ training-এ।
3. **Contribution (৪টা):**
   - (১) ODIR-5K-এর keyword-ভিত্তিক label audit আর single-diagnosis cohort
   - (২) Referable-DR সংজ্ঞায় রোগী-ভিত্তিক benchmark
   - (৩) Image-level বনাম patient-level মূল্যায়নের পার্থক্য মাপা
   - (৪) 384 px CNN + self-supervised ViT stacking, balanced training আর decision prior বিশ্লেষণ, মাপা XAI সহ
4. **Dataset section:**
   - ODIR-2019 challenge cite করুন, আর Kaggle-এর official copy-র উল্লেখ।
   - Cleaning table: `tab_data_quality`।
   - Referable DR-এর সংজ্ঞা (ICDR: মাঝারি NPDR বা তার বেশি)।
   - A/H/M/O কেন task-এর বাইরে।
   - Imbalance ১৩:১।
5. **Method:** Part 2-এর `CFG` অনুযায়ী হুবহু লিখতে হবে:
   - 512 px থেকে FOV crop আর CLAHE; ResNet50 384 / DINOv2 336
   - Balanced sampler + loss weight; post-hoc logit adjustment (prior power 0.5, আগে থেকে ঠিক করা)
   - AdamW, MixUp, label smoothing, EMA, early stopping, 10 epoch
   - Flip TTA, stacking + temperature scaling
6. **Table:** `tab_data_quality`, `tab_main_results`, `tab_statistics`, `tab_per_class`, `tab_protocol`, `tab_decision_prior`, `tab_dr_grade`, `tab_shortcut`, `tab_normal_glaucoma`, `tab_referral`, `tab_xai`, `tab_complexity`। Macro দিয়ে: `\input{paper_numbers}`।
7. **Figure:** `fig_preprocessing`, `fig_removed_ungradable`, `fig_confusion`, `fig_roc`, `fig_learning_curves`, `fig_calibration_referral`, `fig_gradcam`, `fig_xai_faithfulness`, `fig_benchmark`।
8. **Comparison with literature:**
   - অন্য paper-এর ৯২–৯৫% সাধারণত image-level split বা multi-source data-য়।
   - আমাদের `tab_protocol` দেখায় একই protocol-এ আমাদের model কত পায়।
   - তাই সরাসরি তুলনা শুধু একই protocol-এ। এটা স্পষ্ট লিখতে হবে।
9. **Discussion:**
   - Referable DR কেন (মৃদু NPDR = শুধু microaneurysm, ছবিতে প্রায় অদৃশ্য)।
   - Glaucoma ছোট আর কঠিন।
   - Decision prior-এর trade-off।
   - Quality-only baseline।
10. **Limitations:**
    - এক উৎস (চীন), external test নেই
    - মৃদু DR task-এর বাইরে, তাই screening-এ প্রথম ধাপের DR ধরা যাবে না
    - Label এসেছে keyword থেকে, ডাক্তার আবার যাচাই করেননি
    - Official ছবি 512 px পর্যন্ত
11. **পুরনো অমিল যেন না থাকে:** সব সংখ্যা আর figure একটা run থেকে আসবে। পুরনো "Sylhet Eye Hospital" দাবি আর 93.77% সরাতে হবে।
