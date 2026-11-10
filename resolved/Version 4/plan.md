# Version 4 plan: official ODIR-5K-এর পূর্ণ বিশ্লেষণ, cleaning আর ৫-class classification

**ফাইলগুলো কোথায়:**

| ফাইল | কী আছে |
|---|---|
| `resolved/Version 4/version-4-class-5.ipynb` | নতুন notebook (V2-এর একই ১৬টা Part) |
| `resolved/Version 4/plan.md` | এই plan |
| `resolved/Version 3/` | চালানো হয়নি, রেখে দেওয়া হয়েছে। V4 এর সব সুবিধা নিয়ে আরও এগিয়েছে। |

**Dataset (শুধু একটা):** Kaggle `andrewmvd/ocular-disease-recognition-odir5k` = official ODIR-5K।

---

## ০. সংক্ষেপে

1. **নিজেরা প্রতিটা চোখের label বানানো:** `data.xlsx`-এর keyword থেকে, আর **আসল ৭,০০০টা ছবি** দিয়ে।
   - `full_df.csv`-এ ৬০৮টা চোখ নেই।
   - `full_df.csv` প্রথম keyword দেখে label দেয়, তাই **৫৮টা DR চোখ "Other"-এ লুকিয়ে আছে।**
   - `preprocessed_images` চ্যাপ্টা (512×512-এ টেনে বসানো), আর কিছু ছবির কোণায় সাদা।
2. **Class ৫টা: Cataract, DR (referable), Glaucoma, Myopia, Normal।**
   - Myopia যোগ হলো: ODIR-এর সবচেয়ে সহজে চেনা class (প্রকাশিত রোগী-ভিত্তিক F1 0.89, AUC 0.996)।
   - AMD, Hypertension, Other শুধু audit-এ স্ক্যান হয়।
   - Paper-এর পুরনো ৪টা চাইলে Part 2-এ `classes=["C","D","G","N"]` করলেই হবে।
3. **নতুন cleaning:**
   - ডাক্তারের লেখা quality keyword (৩০টা চোখ)
   - **Lens dust:** camera-র কাঁচের ধুলোর কালো দাগ দেখতে lesion-এর মতো। ২৭৭টা normal চোখে আছে।
   - Normal চোখে অন্য finding
   - V3-এর নিয়মগুলো: referable DR, একাধিক রোগ, "suspected"
4. **নতুন model: optic disc expert।**
   - আসল বড় ছবি থেকে optic disc-এর জায়গা কেটে আলাদা ResNet50 (224 px)।
   - Glaucoma চেনা যায় cup আর rim দেখে, যেটা পুরো ছবি ছোট করলে হারিয়ে যায়।
   - তিনটা model-এর stacking = proposed (DA-MVE)।
5. **V3-এর সুবিধাগুলো থাকছে:** রোগী-ভিত্তিক 5-fold, balanced sampling, আগে থেকে ঠিক করা decision prior, protocol তুলনা।
6. **Preprocessing আপগ্রেড (section ১৩, ১০০টা আসল ছবি মেপে):**
   - **FOV circle fit:** ২২% ছবিতে চোখের গোল অংশ camera-র ফ্রেমে কাটা। পুরনো crop এসব ছবিকে ২৫% পর্যন্ত zoom করত, আর কয়েকটা glaucoma ছবির optic disc-ই কেটে ফেলত। এখন সব camera-র ছবি একই scale-এ, পুরো চোখসহ।
   - **নতুন optic disc locator:** disc ঠিক ধরার হার ~৮২% থেকে ~৯৫%।
   - **রং আর orientation:** চারটা variant-এর মধ্যে সেরাটা fold 0-এর validation data দেখে বাছাই হয় (ablation)।
7. **Augmentation আর class balancing পুনর্যাচাই (section ১৪):** তিনটা জিনিস ঠিক করা হয়েছে।
   - PyTorch-এর weighted label smoothing-এর bug। এর ফলে normal ছবিও glaucoma, myopia বা cataract-এর দিকে ঠেলে দেওয়া হচ্ছিল।
   - খুব শক্ত blur augmentation।
   - Normalization layer আর bias-এ weight decay।
8. **সময়:** `fast` ~৪.৭ ঘণ্টা (ablation-এর ~২৫ মিনিট সহ), `lite` ~১.৬ ঘণ্টা, `full` ~৬.২ ঘণ্টা (অনুমান)।

---

## ১. Dataset বিশ্লেষণ

### ১.১ কী কী আছে

| অংশ | বিষয়বস্তু | আমাদের ব্যবহার |
|---|---|---|
| `ODIR-5K/ODIR-5K/data.xlsx` | ৩,৫০০ রোগী: বয়স, লিঙ্গ, **দুই চোখের আলাদা diagnosis keyword**, রোগী-স্তরের ৮টা label (N D G C A H M O) | **প্রতিটা চোখের label-এর মূল উৎস** |
| `ODIR-5K/ODIR-5K/Training Images/` | ৭,০০০টা আসল ছবি (১.৪৩ GB) | **ছবির উৎস** |
| `ODIR-5K/ODIR-5K/Testing Images/` | ১,০০০টা ছবি, label নেই | ব্যবহার হয় না |
| `preprocessed_images/` | ৬,৩৯২টা, 512×512 | বিকল্প (`image_source="preprocessed"`) |
| `full_df.csv` | ৬,৩৯২টা সারি, প্রতিটা চোখের একটা label | তুলনা আর fallback |

### ১.২ যা পেলাম

1. **রোগী-স্তরের label (N…O কলাম) আর চোখের label এক নয়।**
   - উদাহরণ: রোগী ০-এর বাম চোখে cataract, ডান চোখ normal, অথচ রোগী-স্তরে label "C"।
   - তাই প্রতিটা চোখের label তার **নিজের keyword** থেকে নিতে হবে। অনেক paper রোগীর label দুই চোখেই বসিয়ে দেয়, ফলে normal চোখও "রোগ" label পায়।
2. **`full_df.csv`-এ ৬০৮টা চোখ নেই।** তাদের keyword: lens dust ২৯৯, normal fundus ২২৪, laser spot ৫৬, moderate NPDR ৪৯, …
   - আমাদের নিয়মে এর মধ্যে ২২২টা normal, ৭৫টা DR, ১২টা myopia, ১১টা cataract আর ৯টা glaucoma ব্যবহারযোগ্য।
3. **`full_df.csv` প্রথম keyword দেখে label দেয়।**
   - "macular epiretinal membrane, moderate non proliferative retinopathy" → **Other**, অথচ চোখে DR আছে।
   - এভাবে **৫৮টা DR চোখ "O"-তে** গেছে।
   - আমাদের নিয়ম: চোখের keyword-এ নাম থাকা **প্রথম ODIR রোগ** (DR / glaucoma / cataract / AMD / hypertensive / myopia), রোগ না থাকলে "normal fundus" → N, তাও না থাকলে O।
   - এই তুলনা notebook নিজে `label_audit_full_df_vs_keywords.csv`-এ বানায়, যেটা paper-এর জন্য একটা contribution।
4. **আসল ছবি একাধিক camera-র।** ২৫৯২×১৭২৮, ২০৪৮×১৫৩৬, ১৯৩৬×১২৯৬, ১৪৪৪×১৪৪৪, ২৯৭৬×২৯৭৬ ইত্যাদি। ছোট বাহুর median ১,৭২৮ px, সবচেয়ে ছোট ৩১৬ px।
   - Camera আর class-এর মধ্যে যোগ আছে কিনা, তা "acquisition metadata only" shortcut baseline মাপবে।
5. **`preprocessed_images`-এর দুটো সমস্যা:**
   - আসল ছবি আয়তাকার, অথচ 512×512-এ টানা হয়েছে, ফলে গোল field of view ডিম্বাকার হয়ে গেছে।
   - কিছু ছবির কোণায় সাদা letterbox।
   - আসল ছবি থেকে গোল crop করলে দুটো সমস্যাই থাকে না।
6. **রোগীর দুই চোখ:** ৩,০৩৪ জন রোগীর দুই চোখই আছে। NN ১,০২১, DD ৫৮১, DN ২২৭… তাই রোগী-ভিত্তিক split ছাড়া result কৃত্রিমভাবে বাড়ে।

---

## ২. প্রতি class-এর সমস্যা আর সমাধান

| Class | সমস্যা | V4-এ সমাধান |
|---|---|---|
| **Normal** | ২৭৭টা চোখে lens dust (কালো দাগ, DR-এর lesion-এর মতো); ২২২টা normal চোখ `full_df`-এ নেই; DR রোগীর অন্য চোখ (প্রায়ই খুব প্রাথমিক DR) | Lens dust বাদ; আসল ছবি থেকে সব চোখ; "normal + অন্য finding" বাদ; রোগী-ভিত্তিক fold |
| **DR** | ৫২৭টা মৃদু NPDR (৩২%), প্রায় অদৃশ্য; ২৯টায় মাত্রা লেখা নেই; ৫৮টা "O"-তে লুকানো; hypertensive আর অন্য রোগ একসাথে | Keyword থেকে relabel (৫৮টা ফিরে আসে); referable DR (মাঝারি বা তার বেশি); একাধিক রোগ বাদ; DR-মাত্রা অনুযায়ী recall table |
| **Glaucoma** | সবচেয়ে ছোট (২৩২); ২৮টা "suspected"; myopia বা DR একসাথে; ছবি বেশি অন্ধকার আর ঝাপসা; cup আর rim-এর detail ছোট ছবিতে হারায় | Suspected আর একাধিক রোগ বাদ; **আসল ছবি থেকে disc crop + disc expert**; balanced sampling; quality-only baseline |
| **Cataract** | স্বভাবতই ঝাপসা; preprocessed-এ সাদা কোণা বেশি | Blur নিয়ম থেকে বাদ; আসল ছবি + সাদা-কোণা পরিষ্কার |
| **Myopia** | ছোট (২৫১); "myopia retinopathy" glaucoma-র সাথে আসে | একাধিক রোগ বাদ; class স্পষ্ট বলে সহজ |
| AMD (বাদ) | Drusen সূক্ষ্ম, DR-এর exudate-এর সাথে গোলমাল; ২৪২টা | Task-এর বাইরে |
| Hypertension (বাদ) | ১১৪টা; লক্ষণ অদৃশ্য; DR-এর সাথে মেশানো | Task-এর বাইরে |
| Other (বাদ) | ৮৭৬টা, নানা রোগ মেশানো | Task-এর বাইরে |
| **সবার** | ৩০টা চোখে ডাক্তার নিজেই লিখেছেন "low image quality / image offset / no fundus / anterior segment" | বাদ (`quality_keyword`) |

---

## ৩. কোন class নিলে result ভালো হবে

**প্রমাণ:**
- **Cleaning-এর পর ছবির সংখ্যা** (আমাদের নিয়মে, ৭,০০০টা চোখ থেকে)।
- **রোগী-ভিত্তিক প্রকাশিত per-class F1 / AUC** (ODIR-5K, EfficientNet-B4 + Inception-ResNet-v2, 448 px; [model card](https://huggingface.co/Mjolnirslams/retinal-disease-ensemble))।

| Class | Cleaning-এর পর | প্রকাশিত F1 | প্রকাশিত AUC | মন্তব্য |
|---|---|---|---|---|
| Myopia | ~২৪৫ | **0.889** | **0.996** | সবচেয়ে সহজ |
| Cataract | ~২৬৯ | **0.857** | 0.980 | সহজ |
| DR | ~১,১০০ (referable) | 0.717 (সব DR) | 0.860 | Referable করলে আরও ভালো হওয়ার কথা |
| Normal | ~২,৮২০ | 0.659 | 0.821 | — |
| AMD | ~২৪২ | 0.659 | 0.938 | মাঝারি কঠিন |
| Glaucoma | ~২২২ | 0.649 | 0.959 | কঠিন, তবে paper-এর মূল class |
| Hypertension | ১১৪ | **0.306** | 0.816 | সবচেয়ে খারাপ |
| Other | ৮৭৬ | 0.518 | 0.737 | একটা রোগ নয় |

**সিদ্ধান্ত: C, D (referable), G, M, N।**
- Myopia আর Cataract সহজ, তাই মোট accuracy আর macro-F1 বাড়ে।
- Glaucoma paper-এর মূল class, তাই থাকছে। Disc expert দিয়ে শক্তিশালী করা হয়েছে।
- AMD, Hypertension, Other যোগ করলে result কমত।
- **মোট আনুমানিক ~৪,৬০০টা ছবি।** Normal : Glaucoma ≈ ১৩:১।

---

## ৪. ODIR-এ ৯০%+ result: কারা, কীভাবে

| Paper বা উৎস | দাবি | কীভাবে / protocol | আমাদের মন্তব্য |
|---|---|---|---|
| [HybridVisionNet (2025)](https://www.sciencedirect.com/science/article/pii/S2090447925003351) | accuracy 99.71% | Full text পাইনি (403) | ODIR-এ রোগী-ভিত্তিক split-এ এটা অবিশ্বাস্য। খুব সম্ভব image-level split বা augmentation split-এর আগে। |
| আরেকটা ODIR paper ([survey listing](https://www.researchgate.net/publication/391691991_Deep_Learning_for_Ocular_Disease_Classification_A_Study_on_ODIR-5K_Dataset)) | 99.44% acc | যাচাই করা যায়নি | একই সন্দেহ |
| 5-class tailored CNN (একই উৎস) | 96.32% | যাচাই করা যায়নি | — |
| [DKCNet](https://arxiv.org/abs/2207.07918) | AUC 96.08, F1 94.28, kappa 0.81 | Keyword দিয়ে প্রতিটা চোখের label (আমাদের মতো), **oversampling + undersampling**, InceptionResNet backbone। Split-এর বিবরণ abstract-এ নেই। | চোখ-ভিত্তিক label আর শক্তিশালী backbone ভালো ধারণা। Oversampling split-এর আগে হলে leakage। |
| [EfficientNetB3 (Kaggle 4-class)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12387119/) | 95.12% | Random 70/20/10, duplicate পরীক্ষা নেই, test validation-এর চেয়ে ভালো | Leaky protocol |
| [রোগী-ভিত্তিক ODIR ensemble](https://huggingface.co/Mjolnirslams/retinal-disease-ensemble) | macro-AUC 0.888 | EfficientNet-B4 + Inception-ResNet-v2, 448 px, রোগী ধরে 80/20 | সৎ protocol-এ বাস্তব মাত্রা |
| একই card-এর দাবি (যাচাই হয়নি) | ~0.93 AUC | EyePACS / APTOS / MESSIDOR-এ আগে pretraining | বাইরের data লাগে, তাই বাদ (এক dataset) |

**সারকথা:**
- ৯৫–৯৯% সাধারণত চার কারণে আসে:
  - image-level split (এক রোগীর দুই চোখ train আর test-এ)
  - split-এর আগে oversampling বা augmentation
  - রোগীর label দুই চোখে বসানো
  - multi-label-এ প্রতি label-এর গড় accuracy (negative অনেক বলে কৃত্রিমভাবে বেশি)
- **আমরা নিচ্ছি:**
  - চোখ-ভিত্তিক keyword label
  - বড় resolution
  - শক্তিশালী backbone আর ensemble
  - **split-এর পরে** balanced sampling
  - সহজ, পরিষ্কার class-এর সংজ্ঞা
- **আমরা নিচ্ছি না:** leaky split-কে মূল result বানানো। সেটা আলাদা `tab_protocol` table-এ যায়, যাতে literature-এর সাথে একই protocol-এ তুলনা করা যায়।

---

## ৫. Model নির্বাচন

| Model | Input | কেন |
|---|---|---|
| ResNet50 (পুরো fundus) | 384 px | T4-এ দ্রুত, শক্তিশালী CNN baseline |
| DINOv2 ViT-B/14 (পুরো fundus) | 336 px | Self-supervised transformer। V2-এ সেরার কাছাকাছি (84.6%)। |
| **ResNet50 disc expert** | 224 px disc crop, আসল ছবি থেকে | Glaucoma-র cup আর rim। V1-এ disc crop 512 px ছবি থেকে হতো বলে আসল detail ছিল না। এখন ~১,৭০০ px ছবির ৩০% কাটা অংশ, ফলে সত্যিকারের detail। |
| **Stacking (DA-MVE) + temperature scaling** | – | Proposed |

**কেন অন্যগুলো নয়:**
- **EfficientNet-B4, ConvNeXt:** depthwise convolution-এর জন্য T4-এ ধীর। V2-তে ConvNeXt-T 224 px-এই প্রতি fold-এ ২৪ মিনিট নিত।
- **Inception-ResNet-v2:** literature-এ ভালো, কিন্তু 384 px-এ ResNet50-এর ~১.৮ গুণ সময়।
- **RETFound (ViT-L, ৩০৩M):** HuggingFace-এ access চাইতে হয়, আর T4-এ খুব ভারী। Retina-র foundation model শুধু কঠিন DR grading-এ সামান্য সুবিধা দেখিয়েছে ([arXiv 2511.22001](https://arxiv.org/html/2511.22001v1))।
- **RETFound-Green (ViT-S, ৭৫k public ছবি):** weight কোথায় আর pretraining-এ ODIR আছে কিনা, যাচাই করা যায়নি। ভবিষ্যতের বিকল্প।

---

## ৬. Cleaning-এর ক্রম (training-এর আগে; বাদ পড়া ছবি কোথাও থাকে না)

1. **Keyword থেকে:**
   - quality keyword
   - lens dust
   - মৃদু বা মাত্রা-না-লেখা DR
   - normal + অন্য finding
   - "suspected" label
   - একাধিক ODIR রোগ
2. **ছবি থেকে:**
   - Ungradable: খুব অন্ধকার, ঝলসানো বা ঝাপসা; cataract-এ blur নিয়ম নেই
   - Fundus না এমন ছবি
   - প্রায়-একই কপি (cluster প্রতি একটা রাখা)
   - Label conflict (একই বা প্রায়-একই ছবি দুই label-এ)
3. **Preprocessing:** সাদা কোণা কালো করা, **fit করা FOV circle-এ একই scale-এর crop**, cache 384 px (রং বদলানোর আগের অবস্থায়), আসল ছবি থেকে disc crop 224 px। রং আর contrast পরিবর্তন training-এর সময় হয়, ablation-এর বাছাই অনুযায়ী। বিস্তারিত section ১৩-এ।

**আনুমানিক যা থাকবে** (keyword নিয়ম থেকে গোনা; ছবি-ভিত্তিক cleaning-এর পর সামান্য কমবে):

| Cataract | DR | Glaucoma | Myopia | Normal | মোট |
|---|---|---|---|---|---|
| ~২৬৯ | ~১,১০০ | ~২২২ | ~২৪৫ | ~২,৮১৮ | **~৪,৬৫০** |

---

## ৭. Balancing, decision prior, protocol তুলনা (V3-এর মতো)

- **Training:** sampler (1/√n) আর loss weight (1/√n), দুটো মিলিয়ে সব class সমান। কপি বা কৃত্রিম ছবি নেই, আর sampling হয় split-এর পরে।
- **Decision prior:** post-hoc logit adjustment, power 0.5 আগে থেকে ঠিক করা। `tab_decision_prior`-এ natural, square root আর balanced, তিনভাবেই দেখানো হয়।
- **Protocol তুলনা:** ResNet50 একবার random image-level split-এ। কত % test ছবির অন্য চোখ training-এ ছিল, সেটাসহ `tab_protocol`-এ।

---

## ৮. সময় (V2-এর মাপা epoch-সময় থেকে; অনুমান)

| Preset | কী | মোট |
|---|---|---|
| `fast` (default) | Preprocessing ablation (~২৫ মিনিট), তারপর 5 fold × 10 epoch: ResNet50 @384 + DINOv2 @336 + disc expert @224, সাথে protocol run | ~৪.৭ ঘণ্টা |
| `lite` | 3 fold, DINOv2 @224, disc expert নেই | ~১.৬ ঘণ্টা |
| `full` | Ablation + 5 fold × 14 epoch | ~৬.২ ঘণ্টা |

`lite` preset-এ ablation হয় না। সেখানে সরাসরি "রং normalize + CLAHE + বাম চোখ উল্টানো" ব্যবহার হয়, কারণ section ১৩-এর বিশ্লেষণে এটাই সবচেয়ে সম্ভাবনাময়।

আসল ছবি একবার preprocess করতে Kaggle-এ ~৩–৫ মিনিট লাগবে। Time budget পার হতে যাবে এমন run শুরু হয় না, তাই report সবসময় তৈরি হয়।

---

## ৯. Result কেমন আশা করবেন (সৎ হিসাব)

- **ভিত্তি:** V2-এ শুধু ODIR অংশে (4 class, সব DR সহ, 224–256 px) accuracy 78%।
- **V4-এ সাহায্য করবে:**
  - Referable DR
  - Myopia (সহজ class)
  - পরিষ্কার label (relabel, lens dust, quality)
  - 384 px আর disc expert
  - 5 fold
- **আমার অনুমান:** রোগী-ভিত্তিক মূল accuracy **~৮৫–৯০%**, macro-AUC ~০.৯৫ বা তার বেশি। এটা মাপা নয়।
- **Image-level protocol-এ (`tab_protocol`)** এর চেয়ে বেশি হবে বলে আশা করছি। সেটা "literature protocol" হিসেবে আলাদা দেখাতে হবে।
- **৯৫–৯৯%-এর প্রতিশ্রুতি দিচ্ছি না।** Section ৪ অনুযায়ী ওই সংখ্যা সৎ protocol-এ পাওয়া যায় না।

---

## ১০. Reviewer-এর আপত্তি → Version 4

| আপত্তি | সমাধান |
|---|---|
| R1: contribution কম | (১) Official ODIR-5K-এর label audit: `full_df.csv`-এর ৬০৮টা হারানো চোখ আর ৫৮টা লুকানো DR। (২) চোখ-ভিত্তিক keyword label আর মাপা cleaning। (৩) রোগী-ভিত্তিক benchmark আর image-level protocol-এর সাথে তুলনা। (৪) Disc-aware multi-view ensemble। (৫) DR-মাত্রা আর quality-shortcut বিশ্লেষণ। |
| R1: figure | 600 dpi PNG আর vector PDF |
| R4: novelty | Disc expert + stacking, আর data-centric contribution |
| R4: k-fold, statistics | রোগী-ভিত্তিক 5-fold, CI, McNemar + Holm, corrected t-test, Friedman |
| R4: patient-level | রোগী ধরে fold, আর random split-এর সাথে সরাসরি তুলনা |
| R4: external validation | খোলা থাকছে (এক dataset)। Limitation-এ লিখতে হবে। Part 6-এ folder দিলে চলবে। |
| R4: Grad-CAM | Deletion/insertion, pointing game (glaucoma-তে disc), random baseline। Disc expert-এর Grad-CAM-ও আসে। |

---

## ১১. Kaggle-এ চালানো

1. **File → Import Notebook** → `Version 4/version-4-class-5.ipynb`।
2. **Add Input:** শুধু `andrewmvd/ocular-disease-recognition-odir5k`।
3. **Settings:** GPU **T4 x2**, Internet **ON**।
4. প্রথমে Part 1–8 interactive চালিয়ে মিলিয়ে নিন:
   - **Part 3:**
     - `keywords from: data.xlsx | images: original (…/Training Images) | eyes: 7000`
     - `Label audit` table: O সারিতে D কলামে ~৫৮, আর "-" সারি (full_df-এ নেই এমন চোখ)
   - **Part 4:**
     - `DATA-QUALITY TABLE` (Used: C ~২৬৫, D ~১,০৯০, G ~২২০, M ~২৪০, N ~২,৮০০)
     - `IMAGE QUALITY PER CLASS`
     - `fig_removed_ungradable`
   - **Part 3:** `Geometry: … images with a field of view cut by the camera frame` লাইন। আশা করছি মোট ছবির ~২০%; "without a usable circle fit" হাতে-গোনা কয়েকটা।
   - **Part 7:**
     - `fig_preprocessing`: সবুজ বৃত্ত চোখের কিনারায় বসেছে কিনা, আর disc box optic disc-এর ওপর কিনা।
     - `fig_fov_normalisation`: কাটা-FOV ছবিতে পুরনো crop (zoom করা) বনাম নতুন crop (পুরো চোখ)।
   - **Part 9-এর শুরুতে:** preprocessing ablation-এর ৪টা লাইন আর `Preprocessing used for training: …` (কোনটা বাছাই হলো)।
   - **Part 8:** তিনটা model-এর পাশে `ok`, disc-এরটা সহ।
5. তারপর **Save Version → Save & Run All (Commit)**।
6. **Output:** `eyev4/v4_results.zip`, বা `/kaggle/working/v4_results.zip`। ভেতরে থাকবে:
   - REPORT.md আর paper_numbers.tex
   - সব table (CSV + LaTeX), label audit সহ
   - সব figure (600 dpi PNG + PDF)
7. Zip আমাকে দিন। ওটা দেখে paper-এর সংখ্যা আর লেখা ঠিক করে দেব।

**যাচাই:**
- **ছোট CPU test:** নকল official কাঠামো দিয়ে চালিয়েছি (১৫৯টা আসল নমুনা ছবি, চওড়া কালো পাড়ের "আসল" ছবি, নকল table)।
  - Part 2–16 শেষ পর্যন্ত চলেছে (~২ মিনিট)।
  - Label audit table তৈরি হয়েছে।
  - ইচ্ছে করে রাখা "ERM + moderate NPDR" চোখ, যেটা full_df বলে O, সেটা DR হিসেবে ধরা পড়েছে।
  - Lens dust, quality keyword আর "normal + drusen" চোখ বাদ পড়েছে।
  - Disc expert train হয়েছে, DA-MVE তৈরি হয়েছে।
- **এই PC-তে Excel পড়ার library নেই।** তাই এখানে `full_df.csv`-এর fallback পথ পরীক্ষা হয়েছে। Kaggle-এ `data.xlsx` পড়া যাবে, কারণ সেখানে openpyxl আছে।
- **এখনো হয়নি:** আসল GPU-তে চালানো। সময় আর result অনুমান।

---

## ১২. Paper-এ যা যা বদলাতে হবে

1. **Title:** উদাহরণ: *"Auditing ODIR-5K: Per-Eye Relabelling, Patient-Level Evaluation and a Disc-Aware Multi-View Ensemble for Five-Class Fundus Disease Classification"*।
2. **Abstract:**
   - Official ODIR-5K, চোখ-ভিত্তিক keyword label
   - Audit-এর সংখ্যা: হারানো চোখ, লুকানো DR, `\LensDustRemoved`, `\QualityKeywordRemoved`, `\UsedImages`
   - ৫ class, referable DR
   - রোগী-ভিত্তিক 5-fold result: `\PropAcc` ± `\PropAccSD`, `\PropFOne`, `\PropBalAcc`, `\PropAUC`
   - Protocol তুলনা: `\CVAccSameModel` বনাম `\NaiveSplitAcc`
3. **Contribution:**
   - (১) ODIR-5K-এর label audit (`full_df.csv`-এর সমস্যা)
   - (২) চোখ-ভিত্তিক, ডাক্তারের keyword-ভিত্তিক cleaning protocol
   - (৩) রোগী-ভিত্তিক benchmark আর image-level মূল্যায়ন কতটা আশাবাদী, তার মাপ
   - (৪) আসল ছবির disc crop দিয়ে disc-aware multi-view ensemble
   - (৫) DR মাত্রা, quality shortcut, decision prior আর মাপা XAI
4. **Dataset section:**
   - ODIR-2019 challenge cite
   - Section ১-এর findings
   - Label নিয়ম (প্রথম ODIR রোগ)
   - Cleaning table (`tab_data_quality`)
   - Label audit table
   - Class নির্বাচনের যুক্তি (section ৩)
5. **Method:** Part 2-এর `CFG` অনুযায়ী লিখতে হবে:
   - Preprocessing (section ১৩): FOV circle fit আর একই scale-এর crop, 384 px; disc locator (DoG + রক্তনালী + চোখের পাশ); আসল ছবি থেকে disc crop 224 px; ablation-এ বাছাই হওয়া photometric variant আর orientation (`\PrepPhotometric`, `\PrepOrientation`)
   - ResNet50, DINOv2, disc ResNet50
   - Balanced sampler + loss; logit adjustment (0.5)
   - AdamW, MixUp, label smoothing, EMA, early stopping, 10 epoch
   - Flip TTA, stacking + temperature scaling
6. **Table:** `tab_data_quality`, label audit, **`tab_prep_ablation`** (preprocessing ablation, fold 0 validation), `tab_main_results`, `tab_statistics`, `tab_per_class`, `tab_protocol`, `tab_decision_prior`, `tab_dr_grade`, `tab_shortcut`, `tab_normal_glaucoma`, `tab_referral`, `tab_xai`, `tab_complexity`। Macro দিয়ে: `\input{paper_numbers}`।
7. **Figure:** `fig_preprocessing` (FOV circle, ৩টা photometric variant, disc ROI), **`fig_fov_normalisation`** (কাটা-FOV: পুরনো বনাম নতুন crop), `fig_removed_ungradable`, `fig_confusion`, `fig_roc`, `fig_learning_curves`, `fig_calibration_referral`, `fig_gradcam` (disc expert সহ), `fig_xai_faithfulness`, `fig_benchmark`।
8. **Comparison:** Section ৪-এর table থেকে লিখুন যে ৯৫–৯৯%-এর দাবিগুলো image-level বা oversampled protocol-এ, আর আমাদের সংখ্যা একই protocol-এ কত (`tab_protocol`)।
9. **Limitations:**
   - এক উৎস (চীনের হাসপাতাল), external test নেই
   - Label এসেছে keyword থেকে, ডাক্তার আবার যাচাই করেননি
   - মৃদু DR task-এর বাইরে
   - AMD, Hypertension, Other বাদ
   - Disc-এর অবস্থান heuristic দিয়ে বের করা
10. **পুরনো অমিল যেন না থাকে:** "Sylhet Eye Hospital", 93.77% সরাতে হবে। সব সংখ্যা আর figure একটা run থেকে।

---

## ১৩. Preprocessing আপগ্রেড: ১০০টা আসল ছবি মেপে যা পেলাম আর যা বদলালাম

নমুনা: আসল `Training Images` থেকে প্রতি class (C, D, G, M, N) ২০টা করে, মোট ১০০টা (২০.৮ MB)। সব পরীক্ষা এই PC-তে হালকাভাবে চালানো হয়েছে। চূড়ান্ত পরীক্ষাটা notebook-এর নিজের ফাংশন দিয়েই।

### ১৩.১ যা পেলাম

| সমস্যা | মাপ | কেন গুরুত্বপূর্ণ |
|---|---|---|
| **চোখের গোল অংশ (FOV) camera-র ফ্রেমে কাটা** | ১০০টার ২২টায় (median ব্যাসের ১০.৬%, সর্বোচ্চ ২৩%); বিশেষ করে 1624×1232 আর 1536×1152 camera-য় | V1–V4-এর crop (ছোট বাহুর 98% বৃত্ত) এসব ছবিতে আসল FOV-এর মাত্র **৭৫–৮৬%** রাখত। ফলে এগুলো অন্যদের চেয়ে ~২৫% পর্যন্ত zoom হতো, কিনারার retina হারাত, আর **কয়েকটা glaucoma ছবিতে optic disc crop-এর বাইরে চলে যেত** (G_1371_right, G_1383_left)। |
| **Optic disc ভুল জায়গায় ধরা** | DR, glaucoma আর normal-এর ৬০টা ছবি চোখে মিলিয়ে দেখা: পুরনো পদ্ধতি (সবচেয়ে উজ্জ্বল জায়গা) ~৮২% ঠিক | Lens flare, cataract-এর ধোঁয়াশা বা কিনারার আলোকে disc ভেবে বসত। ফলে disc expert ভুল জায়গা পেত, আর XAI pointing game-ও ভুল হতো। |
| **বাম আর ডান চোখের দিক** | দেখা সব ছবিতে ডান চোখের disc ডানে, বাম চোখের বামে | বাম চোখের ছবি উল্টালে সব চোখের anatomy এক দিকে আসে, ফলে কম data-তে model শিখতে সুবিধা হয়। |
| **Camera-র রং আর উজ্জ্বলতা** | শুধু রং বা উজ্জ্বলতা দেখে class আন্দাজ (৫ class, এলোমেলো আন্দাজে ২০%): raw ৪৭%, CLAHE ৫১%, **রং normalize + CLAHE ৩৭%**, Graham ৪৬% | CLAHE camera-র আভা সরায় না। Normalize করলে camera বা আলো দেখে class চেনার shortcut কমে (বাকি অংশের কিছু আসল লক্ষণ, যেমন myopia-র সাদা atrophy)। |
| **আসল ছবিতেও সাদা পটভূমি** | ২০টা cataract-এর ২টায় (2464×1632 camera) | Cataract-এর shortcut হতে পারত। Notebook আগে থেকেই সাদা অংশ কালো করে। |

### ১৩.২ যা বদলালাম (notebook-এ)

1. **FOV circle fit** (Part 3, `fov_circle`):
   - FOV-এর সীমানার বিন্দুগুলোর ওপর least-squares দিয়ে বৃত্ত fit করা হয়।
   - ছবির ফ্রেমে কাটা সোজা অংশের বিন্দু বাদ দেওয়া হয়, তাই কাটা FOV-ও পুরো বৃত্তে ফেরে।
   - নমুনায় fit-এর গড় পার্থক্য ০.২৩ px।
   - তারপর বৃত্তকে কেন্দ্র করে **একই scale-এর বর্গ crop**, আর ফ্রেমের বাইরের অংশ কালো রাখা হয়।
   - ফল: সব camera-র চোখ এক মাপে, পুরো retina, disc কখনো কাটা পড়ে না।
2. **নতুন optic disc locator** (`locate_disc`):
   - "ছোট গোল উজ্জ্বল দাগ" খোঁজা হয় (DoG filter), যাতে flare বা ধোঁয়াশার মতো বড় উজ্জ্বল এলাকা বাদ পড়ে।
   - সাথে যেখানে বড় রক্তনালী মেশে (black-hat), আর চোখের পাশ (ফাইলের নামে left/right) থেকে একটা নরম prior।
   - ঠিক ধরার হার ~৮২% থেকে **~৯৫%** (৬০টা ছবিতে চোখে মিলিয়ে দেখা)।
3. **Cache-এ রং বদলানোর আগের ছবি:** রং আর contrast পরিবর্তন training-এর সময় CPU-তে হয়। তাই ৭,০০০টা আসল ছবি বারবার পড়তে হয় না, আর ablation-এ variant বদলানো সহজ।
4. **তিনটা photometric variant:**
   - `clahe`: V1–V4-এর মতো
   - `norm_clahe`: প্রতিটা চোখের LAB mean আর std dataset-এর reference-এ আনা (Reinhard), তারপর CLAHE। উজ্জ্বলতা বাড়ানো ০.৫–২.৫ গুণে সীমিত, যাতে অন্ধকার ছবিতে noise না বাড়ে। Disc crop-এ পুরো চোখের মাপ ব্যবহার হয়।
   - `graham`: Kaggle DR 2015-এর পদ্ধতি। শুধু retina-র pixel দিয়ে গড় নেওয়া হয়, তাই FOV-এর কিনারায় বা কাটা জায়গায় উজ্জ্বল দাগ পড়ে না (পরীক্ষা করা)।
5. **Orientation normalization:** বাম চোখের ছবি (আর তার disc crop) আয়নার মতো উল্টানো হয়। তখন augmentation আর TTA থেকে বাম-ডান উল্টানো বাদ, TTA = নিজে + উপর-নিচ উল্টানো। Grad-CAM-এর disc box-ও সেই অনুযায়ী উল্টে নেওয়া হয়।
6. **Preprocessing ablation** (Part 9-এর শুরুতে, ~২৫ মিনিট):
   - ResNet50, 256 px, ৬ epoch। Fold 0-এর training data-য় train, **validation data-য়** তুলনা। Test fold ছোঁয়া হয় না।
   - ৪টা variant:
     - (১) V1–V4 (CLAHE, উল্টানো নেই)
     - (২) CLAHE + orientation
     - (৩) রং normalize + CLAHE + orientation
     - (৪) Graham + orientation
   - সবচেয়ে বেশি validation macro-F1 যেটার, সেটা সব model-এ ব্যবহার হয়।
   - ফল যায় `tab_prep_ablation`-এ, যেটা paper-এর জন্য একটা preprocessing ablation table। Resume করলে `prep_ablation.json` থেকে একই বাছাই আবার ব্যবহার হয়।
7. **নতুন figure আর তথ্য:**
   - `fig_fov_normalisation`: কাটা-FOV-এ পুরনো বনাম নতুন crop।
   - `fig_preprocessing`: fit করা বৃত্ত, ৩টা variant আর disc ROI।
   - audit-এ `fov_truncated_images`; macro `\FOVTruncatedImages`, `\PrepPhotometric`, `\PrepOrientation`।

### ১৩.৩ সৎ সীমা

- **Result-এ কতটা লাভ, এখনো মাপা হয়নি।**
  - Geometry আর disc-এর উন্নতি সরাসরি মাপা (crop আর disc-এর নির্ভুলতা)।
  - রং বা orientation কোনটা ভালো, সেটা Kaggle run-এর ablation-ই বলবে।
- **Ablation-এর বাছাই কিছুটা noisy:** একটা fold-এর validation-এ, glaucoma মাত্র ~২০টা ছবি। দুটো variant কাছাকাছি হলে যেকোনোটাই যুক্তিসঙ্গত। Paper-এ table-টা "validation-ভিত্তিক বাছাই" হিসেবেই দেখাতে হবে।
- **Disc locator এখনো heuristic।** Pathologic myopia (বড় atrophy) আর ঘন cataract-এ disc প্রায়ই দেখাই যায় না। এসব class মূলত পুরো ছবির model দিয়ে চেনা হয়।

---

## ১৪. Augmentation, class balancing আর model: পুনর্যাচাই

Notebook-এর training code (Part 8) আবার লাইন ধরে পড়ে প্রচলিত নিয়মের সাথে মিলিয়েছি। যা ঠিক ছিল সেটা "✓", যা বদলাতে হয়েছে সেটা "**ঠিক করা হয়েছে**"।

### ১৪.১ Augmentation

| বিষয় | এখন যা হয় | অবস্থা |
|---|---|---|
| কোথায় | শুধু training batch-এ। Validation আর test-এ কখনো না (inference-এ শুধু flip TTA)। | ✓ |
| Class-নিরপেক্ষ | প্রতিটা class-এ একই নিয়ম, কোনো রোগের লক্ষণ তৈরি বা মুছে ফেলে না | ✓ (blur ঠিক করার পর) |
| জ্যামিতি | rotation ±২৫°, scale ±৮%, shift ৪% (disc-এ ৮%, disc খোঁজার ভুল সামলাতে), উপর-নিচ উল্টানো; বাম-ডান উল্টানো শুধু orientation normalization বন্ধ থাকলে। সব একবারের resampling-এ, তাই বারবার interpolation-এর ঝাপসাভাব নেই। | ✓ |
| রং | brightness আর contrast ±২৫%, saturation ±২০%, hue ±০.০২ (p 0.8) | ✓ |
| Blur | আগে p 0.2, σ ≤ 1.2 px; **এখন p 0.1, σ ≤ 0.6 px** | **ঠিক করা হয়েছে।** 384 px-এ microaneurysm বা ছোট hemorrhage ২–৪ px চওড়া, σ 1.2 সেগুলো মুছে দেয় আর ছবিকে cataract-এর মতো ঝাপসা করে। ফলে আগে augmentation class-এর প্রমাণই বদলে দিত। |
| Noise | p 0.2, sd 0.02 | ✓ |
| MixUp | α 0.2, p 0.5 | ✓ |
| কাজের ধরন | GPU-তে, প্রতি ছবিতে আলাদা random মান; CPU শুধু preprocessing করে | ✓ |

### ১৪.২ Class balancing

| ধাপ | এখন যা হয় | অবস্থা |
|---|---|---|
| কখন | Split-এর **পরে**, শুধু training fold-এ, প্রতি epoch-এ নতুন করে। কোনো ছবি disk-এ কপি বা কৃত্রিমভাবে তৈরি হয় না, তাই train আর test-এর মধ্যে leakage নেই। | ✓ |
| Sampling | Square-root balanced sampling (1/√n সম্ভাবনায়, replacement সহ)। Glaucoma-র একটা ছবি epoch-এ ~২ বার আসে, প্রতিবার আলাদা augmentation-এ। | ✓ |
| Loss weight | 1/√n | ✓ (sampling আর weight মিলিয়ে সব class-এর মোট ওজন সমান) |
| Label smoothing | **ঠিক করা হয়েছে।** PyTorch-এর `CrossEntropyLoss(weight, label_smoothing)` smoothing-এর ভাগ সব class-এ সমান না দিয়ে class weight অনুযায়ী দেয়। সংখ্যায় যাচাই করেছি: একটা normal ছবির জন্য loss-এর সর্বোত্তম prediction হয়ে যেত P(Normal)=0.80, বাকিটা glaucoma, myopia আর cataract-এর দিকে, যেখানে হওয়া উচিত 0.92 আর বাকি প্রতিটায় 0.02। নিজস্ব `soft_ce` দিয়ে এখন ঠিক 0.92 / 0.02। Normal→Glaucoma ধরনের false positive এতে কমার কথা, যেটা reviewer-এর মূল আপত্তি ছিল। | **ঠিক করা হয়েছে** |
| Early stopping | Balanced validation NLL (training-এর objective-এর সাথে মিল রেখে) | ✓ |
| শেষ সিদ্ধান্ত | Post-hoc logit adjustment, prior power 0.5 আগে থেকে ঠিক করা; stacker-এ একই prior; temperature scaling। তিনটা prior-এর ফল আলাদা table-এ। | ✓ |
| Report | Balanced accuracy, macro-F1, প্রতি class-এর sensitivity আর specificity, confusion matrix | ✓ |

### ১৪.৩ Optimizer

- **AdamW, cosine schedule আর warm-up, layer-wise LR decay (DINOv2-এ 0.75), EMA, gradient clipping, mixed precision:** ✓
- **Weight decay: ঠিক করা হয়েছে।** আগে normalization layer-এর weight, bias আর ViT-এর position/class token-এও weight decay পড়ত। এখন প্রচলিত নিয়ম অনুযায়ী এগুলো বাদ।
  - যাচাই: ResNet50-এ ১০৭টা আর DINOv2-এ ১২৬টা tensor-এ decay বাদ পড়েছে, প্রতিটা parameter ঠিক একবার গোনা হয়েছে।

### ১৪.৪ কোন কোন model ব্যবহার হচ্ছে

| ভূমিকা | Model | Input | Pre-training | কেন |
|---|---|---|---|---|
| পুরো fundus, CNN | **ResNet50** (torchvision) | 384 px | ImageNet-1k | T4-এ দ্রুত, শক্তিশালী CNN। V2-এ 83.97%। |
| পুরো fundus, Transformer | **DINOv2 ViT-B/14** (timm `vit_base_patch14_dinov2.lvd142m`) | 336 px | Self-supervised, ১৪২ মিলিয়ন ছবি | V2-এ সেরা single-এর কাছাকাছি (84.64%) |
| Optic disc expert | **ResNet50** | 224 px disc crop, আসল বড় ছবি থেকে | ImageNet-1k | Glaucoma-র cup আর rim |
| **Proposed (DA-MVE)** | তিনটার stacking (validation-এ fit করা logistic regression) + temperature scaling | – | – | তিন view-এর তথ্য মেলায় |
| তুলনা | প্রতিটা single model, validation-এ বাছা best single, soft vote (top-2 আর সব), শুধু global model-এর stacking | – | – | Ensemble কতটা যোগ করে তা দেখাতে |
| সহায়ক run | ResNet50 @256 (preprocessing ablation), ResNet50 @384 random split (protocol তুলনা) | – | – | – |

**কেন অন্যগুলো নয়:**
- **ConvNeXt-T:** বড় resolution-এ T4-এ ধীর। V2-তে 224 px-এই ২৪ মিনিট লাগত, আর accuracy DINOv2-এর সমান।
- **DINOv3:** V2-এ প্রতিটা fold-এ সবার শেষে ছিল।
- **EfficientNet-B4 আর Inception-ResNet-v2:** ধীর।
- **RETFound:** access চাইতে হয়, আর ViT-L T4-এর জন্য খুব ভারী।
- **Model বদলাতে চাইলে:** Part 2-এর `global_models` আর `disc_models` তালিকায় নাম বদলালেই হবে।
