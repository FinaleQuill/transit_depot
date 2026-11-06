# Geometry review: frozen preliminary candidates

Originals are full frame. Cyan marks the original boundary; red marks the CSV normalized candidate.
The edge panel is display-only full-decode Pillow FIND_EDGES, not Canny or a pixel-identical replay of audit JPEG draft. Red tinted 1:1 strips show removed source pixels.
CSV pixel coordinates and their difference from normalized coordinates are printed on each image.
These are visual review candidates, not medical safety judgments.

NO CUT / full-frame fallback means the frozen CSV proposes no removal; such images remain useful for comparison.

- `train` CSV SHA256: `7b2203192028e5539601019281bbb44fabc82b365486f6154d7b4971a8a1aca5`
- `valid` CSV SHA256: `77674dd77dd5e204fa1d6cd5f3a9353d92566a149e2401365afa017d899a84db`
- Renderer SHA256: `4fc950b0f8cc0f1f9bf5cf935396869ebcae6b3191cbcca60c6e88199cef1404`

## train / AP / train:008532

- Reason: smallest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2597, 2531]; SHA256: `918adb7d39c4cfd80c9bcda53bba2af98d5fab353dd61bd889667c940275c5d4`
- Normalized box: `[0.0, 0.0, 1.0, 0.9979959919839679]`
- CSV original pixel box: `[0, 0, 2597, 2526]`; error px: `[0.0, 0.0, 0.0, 0.072]`
- [Review image](train/AP/train_AP_01_train_008532.jpg)
- [bottom 1:1 strip](train/AP/train_AP_01_train_008532_bottom_1to1.png) / [raw pixels](train/AP/train_AP_01_train_008532_bottom_raw_1to1.png)

## train / AP / train:024667

- Reason: median preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2215, 2143]; SHA256: `0bca3f3c43e421b0eacee72a2e2655fa18cde5b570289a6927e7d053b8f35494`
- Normalized box: `[0.0, 0.0, 1.0, 0.9919191919191919]`
- CSV original pixel box: `[0, 0, 2215, 2124]`; error px: `[0.0, 0.0, 0.0, -1.683]`
- [Review image](train/AP/train_AP_02_train_024667.jpg)
- [bottom 1:1 strip](train/AP/train_AP_02_train_024667_bottom_1to1.png) / [raw pixels](train/AP/train_AP_02_train_024667_bottom_raw_1to1.png)

## train / AP / train:019013

- Reason: largest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2539, 3050]; SHA256: `a6bc4c3676ee4f1de80d5801bb801477074e5fcf5131abf35788d498a396765c`
- Normalized box: `[0.0, 0.0, 0.9460093896713615, 0.943359375]`
- CSV original pixel box: `[0, 0, 2401, 2877]`; error px: `[0.0, 0.0, -0.918, -0.246]`
- [Review image](train/AP/train_AP_03_train_019013.jpg)
- [right 1:1 strip](train/AP/train_AP_03_train_019013_right_1to1.png) / [raw pixels](train/AP/train_AP_03_train_019013_right_raw_1to1.png)
- [bottom 1:1 strip](train/AP/train_AP_03_train_019013_bottom_1to1.png) / [raw pixels](train/AP/train_AP_03_train_019013_bottom_raw_1to1.png)

## train / AP / train:006758

- Reason: widest black border
- Frame status: NO CUT / full-frame fallback
- Original: [3056, 2544]; SHA256: `f6edda4534b6fc0266cde6529c864469939f005e4bc1c80684560a62816e6508`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 3056, 2543]`; error px: `[0.0, 0.0, 0.0, -1.0]`
- [Review image](train/AP/train_AP_04_train_006758.jpg)

## train / AP / train:005642

- Reason: highest text-like proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2539, 3050]; SHA256: `9134f0aa54629b39765c680ae49737021d7a4031bdf08ddd4fa79231812e5b8c`
- Normalized box: `[0.0, 0.0, 1.0, 0.951171875]`
- CSV original pixel box: `[0, 0, 2538, 2901]`; error px: `[0.0, 0.0, -1.0, -0.074]`
- [Review image](train/AP/train_AP_05_train_005642.jpg)
- [bottom 1:1 strip](train/AP/train_AP_05_train_005642_bottom_1to1.png) / [raw pixels](train/AP/train_AP_05_train_005642_bottom_raw_1to1.png)

## train / AP / train:009463

- Reason: highest irregularity proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [3056, 2544]; SHA256: `c55f9cfef3ccab766efc3d988122dd82d02f9f862ef5a7597a140eb1e19bcaa1`
- Normalized box: `[0.0, 0.0, 0.94921875, 1.0]`
- CSV original pixel box: `[0, 0, 2901, 2543]`; error px: `[0.0, 0.0, 0.188, -1.0]`
- [Review image](train/AP/train_AP_06_train_009463.jpg)
- [right 1:1 strip](train/AP/train_AP_06_train_009463_right_1to1.png) / [raw pixels](train/AP/train_AP_06_train_009463_right_raw_1to1.png)

## train / AP / train:009704

- Reason: bright outer-edge risk proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2539, 2553]; SHA256: `63b55d58e105ad0025c195dbed17b02044fabaf8479a1cbecff05614f74350ed`
- Normalized box: `[0.0, 0.0, 1.0, 0.99609375]`
- CSV original pixel box: `[0, 0, 2538, 2543]`; error px: `[0.0, 0.0, -1.0, -0.027]`
- [Review image](train/AP/train_AP_07_train_009704.jpg)
- [bottom 1:1 strip](train/AP/train_AP_07_train_009704_bottom_1to1.png) / [raw pixels](train/AP/train_AP_07_train_009704_bottom_raw_1to1.png)

## train / AP / train:028253

- Reason: edge consistency risk proxy
- Frame status: NO CUT / full-frame fallback
- Original: [2949, 2539]; SHA256: `fe359df3aa920690f04df71546e19084554f4260fc2fc7498ff464ad97458c23`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2949, 2540]`; error px: `[0.0, 0.0, 0.0, 1.0]`
- [Review image](train/AP/train_AP_08_train_028253.jpg)

## train / AP / train:025640

- Reason: largest top cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `fa7cab6500bdf4d9972760b4a618a9e392b516530e7b1f84733586dc36623d50`
- Normalized box: `[0.0, 0.0390625, 1.0, 1.0]`
- CSV original pixel box: `[0, 119, 2543, 3056]`; error px: `[0.0, -0.375, -1.0, 0.0]`
- [Review image](train/AP/train_AP_09_train_025640.jpg)
- [top 1:1 strip](train/AP/train_AP_09_train_025640_top_1to1.png) / [raw pixels](train/AP/train_AP_09_train_025640_top_raw_1to1.png)

## train / AP / train:008222

- Reason: largest bottom cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `aea43a5d19684f8de5e9a04ce98d8ff7f6ce9b8118992dd57963bb6765525c97`
- Normalized box: `[0.0, 0.0, 1.0, 0.94140625]`
- CSV original pixel box: `[0, 0, 2543, 2877]`; error px: `[0.0, 0.0, -1.0, 0.062]`
- [Review image](train/AP/train_AP_10_train_008222.jpg)
- [bottom 1:1 strip](train/AP/train_AP_10_train_008222_bottom_1to1.png) / [raw pixels](train/AP/train_AP_10_train_008222_bottom_raw_1to1.png)

## train / AP / train:012582

- Reason: largest left cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `c0c874318a52a357e683947aa884500e667cf4714f61be263b300577e11e75c1`
- Normalized box: `[0.05868544600938967, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[149, 0, 2543, 3056]`; error px: `[-0.296, 0.0, -1.0, 0.0]`
- [Review image](train/AP/train_AP_11_train_012582.jpg)
- [left 1:1 strip](train/AP/train_AP_11_train_012582_left_1to1.png) / [raw pixels](train/AP/train_AP_11_train_012582_left_raw_1to1.png)

## train / AP / train:004200

- Reason: largest right cut
- Frame status: preliminary candidate with cut
- Original: [3056, 2544]; SHA256: `40d119f01d818ba4b16f30b00f06c8e1393e4287ce2acb8602c67f70d8a4a009`
- Normalized box: `[0.0, 0.0, 0.943359375, 1.0]`
- CSV original pixel box: `[0, 0, 2883, 2543]`; error px: `[0.0, 0.0, 0.094, -1.0]`
- [Review image](train/AP/train_AP_12_train_004200.jpg)
- [right 1:1 strip](train/AP/train_AP_12_train_004200_right_1to1.png) / [raw pixels](train/AP/train_AP_12_train_004200_right_raw_1to1.png)

## train / PA / train:020362

- Reason: smallest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 2981]; SHA256: `2ade31896e0fe2287ebe53ba76ec1723869a29496bf3d71946cd81487fe7a4c3`
- Normalized box: `[0.002288329519450801, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[6, 0, 2544, 2981]`; error px: `[0.178, 0.0, 0.0, 0.0]`
- [Review image](train/PA/train_PA_01_train_020362.jpg)
- [left 1:1 strip](train/PA/train_PA_01_train_020362_left_1to1.png) / [raw pixels](train/PA/train_PA_01_train_020362_left_raw_1to1.png)

## train / PA / train:027181

- Reason: median preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `179d9ac87307c9468610cfb7226c9f4e3226a1edeb46fa929408f5be96dfdccc`
- Normalized box: `[0.0, 0.0, 1.0, 0.986328125]`
- CSV original pixel box: `[0, 0, 2543, 3014]`; error px: `[0.0, 0.0, -1.0, -0.219]`
- [Review image](train/PA/train_PA_02_train_027181.jpg)
- [bottom 1:1 strip](train/PA/train_PA_02_train_027181_bottom_1to1.png) / [raw pixels](train/PA/train_PA_02_train_027181_bottom_raw_1to1.png)

## train / PA / train:015295

- Reason: largest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `947eeca03a4803384a61476c97609007423c031a4f5593bc84efbb8d18146aa8`
- Normalized box: `[0.035211267605633804, 0.0, 1.0, 0.94140625]`
- CSV original pixel box: `[90, 0, 2543, 2877]`; error px: `[0.423, 0.0, -1.0, 0.062]`
- [Review image](train/PA/train_PA_03_train_015295.jpg)
- [left 1:1 strip](train/PA/train_PA_03_train_015295_left_1to1.png) / [raw pixels](train/PA/train_PA_03_train_015295_left_raw_1to1.png)
- [bottom 1:1 strip](train/PA/train_PA_03_train_015295_bottom_1to1.png) / [raw pixels](train/PA/train_PA_03_train_015295_bottom_raw_1to1.png)

## train / PA / train:010083

- Reason: widest black border
- Frame status: NO CUT / full-frame fallback
- Original: [2544, 3056]; SHA256: `34ec8a7d3db686ce846e2a29bef049985ab8c83267daaf06b0886cba996d2d32`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2543, 3056]`; error px: `[0.0, 0.0, -1.0, 0.0]`
- [Review image](train/PA/train_PA_04_train_010083.jpg)

## train / PA / train:029718

- Reason: highest text-like proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [3056, 2544]; SHA256: `68f1a0c706a54175093ebbd06d37e7f9d61a5cba9834f6d130ebff58329c180f`
- Normalized box: `[0.0, 0.0, 1.0, 0.9906103286384976]`
- CSV original pixel box: `[0, 0, 3056, 2519]`; error px: `[0.0, 0.0, 0.0, -1.113]`
- [Review image](train/PA/train_PA_05_train_029718.jpg)
- [bottom 1:1 strip](train/PA/train_PA_05_train_029718_bottom_1to1.png) / [raw pixels](train/PA/train_PA_05_train_029718_bottom_raw_1to1.png)

## train / PA / train:001958

- Reason: highest irregularity proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2022, 2022]; SHA256: `06c33ea5c9f284f3780f21c4de701f0dd9b384e605a23523d3e57f17822a1da1`
- Normalized box: `[0.0, 0.0, 1.0, 0.97265625]`
- CSV original pixel box: `[0, 0, 2022, 1967]`; error px: `[0.0, 0.0, 0.0, 0.289]`
- [Review image](train/PA/train_PA_06_train_001958.jpg)
- [bottom 1:1 strip](train/PA/train_PA_06_train_001958_bottom_1to1.png) / [raw pixels](train/PA/train_PA_06_train_001958_bottom_raw_1to1.png)

## train / PA / train:020336

- Reason: bright outer-edge risk proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2544, 2622]; SHA256: `aacbfed771ee19169470e668cc756eb967978dd67185ca62ab8e89b59ec66a11`
- Normalized box: `[0.0, 0.0, 1.0, 0.99609375]`
- CSV original pixel box: `[0, 0, 2545, 2612]`; error px: `[0.0, 0.0, 1.0, 0.242]`
- [Review image](train/PA/train_PA_07_train_020336.jpg)
- [bottom 1:1 strip](train/PA/train_PA_07_train_020336_bottom_1to1.png) / [raw pixels](train/PA/train_PA_07_train_020336_bottom_raw_1to1.png)

## train / PA / train:029194

- Reason: edge consistency risk proxy
- Frame status: NO CUT / full-frame fallback
- Original: [2021, 2021]; SHA256: `acf5f15e2caf47471e7f58fe76e083259079eab8234aa9eece2dd2240ff2e287`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2021, 2021]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](train/PA/train_PA_08_train_029194.jpg)

## train / PA / train:029183

- Reason: largest top cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `c37c128f38f3389a77d2945dc96dfa475d9729a4d49c9b7cdb1163eb499577e9`
- Normalized box: `[0.0, 0.041015625, 1.0, 1.0]`
- CSV original pixel box: `[0, 125, 2543, 3056]`; error px: `[0.0, -0.344, -1.0, 0.0]`
- [Review image](train/PA/train_PA_09_train_029183.jpg)
- [top 1:1 strip](train/PA/train_PA_09_train_029183_top_1to1.png) / [raw pixels](train/PA/train_PA_09_train_029183_top_raw_1to1.png)

## train / PA / train:003819

- Reason: largest bottom cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `7c3c88b10b8f9e98ea83ce1e892f6079844e9a18b454bf62cfd3b9fc80aea144`
- Normalized box: `[0.0, 0.0, 1.0, 0.94140625]`
- CSV original pixel box: `[0, 0, 2543, 2877]`; error px: `[0.0, 0.0, -1.0, 0.062]`
- [Review image](train/PA/train_PA_10_train_003819.jpg)
- [bottom 1:1 strip](train/PA/train_PA_10_train_003819_bottom_1to1.png) / [raw pixels](train/PA/train_PA_10_train_003819_bottom_raw_1to1.png)

## train / PA / train:021256

- Reason: largest left cut
- Frame status: preliminary candidate with cut
- Original: [3056, 2544]; SHA256: `8d03d3acfdfdc331309f68b5cfece21e824b56adbaee9c347018e76fa27d63ad`
- Normalized box: `[0.0546875, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[167, 0, 3056, 2543]`; error px: `[-0.125, 0.0, 0.0, -1.0]`
- [Review image](train/PA/train_PA_11_train_021256.jpg)
- [left 1:1 strip](train/PA/train_PA_11_train_021256_left_1to1.png) / [raw pixels](train/PA/train_PA_11_train_021256_left_raw_1to1.png)

## train / PA / train:028628

- Reason: largest right cut
- Frame status: preliminary candidate with cut
- Original: [3056, 2544]; SHA256: `c9e01f268d21d798e36125db5b4b0a311cf211290a24a70f9e8d07c39cd33fcd`
- Normalized box: `[0.0, 0.0, 0.953125, 1.0]`
- CSV original pixel box: `[0, 0, 2913, 2543]`; error px: `[0.0, 0.0, 0.25, -1.0]`
- [Review image](train/PA/train_PA_12_train_028628.jpg)
- [right 1:1 strip](train/PA/train_PA_12_train_028628_right_1to1.png) / [raw pixels](train/PA/train_PA_12_train_028628_right_raw_1to1.png)

## train / Lateral / train:000224

- Reason: smallest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `2725eb67fe4079415a05c438c373761c09f89c79d38fd8716ce43dd87c6c62fd`
- Normalized box: `[0.002347417840375587, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[6, 0, 2543, 3056]`; error px: `[0.028, 0.0, -1.0, 0.0]`
- [Review image](train/Lateral/train_Lateral_01_train_000224.jpg)
- [left 1:1 strip](train/Lateral/train_Lateral_01_train_000224_left_1to1.png) / [raw pixels](train/Lateral/train_Lateral_01_train_000224_left_raw_1to1.png)

## train / Lateral / train:007760

- Reason: median preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `fbad418f24225a3ab864cf49bde736e9cbe676c70f3a1bc6ca6dc765816807a0`
- Normalized box: `[0.0, 0.0234375, 1.0, 1.0]`
- CSV original pixel box: `[0, 72, 2543, 3056]`; error px: `[0.0, 0.375, -1.0, 0.0]`
- [Review image](train/Lateral/train_Lateral_02_train_007760.jpg)
- [top 1:1 strip](train/Lateral/train_Lateral_02_train_007760_top_1to1.png) / [raw pixels](train/Lateral/train_Lateral_02_train_007760_top_raw_1to1.png)

## train / Lateral / train:026622

- Reason: largest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `6022e79990f5e81dcb6f27c759356cee88ec6d3338e7f80f543b193073745d91`
- Normalized box: `[0.0, 0.0, 1.0, 0.94140625]`
- CSV original pixel box: `[0, 0, 2543, 2877]`; error px: `[0.0, 0.0, -1.0, 0.062]`
- [Review image](train/Lateral/train_Lateral_03_train_026622.jpg)
- [bottom 1:1 strip](train/Lateral/train_Lateral_03_train_026622_bottom_1to1.png) / [raw pixels](train/Lateral/train_Lateral_03_train_026622_bottom_raw_1to1.png)

## train / Lateral / train:009682

- Reason: widest black border
- Frame status: NO CUT / full-frame fallback
- Original: [2544, 3056]; SHA256: `92b8177246c66a93621ae35e1024170b73c249640af50e0479be2fcc7fadc3a7`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2543, 3056]`; error px: `[0.0, 0.0, -1.0, 0.0]`
- [Review image](train/Lateral/train_Lateral_04_train_009682.jpg)

## train / Lateral / train:026449

- Reason: highest text-like proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `e43c0e666be13be2436859cddd1f23f38bedce81054c65ae2d1b59236ed07ac5`
- Normalized box: `[0.0, 0.0, 1.0, 0.9921875]`
- CSV original pixel box: `[0, 0, 2543, 3032]`; error px: `[0.0, 0.0, -1.0, -0.125]`
- [Review image](train/Lateral/train_Lateral_05_train_026449.jpg)
- [bottom 1:1 strip](train/Lateral/train_Lateral_05_train_026449_bottom_1to1.png) / [raw pixels](train/Lateral/train_Lateral_05_train_026449_bottom_raw_1to1.png)

## train / Lateral / train:007023

- Reason: highest irregularity proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2022, 2022]; SHA256: `f55463322885f9db95e64b5961bcb34729db195dfc435b3da7efda2c08e9a578`
- Normalized box: `[0.0, 0.0, 1.0, 0.970703125]`
- CSV original pixel box: `[0, 0, 2022, 1963]`; error px: `[0.0, 0.0, 0.0, 0.238]`
- [Review image](train/Lateral/train_Lateral_06_train_007023.jpg)
- [bottom 1:1 strip](train/Lateral/train_Lateral_06_train_007023_bottom_1to1.png) / [raw pixels](train/Lateral/train_Lateral_06_train_007023_bottom_raw_1to1.png)

## train / Lateral / train:012635

- Reason: bright outer-edge risk proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2540, 3056]; SHA256: `80faf8487e6b61f825df06351b86b22ea2b24f9bd46b4fe0ce1a371794c2a54d`
- Normalized box: `[0.002347417840375587, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[6, 0, 2543, 3056]`; error px: `[0.038, 0.0, 3.0, 0.0]`
- [Review image](train/Lateral/train_Lateral_07_train_012635.jpg)
- [left 1:1 strip](train/Lateral/train_Lateral_07_train_012635_left_1to1.png) / [raw pixels](train/Lateral/train_Lateral_07_train_012635_left_raw_1to1.png)

## train / Lateral / train:019043

- Reason: edge consistency risk proxy
- Frame status: NO CUT / full-frame fallback
- Original: [2022, 2022]; SHA256: `398c6cf35a75ec235517c224cb61b0d06b571ec2ca47760f085e924131a40cc4`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2022, 2022]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](train/Lateral/train_Lateral_08_train_019043.jpg)

## train / Lateral / train:019971

- Reason: largest top cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `ac5b651fab786618fc051e00550e40030524f9685177a1c208f7633a0a486b8c`
- Normalized box: `[0.0, 0.052734375, 1.0, 1.0]`
- CSV original pixel box: `[0, 161, 2543, 3056]`; error px: `[0.0, -0.156, -1.0, 0.0]`
- [Review image](train/Lateral/train_Lateral_09_train_019971.jpg)
- [top 1:1 strip](train/Lateral/train_Lateral_09_train_019971_top_1to1.png) / [raw pixels](train/Lateral/train_Lateral_09_train_019971_top_raw_1to1.png)

## train / Lateral / train:001939

- Reason: largest bottom cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `cac04b9279952e7363058a82d0497db84df3a740091547489b1407defd551a9a`
- Normalized box: `[0.0, 0.0, 1.0, 0.94140625]`
- CSV original pixel box: `[0, 0, 2543, 2877]`; error px: `[0.0, 0.0, -1.0, 0.062]`
- [Review image](train/Lateral/train_Lateral_10_train_001939.jpg)
- [bottom 1:1 strip](train/Lateral/train_Lateral_10_train_001939_bottom_1to1.png) / [raw pixels](train/Lateral/train_Lateral_10_train_001939_bottom_raw_1to1.png)

## train / Lateral / train:028651

- Reason: largest left cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `44c1c451e2cb45a0cf3bff68679371b9e1958e45165384558b843c2a8bf7c93f`
- Normalized box: `[0.04929577464788732, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[125, 0, 2543, 3056]`; error px: `[-0.408, 0.0, -1.0, 0.0]`
- [Review image](train/Lateral/train_Lateral_11_train_028651.jpg)
- [left 1:1 strip](train/Lateral/train_Lateral_11_train_028651_left_1to1.png) / [raw pixels](train/Lateral/train_Lateral_11_train_028651_left_raw_1to1.png)

## train / Lateral / train:016087

- Reason: largest right cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `8311f11a7de96ab2bbd08e32f699103a2dbce0591c31821e8d05c1449bf33047`
- Normalized box: `[0.0, 0.0, 0.960093896713615, 1.0]`
- CSV original pixel box: `[0, 0, 2441, 3056]`; error px: `[0.0, 0.0, -1.479, 0.0]`
- [Review image](train/Lateral/train_Lateral_12_train_016087.jpg)
- [right 1:1 strip](train/Lateral/train_Lateral_12_train_016087_right_1to1.png) / [raw pixels](train/Lateral/train_Lateral_12_train_016087_right_raw_1to1.png)

## train / Missing / train:002812

- Reason: smallest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [1760, 2140]; SHA256: `63752706b2635ba81fb4d5c463dfcd8ce21b0c9b0e6318dd914040d5ffb80978`
- Normalized box: `[0.0, 0.0, 1.0, 0.99609375]`
- CSV original pixel box: `[0, 0, 1760, 2132]`; error px: `[0.0, 0.0, 0.0, 0.359]`
- [Review image](train/Missing/train_Missing_01_train_002812.jpg)
- [bottom 1:1 strip](train/Missing/train_Missing_01_train_002812_bottom_1to1.png) / [raw pixels](train/Missing/train_Missing_01_train_002812_bottom_raw_1to1.png)

## train / Missing / train:013922

- Reason: median preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [1760, 2140]; SHA256: `378987482b3b4d7ec07cb3945f304df4aa06a1b70c263fc395e95a206f9267d6`
- Normalized box: `[0.0, 0.0, 1.0, 0.9765625]`
- CSV original pixel box: `[0, 0, 1760, 2090]`; error px: `[0.0, 0.0, 0.0, 0.156]`
- [Review image](train/Missing/train_Missing_02_train_013922.jpg)
- [bottom 1:1 strip](train/Missing/train_Missing_02_train_013922_bottom_1to1.png) / [raw pixels](train/Missing/train_Missing_02_train_013922_bottom_raw_1to1.png)

## train / Missing / train:013623

- Reason: largest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [1760, 2140]; SHA256: `e273d244cc1e6df158c0199e356152fcd12671922f944d111191b018a694526e`
- Normalized box: `[0.030878859857482184, 0.0, 1.0, 0.94921875]`
- CSV original pixel box: `[54, 0, 1760, 2031]`; error px: `[-0.347, 0.0, 0.0, -0.328]`
- [Review image](train/Missing/train_Missing_03_train_013623.jpg)
- [left 1:1 strip](train/Missing/train_Missing_03_train_013623_left_1to1.png) / [raw pixels](train/Missing/train_Missing_03_train_013623_left_raw_1to1.png)
- [bottom 1:1 strip](train/Missing/train_Missing_03_train_013623_bottom_1to1.png) / [raw pixels](train/Missing/train_Missing_03_train_013623_bottom_raw_1to1.png)

## train / Missing / train:022957

- Reason: widest black border
- Frame status: NO CUT / full-frame fallback
- Original: [1760, 2140]; SHA256: `dcc17bc7443c53a0009586cde7cd2b30545268c8e7ce0ae26e2b07b97f59f91d`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1760, 2140]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](train/Missing/train_Missing_04_train_022957.jpg)

## train / Missing / train:011662

- Reason: highest text-like proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2140, 1760]; SHA256: `e2559b1e8f5e0be479d1d618ed16da44e6ba675ac3717a8853198b85c8600256`
- Normalized box: `[0.03515625, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[75, 0, 2140, 1760]`; error px: `[-0.234, 0.0, 0.0, 0.0]`
- [Review image](train/Missing/train_Missing_05_train_011662.jpg)
- [left 1:1 strip](train/Missing/train_Missing_05_train_011662_left_1to1.png) / [raw pixels](train/Missing/train_Missing_05_train_011662_left_raw_1to1.png)

## train / Missing / train:003666

- Reason: highest irregularity proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [1760, 2140]; SHA256: `17f8a8772404b08ffe8fcf19484f70ba02c4bde4875748c7ca1ea89b54028410`
- Normalized box: `[0.0, 0.0, 1.0, 0.986328125]`
- CSV original pixel box: `[0, 0, 1760, 2111]`; error px: `[0.0, 0.0, 0.0, 0.258]`
- [Review image](train/Missing/train_Missing_06_train_003666.jpg)
- [bottom 1:1 strip](train/Missing/train_Missing_06_train_003666_bottom_1to1.png) / [raw pixels](train/Missing/train_Missing_06_train_003666_bottom_raw_1to1.png)

## train / Missing / train:009942

- Reason: bright outer-edge risk proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2140, 1760]; SHA256: `d1910207166d0f9bf1a9310f8fc62619af25cc8a017708220ee4228da172d776`
- Normalized box: `[0.0, 0.0, 0.98046875, 1.0]`
- CSV original pixel box: `[0, 0, 2098, 1760]`; error px: `[0.0, 0.0, -0.203, 0.0]`
- [Review image](train/Missing/train_Missing_07_train_009942.jpg)
- [right 1:1 strip](train/Missing/train_Missing_07_train_009942_right_1to1.png) / [raw pixels](train/Missing/train_Missing_07_train_009942_right_raw_1to1.png)

## train / Missing / train:026021

- Reason: edge consistency risk proxy
- Frame status: NO CUT / full-frame fallback
- Original: [2140, 1760]; SHA256: `1075fe9baaa27cf0dec507a02f47c9fbe2f04359feb4887f365f45a369a02919`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2140, 1760]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](train/Missing/train_Missing_08_train_026021.jpg)

## train / Missing / train:017685

- Reason: largest top cut
- Frame status: preliminary candidate with cut
- Original: [1760, 2140]; SHA256: `8454b3dffd6b1cd1e37e040c4c9278635d4aa098cb37cb44528db779feb6a35a`
- Normalized box: `[0.0, 0.02734375, 1.0, 1.0]`
- CSV original pixel box: `[0, 59, 1760, 2140]`; error px: `[0.0, 0.484, 0.0, 0.0]`
- [Review image](train/Missing/train_Missing_09_train_017685.jpg)
- [top 1:1 strip](train/Missing/train_Missing_09_train_017685_top_1to1.png) / [raw pixels](train/Missing/train_Missing_09_train_017685_top_raw_1to1.png)

## train / Missing / train:001234

- Reason: largest bottom cut
- Frame status: preliminary candidate with cut
- Original: [1760, 2140]; SHA256: `27a3724a2e9a36ecfc0ee5ba8a1d72a5bac218f409e4888d0a49d87974adee3e`
- Normalized box: `[0.0, 0.0, 1.0, 0.94140625]`
- CSV original pixel box: `[0, 0, 1760, 2015]`; error px: `[0.0, 0.0, 0.0, 0.391]`
- [Review image](train/Missing/train_Missing_10_train_001234.jpg)
- [bottom 1:1 strip](train/Missing/train_Missing_10_train_001234_bottom_1to1.png) / [raw pixels](train/Missing/train_Missing_10_train_001234_bottom_raw_1to1.png)

## train / Missing / train:003414

- Reason: largest left cut
- Frame status: preliminary candidate with cut
- Original: [1760, 2140]; SHA256: `5626786aeb318610683781c0450c6486bbaf129b1dbd87f636ef6e095432e055`
- Normalized box: `[0.05463182897862233, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[96, 0, 1760, 2140]`; error px: `[-0.152, 0.0, 0.0, 0.0]`
- [Review image](train/Missing/train_Missing_11_train_003414.jpg)
- [left 1:1 strip](train/Missing/train_Missing_11_train_003414_left_1to1.png) / [raw pixels](train/Missing/train_Missing_11_train_003414_left_raw_1to1.png)

## train / Missing / train:009446

- Reason: largest right cut
- Frame status: preliminary candidate with cut
- Original: [1760, 2140]; SHA256: `dcccddcbe9a0f1e4820845e280404f0471ee37ffb5cc3f8904d9c646a3651bfb`
- Normalized box: `[0.0, 0.0, 0.9572446555819477, 1.0]`
- CSV original pixel box: `[0, 0, 1684, 2140]`; error px: `[0.0, 0.0, -0.751, 0.0]`
- [Review image](train/Missing/train_Missing_12_train_009446.jpg)
- [right 1:1 strip](train/Missing/train_Missing_12_train_009446_right_1to1.png) / [raw pixels](train/Missing/train_Missing_12_train_009446_right_raw_1to1.png)

## valid / AP / valid:000198

- Reason: smallest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2726, 2279]; SHA256: `d86d13b988adbcbd86f456cc5897dabe61a9d08702330009efc6133a4d9e9645`
- Normalized box: `[0.0, 0.0, 1.0, 0.9976635514018691]`
- CSV original pixel box: `[0, 0, 2726, 2273]`; error px: `[0.0, 0.0, 0.0, -0.675]`
- [Review image](valid/AP/valid_AP_01_valid_000198.jpg)
- [bottom 1:1 strip](valid/AP/valid_AP_01_valid_000198_bottom_1to1.png) / [raw pixels](valid/AP/valid_AP_01_valid_000198_bottom_raw_1to1.png)

## valid / AP / valid:001087

- Reason: median preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2442, 2319]; SHA256: `b7a1bc9d3fb079cc14436aa4019f42943f9d8c840cd737369020df62f7ffc2cf`
- Normalized box: `[0.0, 0.0, 1.0, 0.9938271604938271]`
- CSV original pixel box: `[0, 0, 2442, 2304]`; error px: `[0.0, 0.0, 0.0, -0.685]`
- [Review image](valid/AP/valid_AP_02_valid_001087.jpg)
- [bottom 1:1 strip](valid/AP/valid_AP_02_valid_001087_bottom_1to1.png) / [raw pixels](valid/AP/valid_AP_02_valid_001087_bottom_raw_1to1.png)

## valid / AP / valid:001178

- Reason: largest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `291c644b4baf6f80505fbf3197f2387a65b0ac86b8944be97f533bbea59d6569`
- Normalized box: `[0.0, 0.0, 1.0, 0.94140625]`
- CSV original pixel box: `[0, 0, 2543, 2877]`; error px: `[0.0, 0.0, -1.0, 0.062]`
- [Review image](valid/AP/valid_AP_03_valid_001178.jpg)
- [bottom 1:1 strip](valid/AP/valid_AP_03_valid_001178_bottom_1to1.png) / [raw pixels](valid/AP/valid_AP_03_valid_001178_bottom_raw_1to1.png)

## valid / AP / valid:001034

- Reason: widest black border
- Frame status: NO CUT / full-frame fallback
- Original: [2544, 3056]; SHA256: `c069e45d57d50458dc91ddb51e47122d63476270d1530ed7a578f3f141e67564`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2543, 3056]`; error px: `[0.0, 0.0, -1.0, 0.0]`
- [Review image](valid/AP/valid_AP_04_valid_001034.jpg)

## valid / AP / valid:001119

- Reason: highest text-like proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2539, 2489]; SHA256: `3e085d35f92566438ff76f5387c3e008fee83f22105c7979545e9a38fdc8c73c`
- Normalized box: `[0.0, 0.0, 1.0, 0.9960159362549801]`
- CSV original pixel box: `[0, 0, 2539, 2479]`; error px: `[0.0, 0.0, 0.0, -0.084]`
- [Review image](valid/AP/valid_AP_05_valid_001119.jpg)
- [bottom 1:1 strip](valid/AP/valid_AP_05_valid_001119_bottom_1to1.png) / [raw pixels](valid/AP/valid_AP_05_valid_001119_bottom_raw_1to1.png)

## valid / AP / valid:000060

- Reason: highest irregularity proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [3056, 2544]; SHA256: `e80c2cde5c7507c9a830960f6a42fd92949ced6be5ec5fbff78c5e25ee66e03a`
- Normalized box: `[0.056640625, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[173, 0, 3056, 2543]`; error px: `[-0.094, 0.0, 0.0, -1.0]`
- [Review image](valid/AP/valid_AP_06_valid_000060.jpg)
- [left 1:1 strip](valid/AP/valid_AP_06_valid_000060_left_1to1.png) / [raw pixels](valid/AP/valid_AP_06_valid_000060_left_raw_1to1.png)

## valid / AP / valid:000687

- Reason: bright outer-edge risk proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2536, 2566]; SHA256: `965b3cd1c6b3f99299cda0b9a7e14fa8888f4120be8240461b63acb56c77cfb4`
- Normalized box: `[0.0, 0.0, 1.0, 0.99609375]`
- CSV original pixel box: `[0, 0, 2536, 2556]`; error px: `[0.0, 0.0, 0.0, 0.023]`
- [Review image](valid/AP/valid_AP_07_valid_000687.jpg)
- [bottom 1:1 strip](valid/AP/valid_AP_07_valid_000687_bottom_1to1.png) / [raw pixels](valid/AP/valid_AP_07_valid_000687_bottom_raw_1to1.png)

## valid / AP / valid:000098

- Reason: edge consistency risk proxy
- Frame status: NO CUT / full-frame fallback
- Original: [2431, 3050]; SHA256: `4c9f467977999a1a4454141e47df566aca154e8c94d99d4d23db5eb676ebc0a1`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2430, 3050]`; error px: `[0.0, 0.0, -1.0, 0.0]`
- [Review image](valid/AP/valid_AP_08_valid_000098.jpg)

## valid / AP / valid:000000

- Reason: largest top cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [2544, 3056]; SHA256: `9164a4f24dc0fdb3ac859424bb7179a4e5b307783dc7d3d739144e93c356a620`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2543, 3056]`; error px: `[0.0, 0.0, -1.0, 0.0]`
- [Review image](valid/AP/valid_AP_09_valid_000000.jpg)

## valid / AP / valid:000187

- Reason: largest bottom cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `2d8f8aa8c12bd705a9a11d831ae9e39a68e3888664465cd9138dedfc9351b3ab`
- Normalized box: `[0.0, 0.0, 1.0, 0.94140625]`
- CSV original pixel box: `[0, 0, 2543, 2877]`; error px: `[0.0, 0.0, -1.0, 0.062]`
- [Review image](valid/AP/valid_AP_10_valid_000187.jpg)
- [bottom 1:1 strip](valid/AP/valid_AP_10_valid_000187_bottom_1to1.png) / [raw pixels](valid/AP/valid_AP_10_valid_000187_bottom_raw_1to1.png)

## valid / AP / valid:000174

- Reason: largest left cut
- Frame status: preliminary candidate with cut
- Original: [3056, 2544]; SHA256: `27faa59a6fafbac2569f22a9d71a7f370d3369ea01d063260262e7fcc064232c`
- Normalized box: `[0.044921875, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[137, 0, 3056, 2543]`; error px: `[-0.281, 0.0, 0.0, -1.0]`
- [Review image](valid/AP/valid_AP_11_valid_000174.jpg)
- [left 1:1 strip](valid/AP/valid_AP_11_valid_000174_left_1to1.png) / [raw pixels](valid/AP/valid_AP_11_valid_000174_left_raw_1to1.png)

## valid / AP / valid:000062

- Reason: largest right cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `0a563784c1bcb221a156227b6c8817f951c7b840111f67a1f5136b2dd6d63548`
- Normalized box: `[0.0, 0.0, 0.9460093896713615, 1.0]`
- CSV original pixel box: `[0, 0, 2405, 3056]`; error px: `[0.0, 0.0, -1.648, 0.0]`
- [Review image](valid/AP/valid_AP_12_valid_000062.jpg)
- [right 1:1 strip](valid/AP/valid_AP_12_valid_000062_right_1to1.png) / [raw pixels](valid/AP/valid_AP_12_valid_000062_right_raw_1to1.png)

## valid / PA / valid:001335

- Reason: smallest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2536, 3056]; SHA256: `b64fad42f1fc6d2788704eb49477357114956001f912b7e3bfde89026d25fc4f`
- Normalized box: `[0.004705882352941176, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[12, 0, 2537, 3056]`; error px: `[0.066, 0.0, 1.0, 0.0]`
- [Review image](valid/PA/valid_PA_01_valid_001335.jpg)
- [left 1:1 strip](valid/PA/valid_PA_01_valid_001335_left_1to1.png) / [raw pixels](valid/PA/valid_PA_01_valid_001335_left_raw_1to1.png)

## valid / PA / valid:000701

- Reason: median preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `abcc296657aa2295657e5c2907c98761d76fb0f5ab37721be0e6f895b3b7d2ae`
- Normalized box: `[0.0, 0.0, 1.0, 0.9921875]`
- CSV original pixel box: `[0, 0, 2543, 3032]`; error px: `[0.0, 0.0, -1.0, -0.125]`
- [Review image](valid/PA/valid_PA_02_valid_000701.jpg)
- [bottom 1:1 strip](valid/PA/valid_PA_02_valid_000701_bottom_1to1.png) / [raw pixels](valid/PA/valid_PA_02_valid_000701_bottom_raw_1to1.png)

## valid / PA / valid:000756

- Reason: largest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `8df230ac1eb8975694eec367a10f896b5b8d200edfaf49304c07e72b04adf2bf`
- Normalized box: `[0.0, 0.0, 0.9577464788732394, 0.95703125]`
- CSV original pixel box: `[0, 0, 2435, 2925]`; error px: `[0.0, 0.0, -1.507, 0.312]`
- [Review image](valid/PA/valid_PA_03_valid_000756.jpg)
- [right 1:1 strip](valid/PA/valid_PA_03_valid_000756_right_1to1.png) / [raw pixels](valid/PA/valid_PA_03_valid_000756_right_raw_1to1.png)
- [bottom 1:1 strip](valid/PA/valid_PA_03_valid_000756_bottom_1to1.png) / [raw pixels](valid/PA/valid_PA_03_valid_000756_bottom_raw_1to1.png)

## valid / PA / valid:000267

- Reason: widest black border
- Frame status: NO CUT / full-frame fallback
- Original: [1736, 2022]; SHA256: `ba4e0cf4087cbf2c2034d5041430d81ab2ee9762f52c8292a77646d6fff1a67a`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1738, 2022]`; error px: `[0.0, 0.0, 2.0, 0.0]`
- [Review image](valid/PA/valid_PA_04_valid_000267.jpg)

## valid / PA / valid:001376

- Reason: highest text-like proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `964318545c4171251627256becd5fe696cd0c965f38845ac7800b628b673303f`
- Normalized box: `[0.007042253521126761, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[18, 0, 2543, 3056]`; error px: `[0.085, 0.0, -1.0, 0.0]`
- [Review image](valid/PA/valid_PA_05_valid_001376.jpg)
- [left 1:1 strip](valid/PA/valid_PA_05_valid_001376_left_1to1.png) / [raw pixels](valid/PA/valid_PA_05_valid_001376_left_raw_1to1.png)

## valid / PA / valid:000545

- Reason: highest irregularity proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [1556, 1760]; SHA256: `8a4694ffc6c6f7367b05160374fd5db0179847a34dcde42d8012813d76fc05df`
- Normalized box: `[0.0, 0.0, 1.0, 0.96484375]`
- CSV original pixel box: `[0, 0, 1557, 1698]`; error px: `[0.0, 0.0, 1.0, -0.125]`
- [Review image](valid/PA/valid_PA_06_valid_000545.jpg)
- [bottom 1:1 strip](valid/PA/valid_PA_06_valid_000545_bottom_1to1.png) / [raw pixels](valid/PA/valid_PA_06_valid_000545_bottom_raw_1to1.png)

## valid / PA / valid:001213

- Reason: bright outer-edge risk proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [3056, 2544]; SHA256: `b57a653d416b6bb1f7de2abcc74f7faa74ad25afded80f79470b62e4e9ee0217`
- Normalized box: `[0.0, 0.0, 1.0, 0.9929577464788732]`
- CSV original pixel box: `[0, 0, 3056, 2525]`; error px: `[0.0, 0.0, 0.0, -1.085]`
- [Review image](valid/PA/valid_PA_07_valid_001213.jpg)
- [bottom 1:1 strip](valid/PA/valid_PA_07_valid_001213_bottom_1to1.png) / [raw pixels](valid/PA/valid_PA_07_valid_001213_bottom_raw_1to1.png)

## valid / PA / valid:000276

- Reason: edge consistency risk proxy
- Frame status: NO CUT / full-frame fallback
- Original: [1767, 2022]; SHA256: `77be57b1f931711a24d2aa6ad600afb95a622fc15e8b9ac2ccb8b5b827fbbbaf`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1765, 2022]`; error px: `[0.0, 0.0, -2.0, 0.0]`
- [Review image](valid/PA/valid_PA_08_valid_000276.jpg)

## valid / PA / valid:000003

- Reason: largest top cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [2544, 3056]; SHA256: `1e7f22423fdae37b6207de54e7633117d1a1ff87c3ea769d1e775612e7bd4dbb`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2543, 3056]`; error px: `[0.0, 0.0, -1.0, 0.0]`
- [Review image](valid/PA/valid_PA_09_valid_000003.jpg)

## valid / PA / valid:000589

- Reason: largest bottom cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `840ab2ff468b6e783e9e1395d4bfad97aa854cd613e31e53fa14cdabcdc03bb1`
- Normalized box: `[0.0, 0.0, 1.0, 0.947265625]`
- CSV original pixel box: `[0, 0, 2543, 2895]`; error px: `[0.0, 0.0, -1.0, 0.156]`
- [Review image](valid/PA/valid_PA_10_valid_000589.jpg)
- [bottom 1:1 strip](valid/PA/valid_PA_10_valid_000589_bottom_1to1.png) / [raw pixels](valid/PA/valid_PA_10_valid_000589_bottom_raw_1to1.png)

## valid / PA / valid:001303

- Reason: largest left cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `f84687b187f07ac532d28b43b0a414ae97fd470ff5062ad90a435cd3866ee998`
- Normalized box: `[0.028169014084507043, 0.0, 1.0, 0.978515625]`
- CSV original pixel box: `[72, 0, 2543, 2990]`; error px: `[0.338, 0.0, -1.0, -0.344]`
- [Review image](valid/PA/valid_PA_11_valid_001303.jpg)
- [left 1:1 strip](valid/PA/valid_PA_11_valid_001303_left_1to1.png) / [raw pixels](valid/PA/valid_PA_11_valid_001303_left_raw_1to1.png)
- [bottom 1:1 strip](valid/PA/valid_PA_11_valid_001303_bottom_1to1.png) / [raw pixels](valid/PA/valid_PA_11_valid_001303_bottom_raw_1to1.png)

## valid / PA / valid:000005

- Reason: largest right cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [2544, 3056]; SHA256: `838909654f0dfa665a49a57c997d747f45d99db0556bddccb9d23564d37a440f`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2543, 3056]`; error px: `[0.0, 0.0, -1.0, 0.0]`
- [Review image](valid/PA/valid_PA_12_valid_000005.jpg)

## valid / Lateral / valid:000889

- Reason: smallest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `4181e1c501e37360dd2ee85d70ad8c24de7b8de5b74a045c5a6f0cc4ec99385f`
- Normalized box: `[0.0, 0.0, 1.0, 0.9921875]`
- CSV original pixel box: `[0, 0, 2543, 3032]`; error px: `[0.0, 0.0, -1.0, -0.125]`
- [Review image](valid/Lateral/valid_Lateral_01_valid_000889.jpg)
- [bottom 1:1 strip](valid/Lateral/valid_Lateral_01_valid_000889_bottom_1to1.png) / [raw pixels](valid/Lateral/valid_Lateral_01_valid_000889_bottom_raw_1to1.png)

## valid / Lateral / valid:000095

- Reason: median preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `6a809af19058ac5236c32555f0887da6852f205500b9ab5023d666ec01707a01`
- Normalized box: `[0.0, 0.021484375, 1.0, 1.0]`
- CSV original pixel box: `[0, 66, 2543, 3056]`; error px: `[0.0, 0.344, -1.0, 0.0]`
- [Review image](valid/Lateral/valid_Lateral_02_valid_000095.jpg)
- [top 1:1 strip](valid/Lateral/valid_Lateral_02_valid_000095_top_1to1.png) / [raw pixels](valid/Lateral/valid_Lateral_02_valid_000095_top_raw_1to1.png)

## valid / Lateral / valid:000757

- Reason: largest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `514e45436b19d65ce4d958db0cf5e712d4741c9c4f96724e2b14fdb5caed80ce`
- Normalized box: `[0.0, 0.0, 1.0, 0.9453125]`
- CSV original pixel box: `[0, 0, 2543, 2889]`; error px: `[0.0, 0.0, -1.0, 0.125]`
- [Review image](valid/Lateral/valid_Lateral_03_valid_000757.jpg)
- [bottom 1:1 strip](valid/Lateral/valid_Lateral_03_valid_000757_bottom_1to1.png) / [raw pixels](valid/Lateral/valid_Lateral_03_valid_000757_bottom_raw_1to1.png)

## valid / Lateral / valid:000196

- Reason: widest black border
- Frame status: NO CUT / full-frame fallback
- Original: [2544, 3056]; SHA256: `313c3745f2d2d28226620051ada9509f76788a735f019e527098a5b6a428b59e`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2543, 3056]`; error px: `[0.0, 0.0, -1.0, 0.0]`
- [Review image](valid/Lateral/valid_Lateral_04_valid_000196.jpg)

## valid / Lateral / valid:001133

- Reason: highest text-like proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `b77ec4e2ee0caf31bd6e8cbcdb23f281c380e981f914cfc15f15fd8dd29c8bda`
- Normalized box: `[0.0, 0.0, 1.0, 0.986328125]`
- CSV original pixel box: `[0, 0, 2543, 3014]`; error px: `[0.0, 0.0, -1.0, -0.219]`
- [Review image](valid/Lateral/valid_Lateral_05_valid_001133.jpg)
- [bottom 1:1 strip](valid/Lateral/valid_Lateral_05_valid_001133_bottom_1to1.png) / [raw pixels](valid/Lateral/valid_Lateral_05_valid_001133_bottom_raw_1to1.png)

## valid / Lateral / valid:000055

- Reason: highest irregularity proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [1736, 2022]; SHA256: `c9a8a191e871c31b51c17b9eb4f5ea4618574c62298e85032d0472e3ccfd9554`
- Normalized box: `[0.0, 0.0, 1.0, 0.951171875]`
- CSV original pixel box: `[0, 0, 1738, 1923]`; error px: `[0.0, 0.0, 2.0, -0.27]`
- [Review image](valid/Lateral/valid_Lateral_06_valid_000055.jpg)
- [bottom 1:1 strip](valid/Lateral/valid_Lateral_06_valid_000055_bottom_1to1.png) / [raw pixels](valid/Lateral/valid_Lateral_06_valid_000055_bottom_raw_1to1.png)

## valid / Lateral / valid:000949

- Reason: bright outer-edge risk proxy among preliminary candidates
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `e8da758b0e58b78cb0efa7b151f75b998e4a370e41110978b7f43f2a3df315ed`
- Normalized box: `[0.0, 0.009765625, 1.0, 1.0]`
- CSV original pixel box: `[0, 30, 2543, 3056]`; error px: `[0.0, 0.156, -1.0, 0.0]`
- [Review image](valid/Lateral/valid_Lateral_07_valid_000949.jpg)
- [top 1:1 strip](valid/Lateral/valid_Lateral_07_valid_000949_top_1to1.png) / [raw pixels](valid/Lateral/valid_Lateral_07_valid_000949_top_raw_1to1.png)

## valid / Lateral / valid:001045

- Reason: edge consistency risk proxy
- Frame status: NO CUT / full-frame fallback
- Original: [1750, 2022]; SHA256: `dce95b4cfb4f2d25f8646c63eba1e8412b407f71c9833de7547de9a482a335b5`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1750, 2022]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Lateral/valid_Lateral_08_valid_001045.jpg)

## valid / Lateral / valid:000001

- Reason: largest top cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [2544, 3056]; SHA256: `9ebb7c87404f81918e996feb4befe6e51227de24e95bf3a72195114e9d9bd0db`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2543, 3056]`; error px: `[0.0, 0.0, -1.0, 0.0]`
- [Review image](valid/Lateral/valid_Lateral_09_valid_000001.jpg)

## valid / Lateral / valid:000004

- Reason: largest bottom cut
- Frame status: preliminary candidate with cut
- Original: [2544, 3056]; SHA256: `bf17da1ca85ff3e7cb3bc31a73791e4290d74de9c34d5d869678d0b82cbe81de`
- Normalized box: `[0.0, 0.0, 1.0, 0.947265625]`
- CSV original pixel box: `[0, 0, 2543, 2895]`; error px: `[0.0, 0.0, -1.0, 0.156]`
- [Review image](valid/Lateral/valid_Lateral_10_valid_000004.jpg)
- [bottom 1:1 strip](valid/Lateral/valid_Lateral_10_valid_000004_bottom_1to1.png) / [raw pixels](valid/Lateral/valid_Lateral_10_valid_000004_bottom_raw_1to1.png)

## valid / Lateral / valid:000006

- Reason: largest left cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [2022, 2022]; SHA256: `41980c5051a7f1e871779b6d51d5044d0324a90fd01ab98659fce0819d8fe68b`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2022, 2022]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Lateral/valid_Lateral_11_valid_000006.jpg)

## valid / Lateral / valid:000007

- Reason: largest right cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [2022, 2022]; SHA256: `dcfa9e6be93f4aea333334c3cc47af16a93ab59f1919bdf59829e7c07d4ccf84`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2022, 2022]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Lateral/valid_Lateral_12_valid_000007.jpg)

## valid / Missing / valid:000309

- Reason: smallest preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [1760, 2140]; SHA256: `bd4656478ad0e14b80086df4297c2537ccc29ff28f8ffe1e44b6bdccf9dce6a4`
- Normalized box: `[0.0, 0.0, 1.0, 0.99609375]`
- CSV original pixel box: `[0, 0, 1760, 2132]`; error px: `[0.0, 0.0, 0.0, 0.359]`
- [Review image](valid/Missing/valid_Missing_01_valid_000309.jpg)
- [bottom 1:1 strip](valid/Missing/valid_Missing_01_valid_000309_bottom_1to1.png) / [raw pixels](valid/Missing/valid_Missing_01_valid_000309_bottom_raw_1to1.png)

## valid / Missing / valid:001355

- Reason: median preliminary candidate
- Frame status: preliminary candidate with cut
- Original: [3520, 4280]; SHA256: `da39a15ca1f8c5894b63c7910531f4458c00ed84df25097dc646ea8b271ee2de`
- Normalized box: `[0.0, 0.0, 1.0, 0.958984375]`
- CSV original pixel box: `[0, 0, 3519, 4104]`; error px: `[0.0, 0.0, -1.0, -0.453]`
- [Review image](valid/Missing/valid_Missing_02_valid_001355.jpg)
- [bottom 1:1 strip](valid/Missing/valid_Missing_02_valid_001355_bottom_1to1.png) / [raw pixels](valid/Missing/valid_Missing_02_valid_001355_bottom_raw_1to1.png)

## valid / Missing / valid:000029

- Reason: largest preliminary candidate (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [1760, 2140]; SHA256: `60e8645a3096e0a6a376f6a5221d8a19cdd544a64400750cfe4573be0693270a`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1760, 2140]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_03_valid_000029.jpg)

## valid / Missing / valid:001022

- Reason: widest black border
- Frame status: NO CUT / full-frame fallback
- Original: [1760, 2140]; SHA256: `3eead24514723b5c66b33aa0d5d53c095a96653949e8a6a159ad861302fa081a`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1760, 2140]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_04_valid_001022.jpg)

## valid / Missing / valid:000030

- Reason: highest text-like proxy among preliminary candidates (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [1760, 2140]; SHA256: `e83b458ead189653db08ed655a552d8c6f3329d62812af308ed38afe0d47e8e6`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1760, 2140]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_05_valid_000030.jpg)

## valid / Missing / valid:000167

- Reason: highest irregularity proxy among preliminary candidates (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [1760, 2140]; SHA256: `5c999ea920ad55528dcc77812f166d87ae09c443202738aec2a0a50409a0a0db`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1760, 2140]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_06_valid_000167.jpg)

## valid / Missing / valid:000168

- Reason: bright outer-edge risk proxy among preliminary candidates (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [1760, 2140]; SHA256: `48a0e9b7a3dc4b89a17311fa934c9d0c0c9e3df432a8169ca1f626f2bc557ac9`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1760, 2140]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_07_valid_000168.jpg)

## valid / Missing / valid:000920

- Reason: edge consistency risk proxy
- Frame status: NO CUT / full-frame fallback
- Original: [2140, 1760]; SHA256: `da7c66842d220ea92e46ffa9402e7623b8a6f62d7b00404c6df01bc5bec467e1`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2140, 1760]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_08_valid_000920.jpg)

## valid / Missing / valid:000188

- Reason: largest top cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [1760, 2140]; SHA256: `22fb8675bbbfe379b78c452544f66bf463696e020d0b72e3aaf76abec678cc1a`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1760, 2140]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_09_valid_000188.jpg)

## valid / Missing / valid:000189

- Reason: largest bottom cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [1760, 2140]; SHA256: `5485bb6047571c46d4e04e06c904d88cf5523571128159f5937977517ab73e3c`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1760, 2140]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_10_valid_000189.jpg)

## valid / Missing / valid:000296

- Reason: largest left cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [1760, 2140]; SHA256: `383c07d5b1d826691637d8343e99a979e2795267bfe5b99a306248fb7da0f13e`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 1760, 2140]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_11_valid_000296.jpg)

## valid / Missing / valid:000297

- Reason: largest right cut (fallback: no distinct matching row)
- Frame status: NO CUT / full-frame fallback
- Original: [2140, 1760]; SHA256: `062d2363522f644a6465f55bff93dddb49c0b13855fe01525acb23ae26ec9d1b`
- Normalized box: `[0.0, 0.0, 1.0, 1.0]`
- CSV original pixel box: `[0, 0, 2140, 1760]`; error px: `[0.0, 0.0, 0.0, 0.0]`
- [Review image](valid/Missing/valid_Missing_12_valid_000297.jpg)
