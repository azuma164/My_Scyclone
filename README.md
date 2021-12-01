# Scycloneを用いたvoice-conversion
- こちらは、東京大学工学部電気電子工学科・電子情報工学科の後期実験のうちの一つ「人工知能演習」の(成果物)[https://github.com/azuma164/MakeTranslationVideo_usingCycleGANVC2]で利用したものです
- 詳しい内容は[こちらのスライド](https://docs.google.com/presentation/d/19x3D4mF2lnlS_hGi-JnrOCVlNS9QgH8cxBwck3ulSCs/edit?usp=sharing)の27-34ページに書いてあります

## Original paper
[![Paper](http://img.shields.io/badge/paper-arxiv.2005.03334-B31B1B.svg)][paper]  
<!-- https://arxiv2bibtex.org/?q=2005.03334&format=bibtex -->
```
@misc{2005.03334,
Author = {Masaya Tanaka and Takashi Nose and Aoi Kanagaki and Ryohei Shimizu and Akira Ito},
Title = {Scyclone: High-Quality and Parallel-Data-Free Voice Conversion Using Spectrogram and Cycle-Consistent Adversarial Networks},
Year = {2020},
Eprint = {arXiv:2005.03334},
}
```

**[Original Paper's Demo](http://www.spcom.ecei.tohoku.ac.jp/nose/research/scyclone_202001/)**

## Difference from original research
- Datum length is based on a paper, not poster (G160/D128 in paper, G240/D240 in poster. Detail is in my summary blog)
- Use Automatic Mixed Precision training (FP32 training is also supported through `no_amp` flag)
