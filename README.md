# 2018年5月祭 応用物理系工学博覧会　アルゴリズム班展示
# Exhibition of Algorithm Section, the Exhibition of Engineering, 91st May Festival, the University of Tokyo
2018年5月19日(土), 20日(日)に開催された[5月祭](https://gogatsusai.jp/91/visitor/)の, 工学部応用物理系([物理工学科](http://www.ap.t.u-tokyo.ac.jp/), [計数工学科](https://www.keisu.t.u-tokyo.ac.jp/))「[工学展覧会](https://ap-phys.net/18/)」で展示したものです.
当日展示, 配布した資料は以下にアップロードしてあります.
- 展示ポスター([1](https://github.com/etttttte/mayfest2018/blob/master/algorithm_NN_poster_1.pdf), [2](https://github.com/etttttte/mayfest2018/blob/master/algorithm_NN_poster_2.pdf))
- [理論冊子](https://github.com/etttttte/mayfest2018/blob/master/algorithm_NN_theory.pdf)

This was a part of [“the Exhibition of Engineering”](https://ap-phys.net/18/) on [91st May Festival](https://gogatsusai.jp/91/visitor/) of the University of Tokyo which was held from May 19 to May 20 in 2018. This exhibition was hosted by [Department of Applied Physics](http://www.ap.t.u-tokyo.ac.jp/en/index_e.html) and [Department of Mathematical Engineering and Information Physics](https://www.keisu.t.u-tokyo.ac.jp/en/index/), [School of Engineering](https://www.t.u-tokyo.ac.jp/foee/index.html).
Handouts and posters are also uploaded (in Japanese, English version is in preparation):
-	posters([1](https://github.com/etttttte/mayfest2018/blob/master/algorithm_NN_poster_1.pdf), [2](https://github.com/etttttte/mayfest2018/blob/master/algorithm_NN_poster_2.pdf))
-	[handouts](https://github.com/etttttte/mayfest2018/blob/master/algorithm_NN_theory.pdf)


<img src="https://github.com/etttttte/mayfest2018/blob/master/fast-neural/images/akamon_balla.jpg" alt="" title="" width="400">   <img src="https://github.com/etttttte/mayfest2018/blob/master/fast-neural/images/akamon_dubuffet.jpg" alt="" title="" width="400">
<img src="https://github.com/etttttte/mayfest2018/blob/master/fast-neural/images/akamon_gogh.jpg" alt="" title="" width="400">   <img src="https://github.com/etttttte/mayfest2018/blob/master/fast-neural/images/akamon_munch.jpg" alt="" title="" width="400">


## Overview: Enchanted with Machine Learning
連日話題の人工知能. 魔法とうたわれ, 錬金術かのようにもてはやされていますが, 所詮は計算にすぎません.
5月祭ではCNNに着目し, その歴史的背景や基礎から, 応用例の画風変換まで展示しました. このレポジトリでは主に画風変換を紹介しています.
Python3, PyTorchで実装しています.

Artificial Intelligence is at the center of attention these days. Although it looks like magic or alchemy, it is just "computation." In the exhibition focused on CNN (convolutional neural network), visitors experienced for themselves from CNN’s historical background and theoretical basis to one of CNN applications "neural style transfer."
In this repository, you can find codes on neural style transfer implemented in Python3 and PyTorch. Learned models are also available.

### 画風変換(Neural Style Transfer)
画風変換とは, コンテンツ画像とスタイル画像が与えられた時, コンテンツの情報をできるだけ保ったままスタイル画像に画風を寄せた画像を出力するアルゴリズムです.
例えば, 赤門の写真をゴッホ風やムンク風に変換できます.
画像がVGGを通ることで, 段々スタイルに関する情報が捨てられ, コンテンツに関する情報のみが抽出されていくという性質を活用しています.

Neural style transfer is an algorithm that outputs an image that keeps the contents of a given contents image as much as possible but in the texture of a given style reference image. As you can see above, we can blend a picture of [Akamon Gate](https://www.u-tokyo.ac.jp/en/whyutokyo/hongo_hi_007.html) with “Gogh” texture or “Munch” texture. This algorithm is using the fact that the information of style is gradually lost and only information of contents is extracted by passing images through VGG.

#### Original
[Gatys et al. (2015)](https://arxiv.org/abs/1508.06576)によって提唱されたアルゴリズムです. PyTorch公式の[チュートリアル](http://pytorch.org/tutorials/advanced/neural_style_tutorial.html)にほぼ従っています.  `$python neural_style_transfer.py style_img content_img`  
で実行されます.
CPU環境で数十分～数時間かかります.

The original algorithm was introduced in [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576). This implementation is following the PyTorch [official tutorial](http://pytorch.org/tutorials/advanced/neural_style_tutorial.html) code. You can run it by:   
 `$python neural_style_transfer.py style_img content_img`  
It will take tens of minutes or hours with CPU.


#### Real-Time Style Transfer
オリジナルのアルゴリズムは任意のコンテンツ画像を任意のスタイル画像に変換できる一方, 毎回変換器を学習するため変換に時間がかかります.
この問題を解決したのが[Johnson et al. (2016)](http://arxiv.org/abs/1603.08155)です. スタイル画像ごとに変換器を事前に学習しておくことで, 画風変換自体のスピードが1000倍程度高速化しました.

[学習済みのモデル](https://github.com/etttttte/mayfest2018/tree/master/fast-neural/models)をダウンロードしたのち,  
`$python fast_style_change.py content_img model`  
で実行されます. content_imgは変換したいコンテンツ画像(.jpg)へのパス, model = {0:Balla, 1:Dubuffet, 2:Gogh, 3:Munch} です.

新たに変換器を学習したい場合は,  
`$python fast_style_train.py style_img`  
で学習できます. 
学習には[Microsoft COCO Dataset](http://cocodataset.org/#home)のtrain2014(~80k枚, 13GB)を用いました. 
GPU環境で2epochの学習に9-10時間かかります.

While the original algorithm allows any contents images to transform any texture, it is time-consuming as it learns transfer part every time. [Johnson et al. (2016)](http://arxiv.org/abs/1603.08155) solved this problem by training “Image Transform Net” beforehand. Although Image Transform Net must be built for each style images, style transfer itself increases 1000 times in speed.

When testing pretrained models, download [pretrained models](https://github.com/etttttte/mayfest2018/tree/master/fast-neural/models) and run:  
 `$python fast_style_change.py content_img model`  
‘content_img’ is a path to your content image (.jpg), and ‘model’ = {0:Balla, 1:Dubuffet, 2:Gogh, 3:Munch}.

When training a new model, run:  
`$python fast_style_train.py path_to_style_img`  
[Microsoft COCO Dataset](http://cocodataset.org/#home) train2014 (~80k images, 13GB) was used for training. It will take roughly 10 hours for 2 epoch training with GPU.


You can also refer:
- https://github.com/jcjohnson/fast-neural-style
- https://github.com/abhiskk/fast-neural-style


### おまけ: Dog or Cat? Another Application of VGG16: Transfer Learning (転移学習)
2010年代初頭まで, 人間にできて人工知能にできないことの代表例だった「[犬猫の分類](https://github.com/etttttte/mayfest2018/blob/master/dog_or_cat.ipynb)」.
今では学習済みVGG16を用いてお手軽に実装することができます.

Until the early 2010s, “[classification of dogs and cats](https://github.com/etttttte/mayfest2018/blob/master/dog_or_cat.ipynb)” was one of the most popular examples to show AI limitations. However, using pretrained VGG16, this task can be easily solved even with your laptop.


## References
\[1] L. Gatys, A. Ecker and M. Bethge. A Neural Algorithm of Artistic Style. 2015. http://arxiv.org/abs/1508.06576.  
\[2] J. Johnson, A. Alahi and F. Li. Perceptual Losses for Real-Time Style Transfer and Super-Resolution. 2016. http://arxiv.org/abs/1603.08155    
\[3] D. Ulyanov, A. Vedaldi and V. Lempitsky. Instance Normalization: The Missing Ingredient for Fast Stylization. 2016. https://arxiv.org/abs/1607.08022
