# 2018年5月祭 応用物理系工学博覧会　アルゴリズム班展示
2018年5月19日(土), 20日(日)に開催された[5月祭](https://gogatsusai.jp/91/visitor/)の, 工学部応用物理系([物理工学科](http://www.ap.t.u-tokyo.ac.jp/), [計数工学科](https://www.keisu.t.u-tokyo.ac.jp/))「[工学展覧会](https://ap-phys.net/18/)」で展示したものです.
当日展示, 配布した資料は以下にアップロードしてあります.
- 展示ポスター([1](https://github.com/etttttte/mayfest2018/blob/master/algorithm_NN_poster_1.pdf), [2](https://github.com/etttttte/mayfest2018/blob/master/algorithm_NN_poster_2.pdf))
- [理論冊子](https://github.com/etttttte/mayfest2018/blob/master/algorithm_NN_theory.pdf)

<img src="https://github.com/etttttte/mayfest2018/blob/master/fast-neural/images/akamon_balla.jpg" alt="" title="" width="400">   <img src="https://github.com/etttttte/mayfest2018/blob/master/fast-neural/images/akamon_dubuffet.jpg" alt="" title="" width="400">
<img src="https://github.com/etttttte/mayfest2018/blob/master/fast-neural/images/akamon_gogh.jpg" alt="" title="" width="400">   <img src="https://github.com/etttttte/mayfest2018/blob/master/fast-neural/images/akamon_munch.jpg" alt="" title="" width="400">


## 展示概要: Enchanted with Machine Learning
連日話題の人工知能. 魔法とうたわれ, 錬金術かのようにもてはやされていますが, 所詮は計算にすぎません.
5月祭ではCNNに着目し, その歴史的背景や基礎から, 応用例の画風変換まで展示しました. このレポジトリでは主に画風変換を紹介しています.
Python3, PyTorchで実装しています.

### 画風変換(neural style transfer)
画風変換とは, コンテンツ画像とスタイル画像が与えられた時, コンテンツの情報をできるだけ保ったままスタイル画像に画風を寄せた画像を出力するアルゴリズムです.
例えば, 赤門の写真をゴッホ風やムンク風に変換できます.
画像がVGGを通ることで, 段々スタイルに関する情報が捨てられ, コンテンツに関する情報のみが抽出されていくという性質を活用しています.
#### オリジナル
[Gatys et al.(2015)](https://arxiv.org/abs/1508.06576)によって提唱されたアルゴリズムです. PyTorch公式の[チュートリアル](http://pytorch.org/tutorials/advanced/neural_style_tutorial.html)にほぼ従っています.  
`$python neural_style_transfer.py style_img content_img`  
で実行されます.
CPU環境で数十分～数時間かかります.

#### リアルタイム画風変換
オリジナルのアルゴリズムは任意のコンテンツ画像を任意のスタイル画像に変換できる一方, 毎回変換器を学習するため変換に時間がかかります.
この問題を解決したのが[Johnson et al.(2016)](http://arxiv.org/abs/1603.08155)です. スタイル画像ごとに変換器を事前に学習しておくことで, 画風変換自体のスピードが1000倍程度高速化しました.

[学習済みのモデル](https://github.com/etttttte/mayfest2018/tree/master/fast-neural/models)をダウンロードしたのち,  
`$python fast_style_change.py content_image model`  
で実行されます. content_imgは変換したいコンテンツ画像(.jpg)へのパス, model = {0:Balla, 1:Dubuffet, 2:Gogh, 3:Munch} です.

新たに変換器を学習したい場合は,  
`$python fast_style_train.py`  
で学習できます. その場合, fast_style_train.py内のline 1を学習したいスタイル画像へのパスに書き換えて下さい.
学習には[Microsoft COCO Dataset](http://cocodataset.org/#home)のtrain2014(~80k枚, 13GB)を用いました. 
GPU環境で2epochの学習に9-10時間かかります.

実装には, 以下を参考にしました.
- https://github.com/jcjohnson/fast-neural-style
- https://github.com/abhiskk/fast-neural-style


### おまけ: VGG16のその他の応用例 - 転移学習(transfer learning)
2010年代初頭まで, 人間にできて人工知能にできないことの代表例だった「[犬猫の分類](https://github.com/etttttte/mayfest2018/blob/master/dog_or_cat.ipynb)」.
今では学習済みVGG16を用いてお手軽に実装することができます.


## 参考文献
\[1] L. Gatys, A. Ecker and M. Bethge. A Neural Algorithm of Artistic Style. 2015. http://arxiv.org/abs/1508.06576.  
\[2] J. Johnson, A. Alahi and F. Li. Perceptual Losses for Real-Time Style Transfer and Super-Resolution. 2016. http://arxiv.org/abs/1603.08155    
\[3] D. Ulyanov, A. Vedaldi and V. Lempitsky. Instance Normalization: The Missing Ingredient for Fast Stylization. 2016. https://arxiv.org/abs/1607.08022
