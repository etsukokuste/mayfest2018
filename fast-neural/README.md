# リアルタイム画風変換

- 学習済みモデルを利用して画風変換したい場合 :  
      [models](https://github.com/etttttte/mayfest2018/tree/master/fast-neural/models)をダウンロードしたのち,  
      `$python fast_style_change.py content_img model`  
      model = {0:Balla, 1:Dubuffet, 2:Gogh, 3:Munch} です.
 
 
- 新たな変換器を学習したい場合 :  
      fast_style_train.pyのline 130のスタイル画像へのパスを書き換えたのち,  
      `$python fast_style_train.py`  
      
### 何が何か
- vgg16.py: Loss FunctionとしてカスタマイズしたVGG16
- transformer_net.py: 変換器本体
- utils.py: 画像読込, VGG16の重み初期化etc.
- fast_style_train.py: 変換器の学習
- fast_style_change.py: 学習済みの変換器を用いて画風変換を行う
- models: 学習済みのモデル
- images: 学習したスタイル画像, 画風変換した赤門たち
