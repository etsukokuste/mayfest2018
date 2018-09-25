# Real-Time Style Transfer (高速画風変換)
This is PyTorch implementation of Real-Time Style Transfer.

- 学習済みモデルを利用して画風変換したい場合 :  
      [models](https://github.com/etttttte/mayfest2018/tree/master/fast-neural/models)をダウンロードしたのち,  
      `$python fast_style_change.py content_img model`  
      model = {0:Balla, 1:Dubuffet, 2:Gogh, 3:Munch} です.
 
- 新たな変換器を学習したい場合 :  
      fast_style_train.pyのline 130のスタイル画像へのパスを書き換えたのち,  
      `$python fast_style_train.py`  
  
    
- testing with pretrained models:
  1. Download pretrained models.
  2. `$python fast_style_change.py content_img model`   
     'content_img' is the path to your content image (.jpg) and model = {0:Balla, 1:Dubuffet, 2:Gogh, 3:Munch}.
     
- training a new model:  
  `$python fast_style_train.py path_to_style_img` 
  
      
### contents of this directory
- vgg16.py: Loss FunctionとしてカスタマイズしたVGG16
- transformer_net.py: 変換器本体
- utils.py: 画像読込, VGG16の重み初期化etc.
- fast_style_train.py: 変換器の学習
- fast_style_change.py: 学習済みの変換器を用いて画風変換を行う
- models: 学習済みのモデル
- images: 学習したスタイル画像, 画風変換した赤門たち

- vgg16.py: customized VGG16 for loss function
- transformer_net.py: Image Transformer Net
- utils.py: loading images, VGG16 weights initialization etc.
- fast_style_train.py: training a model
- fast_style_change.py: style transfer with a pretrained model
- models: pretrained models
- images: images used for training, style-transfered Akamon Gate images
