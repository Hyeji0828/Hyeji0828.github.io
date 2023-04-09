---
layout: posts
title: "[VGGNet] VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE - SCALE IMAGE RECOGNITION 리뷰"
date: 2023-03-30
category: PaperReview
---

# [VGGNet 논문](https://arxiv.org/abs/1409.1556) 리뷰

## 개요

CNN에서 깊이 depth가 accuray에 미치는 영향을 연구했다. 3x3 convolution 필터를 사용해 깊은 네트워크에서의 성능을 눈에 띄게 향상시켰다. 2014년 ImageNet 대회에서 2등을 했다. (1등은 GoogleNet)

## 1. 소개

대량의 이미지와 고성능 gpu 덕분에 최근 cnn이 좋은 성과를 거두고있다. 성능을 향상시키는 여러 방법이 제시되었지만 이 논문에서는 또 다른 중요한 요소인 **네트워크 깊이**에 대해서 다뤄보겠다. **3x3 conv필터**를 사용해 conv 레이어를 추가하였고 결과적으로 성능을 크게 향상 시킬 수 있었다.

## 2. ConvNet Configuration

## 2.1 Architecture 구조

입력은 224x224의 RGB 이미지로 pre-processing 과정에서 트레이닝 셋에서 계산된 RGB 평균을 픽셀마다 빼주었다. (픽셀 - 평균) 이 이미지들은 여러 개의 conv 레이어를 거치고 레이어에서는 **3x3 필터**를 사용했다. 3x3인 이유는 **상하좌우 중앙을 감지할 수 있는 최소한의 크기**이기 때문이다. 구성 중 하나로 1x1 필터를 사용하기도 했는데, 이는 입력 채널에 대한 선형변환(과 그 후 비선형 변환)으로 볼 수 있습니다. conv stride는 1로 고정되어(conv의 spatial 패딩) convolution 이후에도 해상도가 유지됩니다. (3x3 필터의 패딩은 1) Spatial Pooling은 5개의 맥스 풀링으로 이루어졌으며, 이 중 몇개는 conv 이후에 적용됩니다. 맥스 풀링은 2x2 픽셀 window에 stride 2로 연산됩니다.

- Spatial Pooling : conv 레이어 출력을 작은 영역으로 나누고 대표값 (최대값이나 평균값)을 계산하여 나타내기 때문에 출력 크기를 줄일 수 있다. 또한 이미지가 이동하거나 회전하더라도 출력이 동일하게 유지되는 성질을 높여줄 수 있다.

### 2.2 Configurations

A~E 모델은 같은 구조의 다른 레이어 깊이를 가진다. A는 11개의 weight 레이어와 8 conv 레이어, 3 FC 레이어를 가지며 E는 19개의 weight 레이어와 16 conv 레이어, 3 FC 레이어를 가진다.

### 2.3 Discussion

첫 레이어에서 비교적 큰 receptive fields 가지는 다른 모델과 달리 전체 네트워크에서 3x3 receptive fields를 가진다. (stride=1) 중간에 Spatial Pooling이 없는 두 3x3 레이어는 5x5의 receptive field를 갖는다. **3개의 3x3 레이어는 하나의 7x7 receptive field**를 갖는다. 

7x7 레이어 한 개 대신 3x3 레이어 3개를 쓰는 것의 장점은

1. **1개 대신 3개의 비선형 rectification 레이어**를 쓰기 때문에 decision function이 더 잘 구분하게된다. (more discriminative)

1. **파라미터 수를 줄인다.** 7x7 필터에 규제를 적용하고 3x3필터를 통해 decomposition 하는 것이라 볼 수 있다. (그 사이에 비선형 inject 포함) 

1x1 cov 레이어를 사용하면 receptive field에 영향을 주지 않고 decision function의 비선형성을 증가시킬 수 있다.  → 비선형 함수를 한 번 더 쓰기 위해서 1x1 conv 사용.

→ **왜 적은 receptive field를 앞에서 쓰고 뒤에선 큰 receptive field를 뒤에서 쓸까?**

처음에는 작은 receptive로 상위레이어의 가장자리같은 디테일을 캐치하고 후반에 가면 움직임이나 abstract한 정보를 추출한다.

→ max pool을 하면서, 뒷단으로 가면서 abstract한 feature. 앞단에서 많이 쌓아도 local적인 부분만 얻을 수 있다. feature

- Receptive field : conv 레이어 뉴런 하나에 영향을 끼치는 범위. 필터 사이즈와 같다.
- Rectification function : ReLU


## 3. Classification Framework

classification 네트워크의 학습과 평가에 대한 자세한 내용을 다룬다. 전체적으로는 alexnet과 비슷.

### 3.1 Training

- 최적화optimization : multinomial logistic regression + 미니배치 경사하강법 + 모멘텀
- 배치 사이즈 : 256
- 모멘텀 : 0.9
- 규제 : weight decay + 드롭아웃(0.5, 처음 두 FC레이어)
- 학습률 learning rate : $10^{-2}$ + validation 셋의 **정확도가 향상되지 않으면 10으로 나누었다.**
    
    → 학습률은 전체적으로 3번 감소. 
    
- 에폭 : 74
- iteration : 370K

AlexNet보다 많은 파라미터 수와 레이어 깊이를 가졌지만

1. 더 깊은 레이어와 작은 컨볼루션 필터를 사용한 것이 규제로 적용되어 더 적은 epoch에서 수렴했다.
2. 특정 레이어의 pre-initialisation
- pre-initialisation : 가중치+bias를 무작위로 초기화하는 것이 아닌 이전에 큰 데이터셋에서 학습된 모델의 가중치를 가져와 초기화하는 것이다.

가중치 초기화를 잘 못하면 학습이 너무 오래걸리는 문제가 있는데, 이런 문제가 생기지 않는 충분히 얕은 모델 A는 무작위로 가중치를 초기화했다. 그리고 **더 깊은 레이어를 가진 모델은 A의 가중치로 초기화**했다. 이때 A의 가중치를 사용한 레이어는 첫 4개의 conv레이어와 마지막 3개의 fc레이어이고 나머지 레이어는 무작위로 초기화했다. pre-initialisation된 레이어는 학습률을 조정하지 않고 학습 중에 가중치가 변경되도록 했다.

무작위 초기화는 평균0이고 분산이 $10^{-2}$인 정규분포에서 샘플링했으며 편향은 0으로 초기화했다. 그리고 논문 기제 후에 Glorot&Bengio의 랜덤 초기화 방법으로 사전훈련 없이 가중치를 초기화할 수 있다는 것을 알게되었다.

- Glorot & Bengio 방법 : Xavier Glorot Initialization이다. tanh활성함수와 같이 많이 쓰이며 ReLU에는 잘 작동하지 않는다. ReLU는 He Initialization을 주로 쓴다.

224x224 고정 크기의 입력을 얻기위해 training 이미지를 랜덤으로 crop했으며 SGD반복마다 이미지당 하나씩 crop했다. 그리고 랜덤하게 horizontal flip을 적용하고 RGB Shift(AlexNet에서 적용한)를 적용했다.

**Training 이미지 크기**

training 이미지의 크기는 최소 224x224로 224보다 크다면 크롭한다. S = 이미지 크기

1. **S를 고정**하면 단일 스케일 학습 
    
    : S를 256으로 먼저 훈련하고 사전 훈련된 가중치고 s=384 학습 + 초기 학습률 = $10^{-3}$
    
2. 다중 스케일 학습
    
    : S를 min(256)과 max(512)사이에서 무작위로 뽑아 사이즈 조정 → 다양한 크기를 학습할 때 고려하게 함 + 스케일 조정으로 학습 세트를 확장했다고도 볼 수 있다.
    
    속도상의 이유로 동일한 구성의 단일 스케일 모델을 파인튜닝하여 학습했다. S=384로 학습된 단일 스케일 모델을 사용했다.
    

### 3.2 Testing

입력 이미지를 가장 작은 사이즈 Q(테스트 스케일)로 재조정한다. FC레이어는 conv레이어로 변환된다. (**첫 번째 fc레이어 → 7x7 conv, 마지막 두 fc 레이어 → 1x1 conv) → 다양한 크기의 이미지 입력을 받을 수 있다.**

결과는 클래스 갯수 개의 채널을 가진 클래스 score map이며, 이미지 크기에 따라서 **가변적인 해상도를 가진다. 마지막으로 클래스 score 벡터를 얻기위해 score map을 평균화(sum pooling)한다.** 그리고 이미지를 수평으로 뒤집어 테스트 세트를 보강한다. 원래 이미지와 뒤집힌 이미지의 소프트 맥스 사후확률을 평균해서 이미지의 최종 score를 얻는다.

테스트시에는 여러개의 크롭을 샘플링할 필요가 없다. 하지만 많은 크롭 데이터를 사용하면 더 세밀한 샘플링이 가능해 정확성을 향상시킬 수 있다. 

### 3.3 Implementation Details

멀티 gpu를 사용했다.

## 4. Classification Experiments

**데이터 셋**

training 1.3M, validation 50K, testing 100K. 

평가 방법 : top1, top5 에러

### 4.1 Single Scale Evaluation

Q = S (= 고정값). Q = 0.5(Smin + Smax). S = Smin과 Smax 사이값.

1. 정규화(normalization)은 적용하지 않았음.
2. 깊이가 깊어질 수록 오차가 줄었다. **추가적인 비선형성**이 도움이 되었다. 그리고 공간적인 context를 캐치하는 것이 중요했다. → 작은 conv필터를 가진 deep 네트워크가 큰 conv 필터를 가진 얕은 네트워크보다 성능이 좋았다.
3. 1x1을 써서 활성함수를 더 쓰는 것보다 3x3으로 receptive filed를 늘리는게 성능이 더 좋았다.
4. 3x3 두 개를 5x5로 해서 여러번 돌린 것 보다 3x3을 많이 쓴 게 더 좋았다.
5. Scale Jittering (Smin이 256~512사이 값을 가짐)이 고정된 Smin을 가지는 것보다 성능이 좋았다. → multi-scale 이미지의 특징을 잡아내는데 도움이 되었다.

### 4.2 Multi-Scale Evaluation

다양한 크기(Q)의 이미지로 테스트했습니다. 훈련과 테스트 이미지 사이즈가 큰 차이가 나면 성능이 저하되므로 Q는 {S-32, S, @S+32}. 가변적 S로 훈련된 모델은 {Smin, 0.5(Smin + Smax), Smax} 라는 더 큰 범위에서 테스트 가능하다. 테스트 결과 고정된 사이즈로 훈련하는 것 보다 성능이 좋았다.

### 4.3 Multi-Crop Evaluation

dense conv net 평가와 multi-crop 평가를 비교해보았다.  softmax 결과를 평균 냄으로써 두 평가 테크닉의 보완성complementarity 또한 평가하였다. dense 평가보다 multi-crop 평가가 살짝 더 좋은 성능을 보였고 조합하는 것이 개별적인 평가보다 좋은 성능을 보였다. conv 연산 시 경계에 대해 다른 처리를 적용한 것이 이유라고 보았다.

- dense convnet → FC를 평가할 때 7x7, 1x1으로 바꿨던 모델
- multi-crop → 3가지 이미지 크기로 50개씩 자른 모델

### 4.4 ConvNet Fusion

지금까지는 개별의 모델을 평가했지만 이 파트에서는 소프트 맥스 class posterior 값을 평균내 출력 값을 조합했다. 가장 성능이 좋았던 multi-scale 모델 D,E를 앙상블 기법으로 실험해보니 성능이 향상되었다.

- class posterior : 예측 클래스의 확률 분포.

### 4.5 Comparison with the State of the Art

기존의 클래식한 convNet 구조에서 레이어의 깊이만 증가시켜 성능을 향상시켰다.

## 5. Conclusion

깊이가 깊을 수록 분류 accuracy에 이롭다는 것을 알 수 있었다. 그리고 우리의 모델은 다양한 tasks와 데이터셋에서 일반화가 잘 되었고 다른 모델과 같거나 더 좋은 성능을 보여주었다. 우리의 결과는 visual representation에서 깊이의 중요성을 확인시켜 주었다.

## A. Localisation

- Localisation : object detection 문제에서 객체의 위치를 찾는 것. 보통 한 개의 오브젝트를 찾는 것을 말하고 object detection은 여러개의 오브젝트를 찾는 것을 말하는 것 같다.

### A.1 Localisation ConvNet

**Training**

기본적으로 분류 convNet과 같지만, Logistic Regression을 유클리드 loss로 변경하였다. 이렇게 해서 예측 바운딩 박스 파라미터를 정답과의 편차에 페널티를 줬다. S=256과 S=384 두 개의 모델을 학습했고(시간이 부족해 jittering 은 사용X) 일치하는 분류 모델로 초기화 해주었다. 학습률은 $10^{-3}$으로 초기화했다. 모든 레이어를 파인튜닝 하는 것과 처음 두 FC레이어만 파인튜닝하는 것 두 개를 모두 시도해보았다. 마지막 FC레이어는 랜덤으로 초기화되어 scratch 부터 학습했다.

4D vector → x좌표, y좌표, 넓이, 높이

- SCR : 일단 바운딩박스 있는 곳을 찾고
- PCR : 개 바운딩 박스들을 모아서 출력, 고양이 바운딩 박스만 찾는 방법을 따로 학습

**Testing**

한 가지 방법은 Validation 셋을 가지고 다른 수정사항을 적용한 네트워크를 사용하는 방법으로 바운딩 박스의 클래스 예측만 고려했다. 이미지 중앙 crop에만 bounding box 적용.

전체 이미지에 대한 classification task와 과정이 같으며 차이점은 class score map대신 출력 fc 레이어가 bounding box 예측을 내놓는 것이다. greedy merging procedure을 사용. 

- greedy merging precedure : 공간적으로 가까운 예측값을 합치고 (merge) classification convnet에서 얻은 class score를 기반으로 평가(rate)한다.

여러개의 localisation copnv net을 사용했다. 여러개의 bounding box 예측값의 합집합에 대해 merging 을 적용했다. multiple pooling offset 기술을 사용하지 않아 bounding box의 공간 해상도를 높일 수 있었다.

### A.2 Localisation Experiments

**Setting Comparison**

모든 레이어를 fine-tuning하는게 fc레이어만 fine-tuning한것 보다 좋은 결과를 가져왔다.

**Fully-fledged evaluation**

제일 좋은 성능의 (PCR, 모든 레이어가 Fine-tuning된) 세팅을 정했다. classification task와 같이 여러개의 scale을 테스트하고 결과를 종합하는 것이 성능을 개선했다.

**Comparison with the state of the art**

깊은 convnet이면서 간단한 localisation 방법으로 더 좋은 결과를 얻었다.

## B. Generalisation of Very Deep Features

이 섹션에서는 feature 추출기로서의 convnet을 평가하겠다. 큰 모델을 처음부터 학습시키기에는 오버피팅 문제 때문에 불가능하므로 더 작은 데이터셋에 대해 ILSVRC에서 pre-trained되었다. ILSVRC에서 학습한 deep image representations들은 다른 데이터셋에 대해 일반화가 잘 되었는데, 사람보다 뛰어난 성능을 보여줬다. 더 얕은 최신 기술 모델보다 더 좋은 성능을 가지는지 조사해보았다. 가장 좋은 성능을 보여준 D,E 모델을 사용했다.

다른 데이터셋을 사용하기 위해서 (1000개 클래스를 뱉는) 마지막 FC 레이어를 제거하고 마지막에서 두 번째 4096-D 활성함수 를 이미지 feature로 사용했다. 

결과 이미지 discriptor는 L2norm과 linear SVM classifier를 혼합해서 학습했다. pre-trained 된 가중치는 고정했다. (fine-tuning은 실행하지 않았다. fine-tuning은 conv레이어에 대해 낮은 학습률로 학습한다.) 

feature집계는 ILSVRC 평가와 비슷한 방식으로 진행되었다. 즉, 이미지의 가장 작은 사이즈가 Q와 같게 스케일링했다. 

feature map에 global average pooling을 적용해 4096-D 이미지 descriptor를 만들었다. 그리고 flip된 이미지에 대한 descriptor와 평균을 냈다. 다양한 사이즈로 평가한게 beneficial했기 때문에 다양한 사이즈의 Q에서 feature를 추출했다. 이 feature는 스케일에 따라 stacked되거나 pooled 되었다. stacking 쌓는 방식은 다양한 사이즈의 이미지 정보를 효율적으로 합쳤다. 하지만 이 방식은 dim을 증가 시켰다. 

- global average pooling : 평균값을 취하는 pooling으로 어떤 사이즈의 이미지던간에 (1, 1, c)크기가 된다.

[Global Average Pooling 이란](https://gaussian37.github.io/dl-concept-global_average_pooling/)

**VOC2007, 2012**

한 개 이상의 라벨을 가진 이미지 PASCAL VOC → 20개 클래스. 성능은 mAP(mean average precision)으로 평가 되었다.

- mean average precision : 가장 많이 쓰는 물체인식 알고리즘의 평가 기준.

[mAP(Mean Average Precision) 정리](https://ctkim.tistory.com/79)

여러 사이즈에 대해 image descriptor를 집계?aggregating 한 것이 stacking해 집계한 것과 비슷한 결과를 냈다. 이는 voc 데이터에서 나타나는 물체의 사이즈가 다양해 classifier가 혼돈할 만한 크기 정보가 없었던 것 때문으로 추측한다. 평균을 내는 방식은 dim을 증가시키지 않기 때문에 다양한 사이즈로 이미지 descriptor를 집계할 수 있었다. 하지만 차이는 근소했다.

VOC데이터에 대한 D, E 성능은 비슷했고 둘을 합친 것은 좀 더 향상된 결과를 가져왔다. 

**Caltech101,256**

102개의 클래스를 가진 caltech101, 257개인 256. 학습 데이터셋과 테스트 데이터를 랜덤으로 나누고 나눈 셋의 성능을 평균으로 구했다. 학습 데이터의 20%를 validation 데이터로 썼다.

여러 사이즈의 계산 결과를 stacking, 쌓는 것이 average나 max-pooling보다 더 성능이 좋았다. 이 이유는 이 데이터셋은 물체가 이미지의 전체를 차지했기 때문에 멀티 스케일 이미지의 경우는 정보가 달랐다. (전체 물체 vs 물체의 일부) stacking 방식이 사이즈에 특정되는 정보를 잘 잡아냈다. 

**Action Classification on VOC**

학습 방법1. 전체 이미지에 대해 feature를 계산하고 bounding box 무시

학습 방법2. 전체 이미지에 대해 feature를 계산하고 bounding box에 대해서도 계산하고 두 개를 Stacking.


**→ 레이어가 깊으면 다 좋을까?** 

너무 깊으면 오히려 좋지 않다. 무조건 깊이 쌓는게 좋은것은 아니다.

- efficientNet  : 성능과 크기 사이에서 최적점을 찾는 노력

---