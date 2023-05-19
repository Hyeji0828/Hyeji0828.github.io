---
layout: posts
title: "[ResNet] Deep Residual Learning for Image Recognition 리뷰"
date: 2023-04-21
category: PaperReview
---

# [ResNet 논문](https://arxiv.org/abs/1512.03385) 리뷰

## 개요

ResNet은 2015년 ILSVRC에서 1등을 한 모델로 딥러닝 네트워크는 학습이 어렵고 깊게 쌓기 힘들다는 점을 보완했다.

## 1. 소개

성능 향상을 위해서는 네트워크의 레이어를 깊게 쌓는 것이 중요하다. 하지만 깊게 쌓은 모델을 학습하는 것은 그리 쉽지 않다. Gradient Vanishing/Exploding 문제가 주로 발생하고 normalization 레이어와 normalized 초기화로 해결해왔다. 하지만 네트워크 loss가 줄어들고 수렴하기 시작하면서degradation 문제가 발생한다. 이 degradation은 네트워크 깊이가 증가하면서 accuarcy가 saturated포화되다가 빠르게 감소하는 문제다. 이 문제는 사실 overffiting에 의한 것이 아니었고 이 논문에서는 이를 해결하기 위한 deep residual learning 프레임워크를 제시하고 있다.

원하는 결과를 도출하는 mapping을 H(x)라고 할 때, 비선형 레이어를 쌓은 맵핑은 F(x) + x로 치환할 수 있고 H(x)는 F(x)+x로 치환할 수 있다. H(x)보다 F(x)를 최적화하는 것이 더 쉽다는 가정으로 시작한다. 

ImageNet데이터로 실험해본 결과

1. Deep Residual Net은 최적화하기 쉬웠고 그냥 레이어를 쌓기만 한 plain한 네트워크는 깊이에 따라 에러도 증가했다.
2. 정확도가 향상되어 이전 네트워크보다 좋은 성능을 보여주었다.

## 2. Related Work

**Residual Representation**

residual vector를 인코딩하는 것이 original vector를 인코딩하는 것보다 효과적이다. 

reformulation과 preconditioning이 최적화를 간단하게 만든다. → H(x)를 F(x)+x로 치환한 것이 최적화를 간단하게 만들었다.

- VLAD : Vector of Locally Aggregated Descriptros. 이미지에서 하나의 local feature vector를 생성하는 알고리즘이다.
- Fisher Vector : VLAD의 확장 개념으로 더 우수한 성능을 가진 Local Feature Vector를 생성한다.
- Vector Quntization : 특징 데이터를 대표값으로 mapping 하는 것.

**Shortcut Conenctions**

Layer 출력, gradients, propagated erros에 집중하는 방법으로 shortcut connection이 제시되었다. “highway network”는 resnet의 identity mapping과 달리 파라미터가 있는 shortcut connection을 사용한다. 이 shortcut은 0에 가까워지면서 close 상태가 되면 학습을 중지하지만 resnet의 residual functions는 항상 학습을 하고 shortcut은 close상태가 되지 않는다. 이런 차이가 깊은 네트워크에서도 accurcy 향상을 보여준 것이라고 제시한다.

## 3. Deep Residual Learning

### 3.1 Residual Learning

여러 개의 레이어가 쌓인 구조를 H(x)로 가정할 때, 여러 개의 비선형 레이어가 복잡한 function을 근사할 수 있다면 Residual Function F(x)는 H(x)-x로 볼 수 있다. 추가된 레이어들이 identity mapping으로 설계된다면 더 얕은 모델과 같은 에러를 가질 것이다. 

<aside>
📌 y = f(x) + x : f(x) 를 x로 근사하는 것이 어렵고 0으로 근사하는 것은 쉽다 → 그러니 f(x)를 0으로 근사하고 x를 더하자.

</aside>

### 3.2 Identity Mapping by Shortcuts

$F(x, {W_i})$ 는 학습하고자 하는 Redisual mapping이다. $F + x$ 는 shortcut connection과 element-wise addition(덧셈)으로 계산된다. 이 shortcut connection은 추가 파라미터가 필요하지 않고 연산 복잡도를 증가시키지도 않는다. 

이때 x와 F의 차원은 같아야하는데, 그렇지 않을 경우는 $y = F(x, {W_i}) + W_sx$ 로 본다. $W_s$ 는 차원을 맞추기 위해서만 사용된다. 간단하게 표현하기 위해 FC레이어로 설정했지만 Conv 레이어에도 적용가능하다.

### 3.3 Network Architectures

**Plain Network**

VGGNet의 방식을 차용해 plain network를 구성했다. stride=2 의 conv 레이어로 downsampling을 수행하였고 네트워크의 마지막은 global average pooling레이어와 1000-way softmax fc레이어로 구성되었다.

********************************Residual Network********************************

위의 Plain 모델에 shortcut connection을 추가한 모델이다. identity shortcut은 input과 output의 차원이 같을 때 사용하고, 차원을 증가시킬 때는 두 가지 옵션을 고려했다. 1. zeo padding identity mapping과 2. 1x1 conv를 통한 projection shortcut이다. 두 경우 모두 다른 사이즈의 feature map을 사용하는 경우 stride=2 로 정했다.

### 3.4 Implementation

ImageNet의 방식을 차용해 [256,480]사이에 무작위로 선정된 값으로 resize하고 224x224로 crop한다. 각 conv 이후와 activation 이전에 Batch Normalization을 적용했다. dropout은 사용하지 않았으며 fully-convolutional form을 적용했다.

## 4.Experiments

### 4.1 ImageNet Classification

**Plain Network**

plain 모델은 레이어 깊이가 더 깊은 모델이 더 높은 validation error를 기록했다. 이는 backward나 forward 중 일어나는 signal vanish 문제도 아니었다. 

**Residual Network**

plain 모델에 3x3 필터마다 shortcut connection을 추가하니 1. 깊이에 따른 정확도 증가를 관찰할 수 있었다. 2. 깊은 모델에서 residual learning이 효과적이라는 것을 볼 수 있고 3. 깊지 않은 모델에서도 최적화가 쉬워 초기 학습단계에 빨리 수렴한다는 장점이 있다.

Shorcut Connection에는 3가지 옵션이 있다.

A) zero-padding으로 차원을 증가시킨다. (추가 파라미터는 없다)

B) 차원을 증가시켜야할 때 projection shortcut을 사용하고 그렇지 않으면 Identity shortcut 사용

C) 모든 shortcut은 projection이다.

ABC를 비교한 결과 projection shortcut은 degradation문제를 해결하는데 필수적이지는 않았다. Identity shortcut은 아래 소개될 bottleneck network의 복잡도를 증가시키지 않는다.

**Deeper Bottleneck Architecture**

더 깊은 bottleneck design은 1x1, 3x3, 1x1 세 개의 conv 레이어를 가지고 1x1 레이어는 차원을 축소하고 복원하는 역할을 한다. 이 bottleneck 모델은 일반 모델과 같은 시간 복잡도를 가진다. Identity가 아닌 projection shortcut을 사용한다면 시간 복잡도와 모델 사이즈가 두 배가 되고 결과를 두 배 높은 차원이 되기 때문에 Identity shortcut을 사용해야 효율적인 모델을 만들 수 있다.

### 4.2 CIFAR-10 and Analysis

극도로 깊은 네트워크에 대한 실험을 진행했다. 모든 shortcut은 identity 였으며 residual 모델은 일반 모델과 같은 깊이, 넓이, 파라미터 사이즈를 가졌다. 깊은 일반 모델은 깊이가 증가함에 따라 높은 training error를 보여주며 최적화의 어려움을 겪었다. 반면 residual 모델(ResNet)은 깊이가 증가함에 따라 정확도 향상을 누렸다.

**Analysis of Layer Responses**

ResNet은 일반 모델보다 더 작은 reponse를 보여주었고 이는 residual function이 non-residual function보다 0에 근사한다는 것을 알 수 있다.

**Exploring Over 1000 Layers**

ResNet은 1000개의 레이어 이후에도 최적화의 어려움을 겪지 않았다. 하지만 여전히 이런 극단적으로 깊은 모델은 오버피팅의 문제가 있었다. 이는 최적화의 어려움에 중점을 둔 이 논문에서는 다루지 않겠지만 강한 regularizaion을 합치는 것으로 결과를 향상시킬 수 있다고 예상한다.

---