notes with images are in notion (personal)

<h1>
# 신경망이란 무엇인가? | 1장.딥러닝에 관하여
</h1>
<a href="https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2&t=0s">link</a>

'multilayer perceptron'

**&Neuron : 'thing that holds a number' 로 기억하자.**

28*28 픽셀의 이미지가 있을 때, 각 픽셀의 0~1 (gray scale) 로 나타내지는 밝기 정도의 수가 activation (활성치)

그리고 이 784( = 28*28) 개의 neuron 이 하나의 layer 를 이루게 된다.

그리고 어떤 pattern 을 거쳐 ( 학습된 결과 ) 마지막 Output layer 에서 특정 숫자일 확률이 가장 높다고 판단되는 neuron 이 가장 밝게 빛나게 된다.

첫 번째 input layer 에서 두 번째 layer(hidden layer ) 의 neuron <a> 와 갖는 weights (가중치, 그냥 숫자!) 를 정하고,

input layer 의 activation (숫자) 와 각각 곱하여, ,<a> 의 값 (activation)을 구해낸다.

그리고 이렇게 구한 값은 실수범위를 갖게 되는데, 숫자 인식 문제의 경우 0~1 사이의 값을 원하기 때문에,

여기서 **sigmoid 함수 ( = logistic curve )**같은 것이 등장!

(근데, sigmoid 함수는 사실상 이제 잘 안 쓰이고 **ReLU 같은게 더 훈련시키기 좋고 요새 더 잘 쓰이는 함수 !)**

(ReLU : Rectified Linear Unit )

**#bias ( 특정 수치를 넘겨야만 반응하도록 만들고 싶다! ⇒ 이럴때 사용하는 것.)**

# 결국, **Learning :** Finding the right weights and biases


<h1>
# 경사 하강, 신경 네트워크가 학습하는 방법 | 심층 학습, 2장
</h1>
<a href="https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2">link</a>

weights : 직관적으로도 이해 가능한 부분이지만, 결국 n-1 단계의 neuron 과 n 단계의 neuron 의 연결 강도를 나타내는 것이라 볼 수 있다.

처음에 그냥 Input data 때려넣고 network 를 돌리면, 쓸모없는 결과가 나오기 쉽다. 따라서, 여기서 **Cost function** 이 필요해진다!

구체적으로, 원하는 결과값과 쓸모없는 결과와의 차의 제곱의 합 (squares of differences)  ⇒ cost of a single training example

⇒ network 가 뻘짓할 때 값이 크다!

⇒ average cost 를 구해서 ⇒ network 가 잘 작동하고 있는지 판단하는 잣대로 사용 (컴퓨터가)

# #Gradient Descent : **It's a way to converge towards some local minimum of a cost function**

which direction should you step in this input space??

결국, 가장 경사가 큰 지점을 찾고, 그 지점에서 밑으로 가면(step) 된다. (최소화)

각 가중치 (weights) 의 gradient 값을 통해, 어떤 가중치의 변화가 얼마나\어떻게 더 영향력이 있는지를 보여준다.

*unfathomably : 측량할 수 없게, 헤아릴 수 없게.

그리고, 결국 이런 구조는 결국 과거에 연구되었던 'multilayer perceptron' 이라는 개념! ⇒ 이후 convolutional neural network ⇒ LSTM 으로 넘어가는데, 나중에 나온 내용들을 이해하려면 일단 처음에 어떻게 시작되었는지를 이해하는 것이 중요하니까!

#최근연구

<h1>
# What is backpropagation really doing? | Deep learning, chapter 3
</h1>
<a href="https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3">link</a>

cost 를 줄이기 위해 gradient 를 계산하고 적용하는 과정을 반복

결국, Backpropagation은 gradient 를 계산하기 위해 필요한 알고리즘임!
cost 가 weight 에 갖는 의미. (이 예시에서, Wn 의 변화 1이 Wk 의 변화 1 보다 32배만큼 cost 함수의 결과에 더 영향력을 갖는다.)

notations&formulas of BackPropagation

### Neurons that fire together wire together

#Idea of propagating backwords 

각 training sample 이 원하는(desired) change (on weights and biases)를 구하고, 그 변화들의 평균값을 구한다. ⇒ 그리고 이 값이 '대략' negative gradient of the cost function 이라고 볼 수 있다.

#mini-batch

그리고 하나의 training example 이 전체 weight 와 bias에 끼치는 영향을 구하기 보다, 훈련 데이터를 mini-batch 로 쪼개서 진행하는 것이 계산 속도 측면에서 더 낫기 때문에, 이렇게들 하는데, 그걸 : Stochastic gradient descent 라고 부른다.

#정의 (wrap-up)

So, Backpropagation is the algorithm for determining how a single training example would like to nudge the weights and biases, 그냥 올려야하냐 내려야 하냐 뿐만 아니라, cost 를 얼마나 더\덜 줄이는데 영향을 끼치는 것인지도 포함.
<h1>
# Backpropagation calculus | Deep learning, chapter 4
</h1>
<a href="https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4">link</a>

#chain rule

#neurons wire together fire together

여기서 보면, z(l) 을 w(l)에 대해 편미분 했을 때, 그 값이 a(l-1) 인데, 즉, weight 의 변화량이 a(l)에 끼치는 영향은 previous neuron ( = a(l-1) ) 이 얼마나 강한지에 따라 달려있다는 의미.

앞서, 한 층에 하나의 neuron 만 존재한다는 가정에서 설명을 했는데, 한 layer 에 여러 neuron 이 존재한다고 하여도, 크게 달라지는 것은 없다!

L-1 단계 neuron 이 L 단계의 neuron 에 어떤 영향을 끼치는지는, L-1 단계 neuron 과 연결된 L 단계 neuron 들에 각각 나눠지기 때문에, sum 을 하는것임. ( 각 층에 하나의 neuron 만 있다고 생각했을 때와  다른 점 )

note : Backpropagation is really one instance of a more general technique called "reverse mode differentiation" to compute derivatives of functions represented in some kind of directed graph form.