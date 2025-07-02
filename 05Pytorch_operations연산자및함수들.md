## 05. PyTorch 연산자 및 함수 정리

### cat과 stack의 차이

- **cat**: 기존 차원 중 하나를 따라 단순히 이어붙임 (차원 수 유지)
- **stack**: 새로운 차원을 만들어 쌓음 (차원 수 +1)


### 1. torch.cat의 동작 원리

- 여러 텐서를 하나로 합칠 때 사용하는 함수
- **dim 인자**로 합칠 방향(축)을 지정
    - `dim=0`: 행 방향(아래로)
    - `dim=1`: 열 방향(오른쪽)


### 2. shape(행/열) 불일치 시 발생 오류

- cat을 사용할 때, 합치려는 축(dim)을 제외한 모든 차원의 크기가 같아야 함
- 다르면 아래와 같은 오류가 발생


### 예시: 행 개수가 다를 때 (dim=1)

```python
import torch

b = torch.tensor([[0, 1], [2, 3]])  # shape: (2, 2)
c = torch.tensor([[4, 5]])          # shape: (1, 2)
d = torch.cat((b, c), dim=1)
# RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 2 but got size 1 for tensor number 1 in the list.
```


## 3. 행/열 맞추는 방법

### (1) reshape로 맞추기

```python
# c의 shape을 (2, 1)로 변경 (행 개수 맞추기)
c2 = c.reshape(2, 1)
d = torch.cat((b, c2), dim=1)
# 결과:
# tensor([[0, 1, 4],
#         [2, 3, 5]])
```


### (2) 예시 도식

| 텐서명 | shape | 설명 |
| :-- | :-- | :-- |
| b | (2, 2) | 2행 2열 |
| c | (1, 2) | 1행 2열 |
| c2 | (2, 1) | 2행 1열 (reshape 후) |

## 4. tip

- cat 전에 항상 shape을 print로 확인
- `dim=0`: 열(가로) 크기가 같아야 함 (행 방향 합치기)
- `dim=1`: 행(세로) 크기가 같아야 함 (열 방향 합치기)
- shape이 다르면 reshape/unsqueeze 등으로 맞춰주기
- 오류 메시지에 "Sizes of tensors must match except in dimension X"가 나오면, 그 차원을 제외한 나머지 shape을 맞추라는 뜻


## 5. repeat()와 expand()의 차이 및 사용법

### 핵심 비교

| 함수 | 확장 제약 | 메모리 복사 | 특징/용도 |
| :-- | :-- | :-- | :-- |
| expand() | 크기 1인 차원만 확장 가능 | 없음 | 메모리 효율적, 읽기 전용(쓰기 불가), 브로드캐스팅 |
| repeat() | 제약 없음 | 있음 | 실제 데이터 복사, 쓰기 가능, 메모리 사용 많음 |

### expand() 예시

```python
import torch

x = torch.tensor([[1], [2], [3]])  # shape: (3, 1)
y = x.expand(3, 4)
print(y)
# tensor([[1, 1, 1, 1],
#         [2, 2, 2, 2],
#         [3, 3, 3, 3]])
```

- **주의:** 크기가 1이 아닌 차원을 확장하려 하면 에러 발생


### repeat() 예시

```python
import torch

h = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
i = h.repeat(2, 3)
print(i)
# tensor([[1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4],
#         [1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4]])
```


### tip 요약

- **메모리 절약이 중요**: expand() 사용 (단, 크기 1인 차원만 확장 가능, 쓰기 불가)
- **유연한 확장/반복이 필요**: repeat() 사용 (메모리 복사 발생, 쓰기 가능)
- expand로 안 되는 경우 repeat 사용
- expand + clone(): 확장 후 복사본 필요할 때


## 6. in-place 연산 요약

- in-place 연산은 a의 메모리 주소에 결과를 직접 저장
- 새로운 텐서나 추가 메모리 공간을 만들지 않음
- b는 변하지 않고, a만 덮어써짐
- 메모리 절약 효과가 있지만, autograd 사용 시 주의 필요


### 참고

- 언더바(_)가 붙은 함수는 in-place 연산 (예: `a.add_(b)`)
- `+=`, `-=`, 등 파이썬 연산자도 in-place 연산
- in-place 연산은 메모리 절약에 유리하지만, autograd 사용 시 주의


## 7. 브로드캐스팅(broadcasting)이란?

### 산술연산 자동 크기 확장

- 브로드캐스팅은 크기가 다른 텐서 간의 산술 연산(+, -, *, / 등)에서
작은 쪽 텐서의 차원/크기를 자동으로 확장해서 연산을 수행하는 기능입니다.


#### 예시

```python
import torch

c = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
d = torch.tensor([1, 5])            # (2,)
result = c + d  # d가 (1, 2) → (2, 2)로 확장되어 연산
print(result)
# tensor([[2, 7],
#         [4, 9]])
```


### 수학적 행렬 연산과의 차이

- 수학적 행렬 연산은 반드시 행과 열의 크기가 맞아야 연산이 가능
- PyTorch의 브로드캐스팅은
    - 한쪽 텐서의 크기가 1이거나,
    - 차원이 부족할 때 자동으로 차원을 추가,
    - 필요한 만큼 값을 반복해서 연산


### 브로드캐스팅의 장점

- 코드 단순화: 복잡한 for문, 명시적 반복 없이 간단하게 연산
- 효율성: 내부적으로 최적화된 연산으로 메모리와 속도 모두 유리
- 유연성: 다양한 shape의 텐서를 자유롭게 연산
- 가독성: 코드가 짧고 명확해져 유지보수에 용이


### 브로드캐스팅의 단점/주의점

- 규칙을 지키지 않으면 에러
    - 마지막 차원부터 비교해서, 크기가 같거나 한쪽이 1이어야만 브로드캐스팅 가능
    - 그렇지 않으면 `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension N` 에러 발생
- 예상치 못한 메모리 사용 증가
    - 매우 큰 텐서로 브로드캐스팅이 일어나면, 내부적으로 임시로 큰 텐서가 만들어질 수 있음
- 암묵적 확장으로 인한 실수
    - 의도하지 않은 연산 결과가 나올 수 있음
    - shape을 항상 print로 확인하는 습관 필요


### 장단점 요약 표

| 장점 | 단점/주의점 |
| :-- | :-- |
| 코드 단순화, 효율성, 유연성, 가독성 | 규칙 미준수 시 에러, 메모리 사용 증가, 의도치 않은 결과 |

## 8. 주요 연산자/함수 요약

| 함수/연산자 | 설명(역할) | 반환 타입 | in-place 지원 |
| :-- | :-- | :-- | :--: |
| torch.add(a, b) | a와 b를 더함 | Tensor | O (add_) |
| torch.sub(a, b) | a에서 b를 뺌 | Tensor | O (sub_) |
| torch.mul(a, b) | a와 b를 곱함 | Tensor | O (mul_) |
| torch.div(a, b) | a를 b로 나눔 | Tensor | O (div_) |
| torch.pow(a, n) | a의 각 원소를 n제곱 | Tensor | O (pow_) |
| torch.pow(a, 1/n) | a의 각 원소에 n분의 1제곱(루트) | Tensor | O (pow_) |
| torch.eq(a, b) | a와 b가 같은지 비교 | Boolean Tensor | X |
| torch.ne(a, b) | a와 b가 다른지 비교 | Boolean Tensor | X |
| torch.gt(a, b) | a > b 비교 | Boolean Tensor | X |
| torch.ge(a, b) | a >= b 비교 | Boolean Tensor | X |
| torch.lt(a, b) | a < b 비교 | Boolean Tensor | X |
| torch.le(a, b) | a <= b 비교 | Boolean Tensor | X |
| torch.logical_and(a, b) | 논리 AND 연산 | Boolean Tensor | X |
| torch.logical_or(a, b) | 논리 OR 연산 | Boolean Tensor | X |
| torch.logical_xor(a, b) | 논리 XOR 연산 | Boolean Tensor | X |

## 9. 정리

- cat/stack 사용 전 shape을 print로 확인
- expand는 메모리 절약, repeat은 유연한 확장
- in-place 연산은 메모리 절약에 유리하지만, autograd 사용 시 주의
- 브로드캐스팅은 shape 규칙을 꼭 숙지
