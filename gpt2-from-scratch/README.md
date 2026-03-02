# LLMs from Scratch — GPT-2 구현 프로젝트

Sebastian Raschka의 *Build a Large Language Model From Scratch* 교재를 바탕으로 GPT-2를 직접 구현하고 파인튜닝한 프로젝트입니다.

---

## 프로젝트 구성

### 1. 사전 학습 (pretrain.py)
GPT-2 Small (124M) 모델 구조를 처음부터 직접 구현하고, 텍스트 데이터로 사전 학습을 진행합니다.

- GPT-2 아키텍처 구현 (MultiHeadAttention, TransformerBlock 등)
- 학습 완료 후 가중치를 `model_and_optimizer.pth`로 저장

### 2. 분류 파인튜닝 (finetune_classifier.py)
1번에서 학습한 GPT-2 Small (124M) 가중치를 불러와 스팸 분류 모델로 파인튜닝합니다.

- `pretrain.py`의 모델 구조와 저장된 가중치(`model_and_optimizer.pth`)를 기반으로 동작
- 마지막 레이어에 분류 헤드(2-class)를 추가하여 spam/ham 분류 수행
- SMS Spam Collection 데이터셋 사용

### 3. 지시 파인튜닝 (finetune_instruction.py)
OpenAI의 공식 GPT-2 Medium (355M) 가중치를 직접 다운로드하여 지시 따르기(instruction following) 모델로 파인튜닝합니다.

- 1, 2번과 달리 사전 학습 코드와 무관하게 **단독으로 실행** 가능
- GPT-2 Medium은 Small보다 크기가 크기 때문에 OpenAI 서버에서 가중치를 직접 받아 사용
- Alpaca 스타일의 instruction 데이터셋으로 파인튜닝
- 코랩(Google Colab)에서 단독 실행 가능하도록 모든 코드가 한 파일에 포함

---

## 파일 구조

```
├── pretrain.py                  # GPT-2 구현 및 사전 학습
├── finetune_classifier.py       # 분류 파인튜닝 (pretrain 기반)
├── finetune_instruction.py      # 지시 파인튜닝 (단독 실행)
├── model_and_optimizer.pth      # pretrain 저장 가중치
├── finetune_classifier_model_and_optimizer.pth  # 분류 모델 저장 가중치
└── gpt2-medium355M-sft.pth      # 지시 모델 저장 가중치
```

---

## 모델 요약

| 구분 | 모델 | 가중치 출처 | 단독 실행 |
|------|------|------------|----------|
| 사전 학습 | GPT-2 Small (124M) | 직접 학습 | O |
| 분류 파인튜닝 | GPT-2 Small (124M) | pretrain.py | X |
| 지시 파인튜닝 | GPT-2 Medium (355M) | OpenAI 공식 | O |