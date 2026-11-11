# Throttle

pCVR 예측 모델을 제어 루프(PID)에 연결해 광고 예산 소진을 조절하는 시뮬레이터.

> 진행 중. 핵심 결과는 실험 완료 후 기록.

## 데이터

[Criteo Attribution Modeling for Bidding Dataset](https://huggingface.co/datasets/criteo/criteo-attribution-dataset)
(Criteo AI Lab, CC BY-NC-SA 4.0). 31일, 노출 1,650만 건, 캠페인 675개.

- 노출 상위 20개 캠페인 × uid 해시 40% 샘플 → 185만 행
- 날짜 경계로 시간순 분할: train 0~20일차 (68.5%), sim 21~30일차 (31.5%)

## 실행 방법

```bash
mkdir -p data/raw && curl -L -o data/raw/criteo_attribution_dataset.tsv.gz https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/criteo_attribution_dataset.tsv.gz
python src/prepare_data.py
python src/train.py
```
