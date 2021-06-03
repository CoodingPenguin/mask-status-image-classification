# P-Stage 1. 마스크 이미지 분류하기

## 👩‍💻 작성자

- 박소현 T1072

## 🏆 최종 등수

- `Public LB` : **등수** 40등 | **정확도** 80.4762% | **F1-Score** 0.7625
- `Private LB` : **등수** 38등 | **정확도** 80.5238% | **F1-Score** 0.7506

## 📄 같이 보면 좋을 자료

- [제출한 모델의 스펙과 리더보드 결과로그](https://www.notion.so/heyparkso/9dff4eef8e924ad28c4c8b013f7e516c?v=2aabcf81385f4afcb19534b7cd90774d)
- [experiment 폴더에서 진행한 실험과 결과, 결론](https://www.notion.so/heyparkso/1863eb8206104f7a8caba78e78be3eb1?v=dc6b271460db4537b2736d314d5797cd)

## 📁 폴더/파일 구조

```
├── /code                                   // 베이스코드 폴더
│   ├── /ensemble                               // 앙상블할 모델과 결과 폴더
│   ├── /output                                 // 모델별 테스트셋 예측결과가 저장된 폴더
│   ├── /dataset.py
│   ├── /loss.py
│   ├── /madgrad.py                             // criterion: https://github.com/facebookresearch/madgrad
│   ├── /model.py
│   │── /train.py
│   ├── /inference.py
│   ├── /evaluation.py
│   ├── /sample_submission.ipynb
│   ├── /ensemble.ipynb                         // 앙상블 코드
│   └── /requirements.txt
├── /experiment                             // 실험코드 폴더
│   ├── /0_practice.ipynb                       // 코드 연습장
│   ├── /1_eda.ipynb                            // EDA 코드
│   ├── /2_dataset_construction.ipynb           // 데이터셋 구축 코드
│   ├── /3_1_augmentation.ipynb                 // augmentation 실험 코드
│   ├── /3_2_cross_validation.ipynb             // cv 구현 코드
│   ├── /3_3_optimizer.ipynb                    // optimizer 실험 코드
│   ├── /3_4_criterion.ipynb                    // criterion 실험 코드
│   ├── /3_5_lr_scheduler.ipynb                 // scheduler 실험 코드
│   ├── /4_1_resnet50_cv.ipynb                  // model v2.x 학습 코드
│   ├── /4_2_resnet50.ipynb                     // model v3.x 학습 코드
│   └── /4_3_efficientnetb0.ipynb               // model v4.x 학습 코드
└── ./submission                            // experiment 폴더 학습 코드의 예측결과가 저장된 폴더
```
