# DarkIR Evaluation Pipeline — Описание проекта

> **04-deep-learning-systems** — учебный проект по курсу «Разработка систем глубокого обучения»

---

## Общее описание

Этот проект представляет собой **оценочный пайплайн для модели восстановления изображений при низком освещении** — [DarkIR](https://github.com/cidautai/DarkIR).

**DarkIR** — это эффективная нейросеть, представленная на CVPR 2025, которая выполняет три задачи одновременно:
- **Low-light enhancement** — улучшение освещённости тёмных сцен
- **Denoising** — удаление шума
- **Deblurring** — устранение размытия

Данный репозиторий содержит обёртку для **количественной оценки** качества работы модели: сравнение с эталонными результатами, расчёт метрик PSNR/SSIM и генерацию отчёта.

---

## Структура проекта

```
04-deep-learning-systems/
├── assets/
│   ├── inputs/          # Входные тёмные изображения для тестирования
│   ├── results/         # Эталонные результаты для сравнения
│   └── results_*/       # Сгенерированные результаты после запуска (генерируется)
├── models/
│   └── DarkIR_1k_cr_mt.pt  # Предобученные веса модели
├── run_test.py          # Основной скрипт оценки
├── requirements.txt     # Зависимости Python
├── Dockerfile           # Контейнеризация для воспроизводимости
└── evaluation_report.csv # Выходной отчёт (генерируется)
```

---

## Как запустить проект

```bash
# 1. Соберите образ
docker build -t darkir-eval .

# 2. Запустите контейнер
docker run --rm  darkir-eval
```

> Docker-образ основан на `python:3.10-slim`, автоматически клонирует оригинальный репозиторий DarkIR и настраивает `PYTHONPATH` для корректного импорта.

---

## Что делает скрипт `run_test.py`

1. **Загружает модель** DarkIR из файла `.pt` (поддерживает CPU/GPU)
2. **Обрабатывает все изображения** из `assets/inputs/`:
   - Применяет предобработку (ToTensor, нормализация)
   - Запускает инференс модели
   - Сохраняет результат в `assets/results_Full_1k/`
3. **Считает метрики качества**:
   - **PSNR** (Peak Signal-to-Noise Ratio) — отношение сигнал/шум
   - **SSIM** (Structural Similarity Index) — структурное сходство
   - Сравнение с эталоном (если есть в `assets/results/`)
   - Контрольная сумма MD5 для проверки идентичности вывода
4. **Генерирует отчёт**:
   - Детальный `evaluation_report.csv` с метриками по каждому изображению
   - Сводная статистика в консоль: средние PSNR/SSIM, время инференса, количество параметров

---

## Зависимости

Основные пакеты из `requirements.txt`:
- `torch`, `torchvision`, `torchaudio` — фреймворк для глубокого обучения
- `einops`, `kornia` — утилиты для работы с тензорами и аугментациями
- `lpips`, `pyiqa`, `pytorch-msssim` — метрики качества изображений
- `gradio` — (опционально) для создания веб-интерфейса
- `wandb` — логирование экспериментов
- `opencv-python`, `scikit-image`, `pillow` — обработка изображений

> В `requirements.txt` указан `--extra-index-url https://download.pytorch.org/whl/cpu` — по умолчанию устанавливается CPU-версия PyTorch. Для GPU удалите эту строку или замените на CUDA-версию.

---

## Пример вывода

```
===========================================================================
    DarkIR: Multi-Model Quantitative Evaluation Pipeline
===========================================================================

[INFO] Target device: cpu
[INFO] Found 4 images.

---------------------------------------------------------------------------
[INFO] Evaluating: Full_1k (DarkIR_1k_cr_mt.pt)
[INFO] Loaded. Trainable parameters: 3,321,638

Processing images (Full_1k):
  [1/4] 0010.png   | 1120x640  | Time: 12.92s
    Ref vs Out -> PSNR: >= 99.99 dB | SSIM: 1.0    | MD5: True
    In  vs Out -> PSNR: 11.35    dB | SSIM: 0.1355
  [2/4] 0075.png   | 1120x640  | Time: 11.30s
    Ref vs Out -> PSNR: >= 99.99 dB | SSIM: 1.0    | MD5: True
    In  vs Out -> PSNR: 13.66    dB | SSIM: 0.4701
  [3/4] 0087.png   | 1120x640  | Time: 12.20s
    Ref vs Out -> PSNR: >= 99.99 dB | SSIM: 1.0    | MD5: True
    In  vs Out -> PSNR: 11.48    dB | SSIM: 0.5456
  [4/4] 0088.png   | 1120x640  | Time: 11.54s
    Ref vs Out -> PSNR: >= 99.99 dB | SSIM: 1.0    | MD5: True
    In  vs Out -> PSNR: 8.28     dB | SSIM: 0.3566

[INFO] Full_1k finished. Avg inference: 11.99s/image

===========================================================================
AGGREGATED RESULTS
===========================================================================
Model        |     Params | Avg Ref PSNR | Avg Ref SSIM | Avg Enh PSNR | Avg Time (s)
---------------------------------------------------------------------------
Full_1k      |  3,321,638 |  >= 99.99 dB |       1.0000 |        11.19 dB |       11.990
===========================================================================
[INFO] Full report exported to: evaluation_report.csv
```

---

## Настройка под свои нужды

- **Добавить новые модели**: отредактируйте список `models_to_test` в `run_test.py`
- **Изменить метрики**: функция `calc_metrics()` легко расширяется (например, добавить LPIPS)
- **GPU-ускорение**: замените `device = torch.device('cpu')` на `cuda` при наличии видеокарты
- **Пакетная обработка**: скрипт автоматически обрабатывает все `.png/.jpg` в `inputs/`

---

## Источники и ссылки

- Оригинал модели: [cidautai/DarkIR](https://github.com/cidautai/DarkIR)
- Статья на arXiv: [DarkIR: Robust Low-Light Image Restoration](https://arxiv.org/abs/2412.13443)
- Страница на Hugging Face: [Cidaut/DarkIR](https://huggingface.co/Cidaut/DarkIR)

---