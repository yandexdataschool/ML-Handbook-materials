# Глава 10.4 — Трансформеры

Практические ноутбуки к главе: трансформер собирается с нуля, каждый блок
сверяется с эталонной реализацией на чистом numpy. Всё считается на CPU
(обучение — меньше минуты), GPU не нужен.

- `transformer_blocks.ipynb` — кирпичики архитектуры: эмбеддинги, positional
  encoding, scaled dot-product attention, multi-head attention, маски, FFN,
  Add & Norm. Для каждого блока: интуиция → формула → PyTorch → numpy-эталон →
  FLOPs.
- `transformer_training.ipynb` — сборка полной модели (энкодер-декодер),
  обучение на toy-задаче «разворот строки», greedy decoding, KV-cache,
  итоговая сводка сложности.

## Как запускать

Проще всего — открыть в Colab:
[transformer_blocks](https://colab.research.google.com/github/yandexdataschool/ML-Handbook-materials/blob/main/chapters/nlp/transformer_blocks.ipynb),
[transformer_training](https://colab.research.google.com/github/yandexdataschool/ML-Handbook-materials/blob/main/chapters/nlp/transformer_training.ipynb) —
зависимости там уже есть.

Локально:

```bash
conda env create -f environment.yml
conda activate ml-handbook-transformer
python -m ipykernel install --user --name ml-handbook-transformer
jupyter lab transformer_blocks.ipynb
```

## Про .md-файлы

Ноутбуки написаны в markdown и собираются из него через
[jupytext](https://jupytext.readthedocs.io/) — так их удобнее ревьюить
и хранить в git. Править лучше `.md`, а `.ipynb` пересобирать:

```bash
jupytext --to notebook --execute transformer_blocks.md -o transformer_blocks.ipynb
```

Код внутри 💡-врезок (цитат) при сборке не выполняется — это иллюстрации
к тексту врезок.
