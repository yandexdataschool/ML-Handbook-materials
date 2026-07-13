# Глава 12.2 — Variational Autoencoder (VAE)

`vae.ipynb` — практический ноутбук к главе: VAE и CVAE на MNIST. Всё считается на CPU за несколько минут, GPU не нужен.

В ноутбуке с нуля собирается VAE (энкодер, репараметризация, бернуллиевский декодер, ELBO) и воспроизводятся ключевые картинки главы: реконструкции, семплы из априорного распределения, карта латентного пространства при $M = 2$, выученное многообразие, влияние размерности латента и, наконец, CVAE с обуславливанием на класс цифры. В конце — упражнения.

## Как запускать

Проще всего — [открыть в Colab](https://colab.research.google.com/github/yandexdataschool/ML-Handbook-materials/blob/main/chapters/representation_learning/vae.ipynb): зависимости там уже есть.

Локально:

```bash
conda env create -f environment.yml
conda activate ml-handbook-vae
python -m ipykernel install --user --name ml-handbook-vae
jupyter lab vae.ipynb
```

## Про vae.md

Ноутбук написан в markdown (`vae.md`) и собирается из него через [jupytext](https://jupytext.readthedocs.io/) — так его удобнее ревьюить и хранить в git. Править лучше `vae.md`, а `vae.ipynb` пересобирать:

```bash
jupytext --to notebook --execute vae.md -o vae.ipynb
```
