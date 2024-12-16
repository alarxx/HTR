# HTR
> Handwritten Text Recognition

---

## Previous projects

- [Handwriting-Recognition](https://github.com/alarxx/Handwriting-Recognition)
- [Tensor-library](https://github.com/alarxx/Tensor-library)


## Description

For **[Handwriting Recognition (HWR)](https://en.wikipedia.org/wiki/Handwriting_recognition):**
- [Optical Character Recognition (OCR)](https://en.wikipedia.org/wiki/Optical_character_recognition)
- [Intelligent character recognition (ICR)](https://en.wikipedia.org/wiki/Intelligent_character_recognition)
- [Intelligent Word Recognition (IWR)](https://en.wikipedia.org/wiki/Intelligent_word_recognition)

**OCR** engines are primarily focused on character-by-character machine printed text recognition from a scanned document and **ICR** for different fonts or even handwritten text:

![Примеры_pages-to-jpg-0002](https://github.com/user-attachments/assets/f4a0ae3a-7c10-47f5-8640-e374a3f31986)

**Intelligent Word Recognition (IWR)** is the recognition of unconstrained handwritten words. IWR recognizes entire handwritten words or phrases instead of character-by-character, like its predecessors. (our aim).

---

**On-line:**
- pen-based computer screen surface
- pen-up and pen-down switching
- pen pressure
- velocity/changes of writing direction
- specifically

**Off-line:** (our aim)
- piece of paper
- image

**Problem**
Input data: Image
Output data: written text on the image

Sub-tasks:
- Text Detection
- Classification

---

Firstly, [[Text Detection]].

**General Algorithm of the applied Text Detection**
1. Estimation of average text height using EAST
2. Morphological transformation finds text regions.
3. Visualization and cropping out text regions for further analysis.

Morphological kernel is necessary to identify lines of text. Its width is proportional to the height of the text and also conventionally kernel size must be an **odd number**.

The approach combines the advantages of deep model (EAST) with classical image processing methods.

**Limitations**
The method is designed to detect individual words on white paper, where the lines of text are written horizontally. This is a limitation, because handwritten text can be written in different structures, in a circle, in tables, and there can be different formulas and this requires a different research.


## Datasets

- CMNIST: https://github.com/bolattleubayev/cmnist
- KOHTD: Kazakh Offline Handwritten Text Dataset: https://github.com/abdoelsayed2016/KOHTD

## Future research
- Circular text recognition
- Table detection and structure recognition
- Sequential patterns of texts for recognition or post-processing
- End-to-end model for text detection and recognition
- Speech Recognition


## Authors

* **Alar Akilbekov** - [alarxx](https://github.com/alarxx) - [@alarxx](https://t.me/alarxx)
* **Baktiyar Toksanbay**

## Licence 

[Mozilla Public License](https://github.com/alarxx/HTR/blob/main/LICENSE)

### Exhibit A - Source Code Form License Notice
```
This file is part of the HTR Project – a demo pipeline for Intelligent Word Recognition of unconstrained handwritten text.

Copyright © 2024 Alar Akilbekov, Baktiyar Toksanbay
All Rights Reserved.

SPDX-License-Identifier: MPL-2.0
--------------------------------
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

Contact: alar.akilbekov@gmail.com
```


## References:

Fundamental Books:
- Bishop, C. M., & Nasrabadi, N. M. (2006). Pattern recognition and machine learning (Vol. 4, No. 4, p. 738). New York: springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. The MIT press.
- Raschka, S., Liu, Y., Mirjalili, V., & Dzhulgakov, D. (2022). Machine learning with PyTorch and Scikit-Learn: Develop machine learning and deep learning models with Python. Packt.
- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). Dive into deep learning. Cambridge University Press.

Beginner level books:
- Rashid, T. (2016). Make Your own neural network. CreateSpace Independent Publishing Platform.
- Weidman, S. (2019). Deep learning from scratch: Building with Python from first principles (First edition). O’Reilly Media, Inc.
- Patterson, J., & Gibson, A. (2017). Deep learning: A practitioner’s approach (First edition). O’Reilly.

OpenCV:
- Kaehler, A., & Bradski, G. (2016). Learning OpenCV 3: computer vision in C++ with the OpenCV library. " O'Reilly Media, Inc.".
- Szeliski, R. (2022). Computer vision: algorithms and applications. Springer Nature.
- Прохоренок, Н. А. (2018). OpenCV и Java. Обработка изображений и компьютерное зрение. БХВ-Петербург.

YouTube:
- Евгений Разинков. (2023). Machine Learning (2023, spring). https://www.youtube.com/playlist?list=PL6-BrcpR2C5SCyFvs9Xojv24povpBCI6W
- Евгений Разинков. (2022). Лекции по машинному обучению (осень, 2022). https://www.youtube.com/playlist?list=PL6-BrcpR2C5QYSAfoG8mbQUsI9zPVnlBV
- Евгений Разинков. (2021). Лекции по Advanced Computer Vision (2021). https://www.youtube.com/playlist?list=PL6-BrcpR2C5RV6xfpM7_k5321kJrcKEO0
- Евгений Разинков. (2021). Лекции по Deep Learning. https://www.youtube.com/playlist?list=PL6-BrcpR2C5QrLMaIOstSxZp4RfhveDSP
- Евгений Разинков. (2020). Лекции по компьютерному зрению. https://www.youtube.com/playlist?list=PL6-BrcpR2C5RZnmIWs6x0C2IZK6N9Z98I
- Евгений Разинков. (2019). Лекции по машинному обучению. https://www.youtube.com/playlist?list=PL6-BrcpR2C5RYoCAmC8VQp_rxSh0i_6C6
