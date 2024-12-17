# Text Detection

**General Algorithm of the applied Text Detection**
1. Estimation of average text height using EAST
2. Morphological transformation finds text regions.
3. Visualization and cropping out text regions for further analysis.

Morphological kernel is necessary to identify lines of text. Its width is proportional to the height of the text and also conventionally kernel size must be an **odd number**.

The approach combines the advantages of deep model (EAST) with classical image processing methods.

**Limitations**  
The method is designed to detect individual words on white paper, where the lines of text are written horizontally. This is a limitation, because handwritten text can be written in different structures, in a circle, in tables, and there can be different formulas and this requires a different research.

---

# Description

## Tesseract 

On printed text  
![printed_tesseract.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/tesseract/printed_tesseract.png)

On handwritten text  
![handwritten_tesseract.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/tesseract/handwritten_tesseract.png)


## EAST

Мы используем 2 метода: Morphological Transformations, EAST Detector.

East detector расчитан больше на machine printed text. 

Показать примеры  
C:\git\HTR\htr\detection\my_comparisons\EAST_IMAGES

On printed text  
![printed text detection](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/EAST_IMAGES/example2_.jpg)

On handwritten text  
![handwritten text detection](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/EAST_IMAGES/gnhk_019_.png)

![difficult handwritten text detection](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/EAST_IMAGES/gnhk_015_.png)

Можно заметить, что если EAST нашел слово, то это достаточно вероятно является словом, но он часто пропускает слова. Мы используем это в нашем подходе обнаружения текста.

Одним из интересных моментов - EAST может находить слова расположенные под углом.

![round text detection](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/EAST_IMAGES/round_.jpg)


## Our Approach

Morphological Transformations являются heuristic подходом, который состоит из нескольких этапов: Canny Edge Detector, Morphological Dilation followed by Erosion (Closing). 

Метод детекции расчитан на слова написанные горизонтально на более менее чистой бумаге, не на бумаге в клетку и без различных рисунков.

### Canny

Показать картинки Canny  
HTR\htr\detection\my_comparisons\morpho

![Canny](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/morpho/Canny.png)


### Morphological Dilation followed by Erosion (Closing)**  
https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

Original  
![image](https://github.com/user-attachments/assets/ad661286-df81-4722-8fd1-b0fc60880c2d)

Dilation  
![image](https://github.com/user-attachments/assets/d25f49ea-deb4-4af2-b040-df13b0aa118e)

Erosion  
![image](https://github.com/user-attachments/assets/23667997-13a6-4c3a-b259-d29e38cc7f9a)

Closing  
![image](https://github.com/user-attachments/assets/710a3dd2-a721-4a3a-9c87-e3cc343d7d18)

Главный параметр в нашем случае это ширина ядра. В зависимости от ширины мы можем находить строки, слова и так далее. 
Как я уже писал можно заметить, что если EAST нашел слово, то это достаточно вероятно является словом, но он часто пропускает слова. Можно взять среднюю высоту найденного слово и сделать ширину ядра для Closing пропорциональным этой средней высоте. Мы предполагаем, что между контурами больше avg_height/2, то это разные слова. Для поиска линий мы используем c x avg_height, c > 2.


Показывать картинки
HTR\htr\detection\my_comparisons\morpho

closed15x7.png  
![closed15x7.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/morpho/closed15x7.png)

boxes15x7.png  
![boxes15x7.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/morpho/boxes15x7.png)


closed50x7.png  
![closed50x7.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/morpho/closed50x7.png)

boxes50x7.png  
![boxes50x7.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/morpho/boxes50x7.png)

closed150x7.png  
![closed150x7.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/morpho/closed150x7.png)

boxes150x7.png  
![boxes150x7.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/morpho/boxes150x7.png)

closed500x7.png  
![closed500x7.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/morpho/closed500x7.png)

boxes500x7.png  
![boxes500x7.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/morpho/boxes500x7.png)


Morphological kernel is necessary to identify lines of text. Its width is proportional to the height of the text, that we estimate using EAST Detector, and also conventionally kernel size must be an odd number.

HTR\htr\detection\my_comparisons\our_approach

![steps.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/our_approach/steps.png)

![result.png](https://github.com/alarxx/HTR/blob/main/htr/detection/my_comparisons/our_approach/result.png)


# Text Recognition

1) Alphabet Recognition
2) Word Recognition
