# Deep Learning Method Implementaion for Image Car Model Classification

В лабораторой работе

## EfficientNet

Начнём с архитектуры EfficientNet **[1]**. Основной её строительные блоки это *Conv* и *MBConv*.

![image](https://user-images.githubusercontent.com/29786176/207859678-7d949276-861f-48ec-aee5-b2a9513350ae.png)

*Conv* -- это обыкновеннный свёрточный слой (convolution layer) **[2]**.

*MbConv* -- это слой, впервые представленный в статье по MobileNetV2 **[3]**. Данный слой состоит из свёрточного слоя $1 \times 1$ с $tk$ фильтрами, где $k$ -- это количество входных слоёв, а $t$ -- фактор расширения (гиперпараметр). Полученный результат отправляется в слой активации после чего передаётся в depthwise свёрточный слой $H \times W$ с $tk$ фильтрами ($H$ и $W$ гиперпараметры). Результат затем обрабатывается в слое активации и передаётся в свёртку $1 \times 1$ с $k'$ фильтрами ($k'$ гиперпараметр).

![image](https://user-images.githubusercontent.com/29786176/207866300-56a85a7d-d2f4-49f3-8048-35374de65786.png)

В отличии от *MobileNetV2* в *EfficientNet* в качестве функции активации используется не *RiLU6*, а *SiLU*.

![image](https://user-images.githubusercontent.com/29786176/207873177-f06d439c-3a3f-4aea-9cad-e5c78f7795ba.png)

Также в реализации *MBConv*, предаставленной в *EfficientNet*, добавлен squeeze and excitation block **[4]** перед последним свёрточным слоем. В squeeze and excitation block расчитывается вес важности для каждого слоя входной карты признаков. 

Помимо этого В MBConv, предствленной в *EfficientNet*, используется концепция остаточного слоя (residual connection **[5]**). В случае если входная и выходная карта признаков имеют одинаковое количество каналов и одинаковый размер (в моей реализации $stride == 1$), то применяется операция с рандомным занулением одного из сэмплов выходной карты признаков (*DropSample* в предствленной мною реализации) и затем полученная карта признаков складывается с входной картой признаков. Стоит отметить, что DropSample является аналогом Dropout, но распространяется на сэмплы, а не на веса модели (происходит зануление всех значений сэмпла).

Также в *EfficientNet* применяется батч нормализация после каждого свёрточного слоя и depthwise свёрточного слоя перед слоем с активацией (если функции активации нет, то нормализация всё равно применяется).

После всех свёрточных сетей используется average pooling для дальнейшей передачи в полносвязные слои, которые выступают в роли классификатора.

## NAdam

NAdam является примером применения моментов нестерова к оптимизатору Adam.

## Результаты

### Реализация EfficientNet на Numpy (Cupy)
Время расчёта одной тренировочной эпохи реализации на *Numpy* (*Cupy*) вышло ~105 минут при отсутствии в данной реалиации squeeze and excitation block, DropSample, батч нормализации и остаточных слоёв **[8]**. Расчёты производились на процессоре Intel(R) Xeon(R) 2.00GHz с двумя доступными потоками и видеокарте NVIDIA Tesla P100.

## Реализация на Pytorch
![W B Chart 12_15_2022, 5_58_22 PM](https://user-images.githubusercontent.com/29786176/207893855-c25e387f-d623-4fe2-9593-cde3f65a89b9.png)
![W B Chart 12_15_2022, 5_59_19 PM](https://user-images.githubusercontent.com/29786176/207893926-9d6ca9ae-9266-433b-877e-c6c8423876dc.png)
![W B Chart 12_15_2022, 5_59_34 PM](https://user-images.githubusercontent.com/29786176/207893933-93616729-458e-4600-982e-6ec13b84f8a1.png)
![W B Chart 12_15_2022, 5_59_48 PM](https://user-images.githubusercontent.com/29786176/207893944-7e7083d7-c025-49d4-a570-8c778cce0fe5.png)

Модель обучалась с нуля (предобученные веса не использовались). Как можно заметить, c Adam получилось добиться более скорой сходимости в сравнении с NAdam. При практически вдвое меньшем количестве эпох результаты с Adam сравнимы с результатами NAdam.

По графикам Train Loss и Val Loss можно заметить, что модель не была склонна к переобучению на обучающей выборке. 

Точность модели в основном колебалась от 0.0001 до 0.0003 на валидацилнной выборке и тенденции к улучшению результата с уменьшением значения функции ошибки выявлено не было.

## Источники
1. Tan M., Le Q. Efficientnet: Rethinking model scaling for convolutional neural networks //International conference on machine learning. – PMLR, 2019. – С. 6105-6114.
2. Sandler M. et al. Mobilenetv2: Inverted residuals and linear bottlenecks //Proceedings of the IEEE conference on computer vision and pattern recognition. – 2018. – С. 4510-4520.
3. Hu J., Shen L., Sun G. Squeeze-and-excitation networks //Proceedings of the IEEE conference on computer vision and pattern recognition. – 2018. – С. 7132-7141.
4. Kingma D. P., Ba J. Adam: A method for stochastic optimization //arXiv preprint arXiv:1412.6980. – 2014.
5. Dozat T. Incorporating nesterov momentum into adam. – 2016.