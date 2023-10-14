Тема: декомпозиция и анализ временных рядов методом SSA с тензорным разложением

*Аннотация*: Метод <<Гусеница>> или SSA --- популярный способ разложения одномерного временного ряда на состовляющие его сигналы. Он опирается на трансформацию ряда в высокомерное пространство и произведения над ним скелетонного разложения с помощью SVD. В данной работе исследуется обобщение данного метода на многомерные ряды с помощью тензорных разложений, проведение с ним численных экспериментов на различных данных (мультифазные колебания, траффик электроэнергии, акселерометрия) и сравнение с другими методами декомпозиции сигналов.

---

### Литература


[On Multivariate Singular Spectrum Analysis and its Variants](https://arxiv.org/pdf/2006.13448.pdf) - авторы исследуют mSSA через стэкинг траекторных матриц с моделью, где сигнал зашумлён и иногда значения сигналов не известны. Исследуется такой mSSA в плане качества оценки воостановления пропущеных значений и прогнозирования. Упоминается и тензорный вариант SSA как у нас, но он исследуется только на восстановление пропущеных значений.

[Multivariate singular spectrum analysis and the road to phase synchronization](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.036206) - закрытая статья по физике, где авторы исследуют классический mSSA для изучения "фазовую синхронизацию" в больших системах ...

[Tensor based Singular Spectrum Analysis for Automatic Scoring of Sleep EEG](https://irep.ntu.ac.uk/id/eprint/32719/1/PubSub10184_Sanei.pdf) - авторы преобразуют одномерный сигнал в матрицу через стэк неперсекающихся слайсов исходного сигнала (с фиксированным окном), далее слайсят эту матрицу "оконной" матрицей, стакая эти слайсы в тензор и применяют CPD. Применяют к обработке синтетики и к разным сигналам для анализа активности мозга во сне (electromyogram, electroencephalogram)

[Improved Tensor-Based Singular Spectrum Analysis Based on Single Channel Blind Source Separation Algorithm and Its Application to Fault Diagnosis](https://www.mdpi.com/2076-3417/7/4/418) - авторы улучшают предыущий метод в плане того, что сводят алгоритм декомпозиции к выпуклой оптимизации. Применяют к декомпозиции одномерного сигнала диагностики поломок механических конструкций

[Tensor based singular spectrum analysis for nonstationary source separation](https://ieeexplore.ieee.org/abstract/document/6661921) - абсолютно тот же подход, что и в предыдущих статьях. Но вроде опубликовано раньше.

[Tensor Singular Spectral Analysis for 3D feature extraction in hyperspectral images](https://rgu-repository.worktribe.com/preview/1972801/FU%202023%20Tensor%20singular%20spectral%20%28AAM%29.pdf) - авторы используют тензорный SSA для анализа гиперспектральных изображений (в них каждый писель хранит информацию о спектре, а не просто интенсивности), т.к. он позволяет анализировать пространственную и спектральную структуру одновременно

[Three-dimensional singular spectrum analysis for precise land cover classification from UAV-borne hyperspectral benchmark datasets](https://www.sciencedirect.com/science/article/pii/S0924271623001946) - что-то типа того же, с неким предложенным фрэймворком для оптимизации памяти при применении тензорных операций. Применение на большом датасете.