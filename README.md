# TFM
Trabajo de Fin de Master (Master's thesis)

Se añadirá el código que se va a desarrollar para el TFM. La base de datos está disponible en [Zenodo: Proppy Corpus 1.0](https://zenodo.org/record/3271522#.XS6qRUUzau4)

La versión de Python utilizada ha sido Python 3.8.2. El sistema operativo utilizado ha sido Ubuntu 20.04 LTS. El framework que se ha utilizado es [TensorFlow](http://tensorflow.org/), en concreto la versión 2.2.0. 

Se ha desarollado un modelo base usando una [BiLSTM](https://github.com/AlArgente/TFM/blob/master/code/cnnrnn_model.py) y usando los embeddings de [Glove](https://nlp.stanford.edu/projects/glove/)/[FastText](https://fasttext.cc/), obteniendo mejor resultado con FastText que con Glove.

También se han desarrollado modelos de atención, uno basado en la atención que utilizan los transformers ([attention_model.py](https://github.com/AlArgente/TFM/blob/master/code/attention_model.py)) y otro que utiliza tanto atención local como atención global ([mean_model.py](https://github.com/AlArgente/TFM/blob/master/code/mean_model.py)). 

Se ha desarrollado un modelo basado en transformers ([modeltransformer.py](https://github.com/AlArgente/TFM/blob/master/code/modeltransformer.py)), en el que sólo se ha utilizado en encoder del transformer, pero no se ha llegado a profundidar en su uso y por tanto no se ha preparado un buen modelo, sí bien se tiene una base para el futuro.

Se ha decidido utilizar el modelo de BERT para realizar fine-tuning en la predicción de FakeNews. Además puede servir de base para utilizar Albert u otros modelos pre-entrenados sin tener que realizar muchos cambios. El fichero de este modelo es [bertmodel.py](https://github.com/AlArgente/TFM/blob/master/code/bertmodel.py)

