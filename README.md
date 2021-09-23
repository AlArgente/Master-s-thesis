# Master's thesis
### Trabajo de Fin de Master (Master's thesis) Fake News Detection

Se añadirá el código que se va a desarrollar para el TFM. La base de datos está disponible en [Zenodo: Proppy Corpus 1.0](https://zenodo.org/record/3271522#.XS6qRUUzau4)

La versión de Python utilizada ha sido Python 3.8.2. El sistema operativo utilizado ha sido Ubuntu 20.04 LTS. El framework que se ha utilizado es [TensorFlow](http://tensorflow.org/), en concreto la versión 2.2.0. 

Se ha desarollado un modelo base usando una [BiLSTM](https://github.com/AlArgente/TFM/blob/master/code/cnnrnn_model.py) y usando los embeddings de [Glove](https://nlp.stanford.edu/projects/glove/)/[FastText](https://fasttext.cc/), obteniendo mejor resultado con FastText que con Glove.

También se han desarrollado modelos de atención, uno basado en la atención que utilizan los transformers ([attention_model.py](https://github.com/AlArgente/TFM/blob/master/code/attention_model.py)) y otro que utiliza tanto atención local como atención global ([mean_model.py](https://github.com/AlArgente/TFM/blob/master/code/mean_model.py)). 

Se ha desarrollado un modelo basado en transformers ([modeltransformer.py](https://github.com/AlArgente/TFM/blob/master/code/modeltransformer.py)), en el que sólo se ha utilizado en encoder del transformer, pero no se ha llegado a profundidar en su uso y por tanto no se ha preparado un buen modelo, sí bien se tiene una base para el futuro.

Se ha decidido utilizar el modelo de BERT para realizar fine-tuning en la predicción de FakeNews. Además puede servir de base para utilizar Albert u otros modelos pre-entrenados sin tener que realizar muchos cambios. El fichero de este modelo es [bertmodel.py](https://github.com/AlArgente/TFM/blob/master/code/bertmodel.py)

Para poder ejecutar este ćodigo es necesario disponer de los embeddings de Glove y FastText. Una vez se tengan los archivos, se debe actualizar el path del fichero 'embeddings.py'.

### Master's thesis

This master's thesis is about resolving a problem about Fake News detection. The problem is called Propaganda Detection, and the news aren't 100% fake but they try to convince readers to think alike. To reach this goal, the authors of this news use misinformation if needed to get more people by their side. The dataset I'm working with have news from the 2016 U.S. elections, and the data is divided into propaganda/non-propaganda. The data is available in [Zenodo: Proppy Corpus 1.0](https://zenodo.org/record/3271522#.XS6qRUUzau4).

To solve this problem I've developed a base model using a [BiLSTM](https://github.com/AlArgente/TFM/blob/master/code/cnnrnn_model.py) and using two type of embeddings [Glove](https://nlp.stanford.edu/projects/glove/) and [FastText](https://fasttext.cc/), getting better results with FastText than with Glove. 

To improve the BiLSTM, I added some attention mechanism, local, global and self attention. The best model is the model that uses local attention ([mean_model.py](https://github.com/AlArgente/TFM/blob/master/code/mean_model.py)). Also I've built a transformer model ([attention_model.py](https://github.com/AlArgente/TFM/blob/master/code/attention_model.py)), but it didn't work well for this problem.

Also I did transfer learning, fine-tuning BERT and to use it for the Propaganda Detection problem. The code for BERT is at  [bertmodel.py](https://github.com/AlArgente/TFM/blob/master/code/bertmodel.py). Also this could work as a base architecture to use other pre-trained models based on BERT, like DistilBERT, Albert, etc.

To execute the code it's necessary to have the Glove/FastText downloaded, and put the correct path in the embeddings file (embeddings.py).
