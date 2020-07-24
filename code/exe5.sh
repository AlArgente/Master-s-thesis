#!/bin/bash

for i in {1..5}
do
	python3 main.py --mode=4 > "resultados_experimento_2_$i.txt"
done
