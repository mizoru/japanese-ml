import os
f = open('config.txt', 'r')
for l in f.readlines():
	os.system('python3 pattern_classification_train.py ' + l + 'tee -a log')
