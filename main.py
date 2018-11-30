'''
Created on Nov 26, 2018
@author: ishank
'''
#!/usr/bin/env python3
# boilerplate code by Jon May (jonmay@isi.edu)
from __future__ import print_function
import argparse
import sys
import codecs
if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip
import os.path
import gzip
import numpy as np
from sklearn import metrics

scriptdir = os.path.dirname(os.path.abspath(__file__))

reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')

def load_json(infile):
    import json
    texts, labels = [], []
    
    try:
        for row in json.load(infile):
            if len(row['text']) > 0:
                texts.append(str(row['text']))
                labels.append(1 if row['NSFW'] == "True" else 0)            
    except Exception as e:
        print("Failed to load JSON data. ", e)
    
    return texts, labels

def prediction_metrics(labels, truths):
    truths_np = np.array(truths)    
    
#     accuracy = np.mean(truths_np == labels)    
#     print("Accuracy: ", accuracy)

    full_report = metrics.classification_report(truths_np, labels, target_names=['SFW', 'NSFW'])
    print(full_report)
    print(metrics.f1_score(truths_np, labels))
#     report = metrics.f1_score(truths_np, labels)
#     return report
    
    
def prepfile(fh, code):
    if type(fh) is str:
        fh = open(fh, code)
    ret = gzip.open(fh.name, code if code.endswith("t") else code + "t") if fh.name.endswith(".gz") else fh
    if sys.version_info[0] == 2:
        if code.startswith('r'):
            ret = reader(fh)
        elif code.startswith('w'):
            ret = writer(fh)
        else:
            sys.stderr.write("I didn't understand code " + code + "\n")
            sys.exit(1)
    return ret

def addonoffarg(parser, arg, dest=None, default=True, _help="TODO"):
    ''' add the switches --arg and --no-arg that set parser.arg to true/false, respectively'''
    group = parser.add_mutually_exclusive_group()
    dest = arg if dest is None else dest
    group.add_argument('--%s' % arg, dest=dest, action='store_true', default=default, help=_help)
    group.add_argument('--no-%s' % arg, dest=dest, action='store_false', default=default, help="See --%s" % arg)

def main():
    
    parser = argparse.ArgumentParser(
        description="Classifying NSFW or not", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    addonoffarg(parser, 'debug', _help="debug mode", default=False)
    parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
    parser.add_argument("--classifier", "-c", nargs='?', type=int, default=1, help="Choose a classifier")
    
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    
    c_type = args.classifier
    if not c_type or c_type > 5: 
        from classifier import NaiveClassifier as Classifier
        print("Naive Classifier")
    elif c_type == 1: 
        from classifier import LinearSVM as Classifier
        print("Linear SVM Classifier")
    elif c_type == 2: 
        from classifier import LogisticRegressionClassifier as Classifier
        print("Logistic Regression Classifier")
    elif c_type == 3: 
        from classifier import AdaboostClassifier as Classifier
        print("AdaBoost Classifier")
    elif c_type == 4: 
        from classifier import RandomForestClassifier as Classifier
        print("Random Forest Classifier")
    elif c_type == 5: 
        from classifier import EnsembleVotingClassifier as Classifier
        print("Ensemble Voting Classifier")
    
    infile = prepfile(args.infile, 'r')
    texts, labels = load_json(infile)
    
#     alphas = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.1e-4, 1.2e-4, 1.3e-4, 1.4e-4, 1.5e-4]    
#     for alpha in alphas:

    lrc = Classifier()
    lrc.train(texts[:8500], labels[:8500])
    
    print("Train\n-------------")
    predictions = lrc.predict(texts[:8500])    
    prediction_metrics(predictions, labels[:8500])
        
    print("Test\n-------------")
    predictions = lrc.predict(texts[8501:])    
    prediction_metrics(predictions, labels[8501:])
        
#     print("[\"EnsembleVotingClassifier\"", ",", tr, ",", te, "]")
#     outfile = prepfile(args.outfile, 'w')
#     for line in infile:
#         outfile.write(line)

if __name__ == '__main__':
    main()
