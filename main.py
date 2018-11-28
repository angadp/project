'''
Created on Nov 26, 2018
@author: ishank
'''
#!/usr/bin/env python3
# boilerplate code by Jon May (jonmay@isi.edu)
import argparse
import sys
import codecs
if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip
import os.path
import gzip

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

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    infile = prepfile(args.infile, 'r')
    
    texts, labels = load_json(infile)
    
    for i in range(len(texts)):
        print(labels[i], texts[i])
        break

    from classifier import LogisticRegressionClassifier
    
    lrc = LogisticRegressionClassifier()
    lrc.train(texts, labels)
    predictions = lrc.predict(texts)
    
    count = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i] and labels[i]:
            count += 1
            print(labels[i], predictions[i])
    
    print(count)
    
    
    
#     outfile = prepfile(args.outfile, 'w')
#     for line in infile:
#         outfile.write(line)

if __name__ == '__main__':
    main()
