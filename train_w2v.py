#!/usr/bin/python3

import os
import time
import string
import numpy as np
import _pickle as cPickle
import argparse
import multiprocessing


from tqdm import tqdm
from tqdm import trange

from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec

#====================

blank = 'XXXXX'
data_path = 'AI_Course_Final/CBTest/data'
full_text_save_path = './data/full.txt'
w2v_save_path = './data/cbtest_w2v.model'

max_sentence_length = 216

w2v_params = {  'size': 300,
                'window': 10,
                'min_count': 10,
                'workers': max(1, multiprocessing.cpu_count() - 2),
                'sample': 1E-5}

replace_map = {
            'ca n\'t': 'can not',
            'do n\'t': 'do not',
            'does n\'t': 'does not',
            'wo n\'t': 'will not',
            'sha n\'t': 'shall not',
            'ai n\'t': 'am not',
            'n\'t': 'not',
            '\'s': '',
            '-LRB-': '',
            '-RRB-': '',
        }

exclusive_vocab = []

train_list = ['cbtest_CN_train.txt', 
            'cbtest_NE_train.txt', 
            'cbtest_P_train.txt', 
            'cbtest_V_train.txt']

valid_list = ['cbtest_CN_valid_2000ex.txt',
            'cbtest_NE_valid_2000ex.txt',
            'cbtest_P_valid_2000ex.txt',
            'cbtest_V_valid_2000ex.txt']

test_list = ['cbtest_CN_test_2500ex.txt',
            'cbtest_NE_test_2500ex.txt',
            'cbtest_P_test_2500ex.txt',
            'cbtest_V_test_2500ex.txt']

#====================

#====================

def read_data(filename):

    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False
    lines = []
    with open(filename, 'r') as file:
        raw_lines = file.readlines()
        for line in tqdm(raw_lines, desc='Read', ncols=80):
            if is_int(line.split(' ', 1)[0]):
                line = line.lower()
                for key, rep in replace_map.items():
                    line = line.replace(key, rep)
                lines.append(line)
    return lines
 

def parse_questions(lines, has_answer=True):
    question_list = []

    assert(len(lines) % 21 == 0)

    exc = set(string.punctuation)

    def parse(lines, has_answer):
        sent = []
        def filter(line):
            line = ''.join(ch if ch not in exc else ' ' for ch in line)
            return ' '.join(line.split())

        for i in range(20):
            line = lines[i].split(' ', 1)[1]
            sent.append( filter(line) )

        last = [x for x in lines[-1].split('\t') if x!='' ]
        q = filter(last[0].split(' ', 1)[1]).replace(blank.lower(), '<blank>')
        option = []
        question_with_option = []

        # check if has answer
        if has_answer:
            a = ''.join(ch for ch in last[1] if ch not in exc)
            raw_option = last[2].split('|')
        else:
            a = ''
            raw_option = last[1].split('|')


        if len(raw_option) > 10:
            raw_option = raw_option[0:10]
        elif len(raw_option) < 10:
            raw_option = raw_option + ['']*(10-raw_option)

        for x in raw_option:
            o = ''.join(ch for ch in x.replace('\n', '') if ch not in exc)
            option.append(o)
            o = q.replace('<blank>', o)
            question_with_option.append(o)

        assert(len(sent) == 20)
        assert(len(option) == 10)

        if has_answer:
            idx = option.index(a)
        else:
            idx = 0

        question = {'sentences': sent,
                    'question': q, 
                    'answer': a,
                    'options': option,
                    'queries': question_with_option,
                    'index': idx}
        return question

    for i in trange(len(lines) // 21, desc='Parsing', ncols=80):
        question_list.append(parse(lines[ i*21 : (i+1)*21], has_answer))

    return question_list

def save_questions_to_full_text(text_path, questions, has_answer=True):
    if not os.path.exists(os.path.dirname(text_path)):
        os.makedirs(os.path.dirname(text_path))

    with open(text_path, 'w') as out:
        for question in tqdm(questions, desc='Q', ncols=80):
            for sent in question['sentences']:
                out.write(sent + '\n')
            index = question['index']
            out.write(question['queries'][index] + '\n')

def parse_questions_to_full_text(txt, lines, has_answer=True):
    question_list = parse_questions(lines, has_answer)

    with open(txt, 'w') as out:
        for question in question_list:
            for sent in question['sentences']:
                out.write(sent)
            index = question['index']
            out.write(question['queries'][index])

def train_w2v_model(text_path, model_path, max_length, params):
    word2vec = Word2Vec(LineSentence(text_path, max_sentence_length=max_length), **params)

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    print('===== SUMMARY =====')
    print('Embed size: ', params['size'])
    print('Vocab size: ', len(word2vec.wv.vocab))


    print('Saving model to {}'.format(model_path))
    print('Saving text model to {}'.format(model_path+'.txt'))

    word2vec.save(model_path)
    word2vec.wv.save_word2vec_format(model_path + '.txt', binary=False)

if __name__ == '__main__':
    
    print('Parsing input data...')

    all_questions = []

    for name in train_list + valid_list + test_list:
        n, _ = os.path.splitext(name)
        filename = os.path.join(data_path, name)
        print('Reading data:', filename)
        
        lines = read_data(filename)
        print('Read {} lines'.format(len(lines)))
        print('Parsing data:', filename)

        question_list = parse_questions(lines)

        print('Result: {}/{} (actual/expected)'.format(len(question_list), len(lines)//21))

        all_questions.extend(question_list)

    print('Total: {} questions'.format(len(all_questions)))
    print('Saving questions to full text...')

    save_questions_to_full_text(full_text_save_path, all_questions, has_answer=True)

    del all_questions
    
    print('Training word2vec model...')

    train_w2v_model(full_text_save_path, w2v_save_path, max_sentence_length, w2v_params)


