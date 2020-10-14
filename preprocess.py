from konlpy.tag import Okt
from nltk.tokenize import word_tokenize
from gensim import models
from collections import defaultdict
import sys
import os
import argparse
import pandas as pd
from file_path import *
import random

def parse_args(argv):
    parser = argparse.ArgumentParser(description='make file used in vecmap')
    parser.add_argument('--ko_origin', default=ko_origin, help='origin file do you want to make vector')
    parser.add_argument('--en_origin', default=en_origin, help='origin file do you want to make vector')
    parser.add_argument('--ko_token', default=ko_token, help='what file do you want to make vector')
    parser.add_argument('--en_token', default=en_token, help='what file do you want to make vector')
    parser.add_argument('--vocab_size', type=int, default=50000, help='the size of the vocab.')
    parser.add_argument('--vec_size', type=int, default=200000, help='the size of the (embedding)vec.')

    parser.add_argument('--dic', default=dic, help='ko-en 단어set 파일')
    parser.add_argument('--ko_fasttext', default=ko_fasttext, help='fasttext 파일경로')
    parser.add_argument('--en_fasttext', default=en_fasttext, help='fasttext 파일경로')
    parser.add_argument('--ko_emb', default=ko_emb, help='output of make_wordvecor fuction')
    parser.add_argument('--en_emb', default=en_emb, help='output of make_wordvecor fuction')
    parser.add_argument('--cuda', action='store_true', help='when running vecmap use cuda (requires cupy)')

    parser.add_argument('--n_train', type=int, default=100000, help='train set for undreamt')
    parser.add_argument('--n_test', type=int, default=10000, help='test set for undreamt')

    parser.add_argument('--ko_mapped', default=ko_mapped, help='train set for undreamt')
    parser.add_argument('--en_mapped', default=en_mapped, help='train set for undreamt')

    return parser.parse_args(argv[1:])

#원본 data로 부터 토큰화된 데이터 셋 생성하기
def make_ko_corpus(inputdata_path, output_path):
    okt = Okt()
    with open(output_path, 'wt', encoding='utf-8') as f:
        for path in inputdata_path:
            df = pd.read_excel(path)
            sentences = df['원문'].tolist()
            print(path, len(sentences))
            for sent in sentences:
                f.write(' '.join(okt.morphs(sent.strip(), norm=False, stem=False))+'\n')
    print('corpus 전처리 완료')

#원본 data로 부터 토큰화된 데이터 셋 생성하기
def make_en_corpus(inputdata_path, output_path):
    with open(output_path, 'wt', encoding = 'utf-8') as fw:
        with open(inputdata_path, 'rt', encoding='utf-8') as fr:
            for _ in range(10**7):
                sent = fr.readline()
                fw.write(' '.join(word_tokenize(sent.strip().lower()))+'\n')
    print('corpus 전처리 완료')

def ready_train_test_set(corpus, num_train, num_test, shuffle=True): #train set for undreamt
    with open(corpus, 'rt', encoding='utf-8') as fr:
        sentences = fr.readlines()
        if shuffle:
            random.shuffle(sentences)
        with open(corpus[:-4]+'train.txt', 'wt', encoding='utf-8') as fw:
            fw.writelines(sentences[:num_train])
        with open(corpus[:-4]+'test.txt', 'wt', encoding='utf-8') as fw:
            fw.writelines(sentences[num_train:num_train+num_test])
        print('train {}개의 문장\ntest set {}개의 문장 생성완료\n'.format(num_train, num_test))

#tensorflow keras preprocessing.text의 Tokenizer의 기능 중 word_counts의 기능이 필요했다
#따라서 class의 형식만 빌려 기능구현하여 사용하였다 
class Tokenizer:
    def __init__(self, num_words=None, oov_token=None):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_counts = []
    def fit_on_texts(self, sentences):
        self.word_counts = defaultdict(lambda : 0)
        for sent in sentences:
            for token in sent.strip().split():
                self.word_counts[token] = self.word_counts[token] + 1


#corpus : tokened_txt/fasttext를 이용하여 vector화 할 단어집합선별
class Corpus:
    def __init__(self, ko_corpus, en_corpus, vocab_size, num_words):
        self.ko_corpus = ko_corpus
        self.en_corpus = en_corpus
        self.vocab_size = vocab_size
        self.num_words = num_words
        with open(ko_corpus, 'rt', encoding='utf-8') as f:
            ko_sentences = f.readlines()
        self.ko_tokenizer = Tokenizer(num_words=self.num_words, oov_token=None)
        self.ko_tokenizer.fit_on_texts(ko_sentences)

        with open(en_corpus, 'rt', encoding='utf-8') as f:
            en_sentences = f.readlines()
        self.en_tokenizer = Tokenizer(num_words=self.num_words, oov_token=None)
        self.en_tokenizer.fit_on_texts(en_sentences)

    def list_up(self, tokenizer):
        wordcount_list = list(tokenizer.word_counts.items())
        wordcount_list.sort(key=lambda x: x[1], reverse=True)
        wordcount_list = [i[0] for i in wordcount_list]
        print(len(wordcount_list))
        assert len(wordcount_list)>self.num_words, "corpus가 부족합니다"
        return wordcount_list[:self.num_words]

    def treat_dictionary(self, dicpath):
        self.ko_word_list = self.list_up(self.ko_tokenizer)
        self.en_word_list = self.list_up(self.en_tokenizer)
        ko_word_final = self.ko_word_list[:self.vocab_size]
        en_word_final = self.en_word_list[:self.vocab_size]

        # corpus에 없는 단어들은 dictionary에서 삭제
        with open(dicpath,'rt', encoding='utf-8') as f:
            lines = f.readlines()
        with open(dicpath,'wt', encoding='utf-8') as f:
            cn = 0
            for i in lines:
                ko, en = i.strip().split()
                if ko in self.ko_word_list and en in self.en_word_list:
                    cn+=1
                    f.write('{} {}\n'.format(ko,en))
        print('dictionary 파일 필터링을 진행합니다')
        print('원본 단어 수 : {} -> 필터링 후 단어 수: {}'.format(len(lines), cn))
        return ko_word_final, en_word_final

    def fasttext_vectorizing(self, ko_fasttext_pth, en_fasttext_pth, ko_output_pth, en_output_pth):
        print('fasttext emb 진행')
        fasttext_model = models.fasttext.load_facebook_model(ko_fasttext_pth)
        with open(ko_output_pth, 'wt', encoding='utf-8') as f:
            f.write('{} 300\n'.format(self.num_words))
            for i in self.ko_word_list:
                f.write('{} {}\n'.format(i,' '.join(map(str, fasttext_model.wv.word_vec(i).tolist()))))

        fasttext_model = models.fasttext.load_facebook_model(en_fasttext_pth)
        with open(en_output_pth, 'wt', encoding='utf-8') as f:
            f.write('{} 300\n'.format(self.num_words))
            for i in self.en_word_list:
                f.write('{} {}\n'.format(i,' '.join(map(str, fasttext_model.wv.word_vec(i).tolist()))))
        print('emb file 생성 완료')

def vecmap_cutoff(word_list_cutoff, vecmap_pth, output_pth, vocab_size):
    with open(output_pth, 'wt', encoding='utf-8') as fw:
        with open(vecmap_pth, 'rt', encoding='utf-8') as fr:
            vecs = fr.readlines()[1:]
            fw.write('{} 300\n'.format(vocab_size))
            for i in vecs:
                if i.split(' ',1)[0] in word_list_cutoff:
                    fw.write(i)


if __name__ == "__main__":

    args = parse_args(sys.argv)
    doing = input('1 : 문서 토큰화\n2 : 토큰화된 문서 emb\n3 : 둘다 진행\n[1/2/3] : ')
    if doing=='1' or doing=='3':
        make_ko_corpus(args.ko_origin, args.ko_token)
        make_en_corpus(args.en_origin, args.en_token)
        ready_train_test_set(args.ko_token, args.n_train, args.n_test, shuffle=True)
        ready_train_test_set(args.en_token, args.n_train, args.n_test, shuffle=True)
    if doing=='2' or doing=='3':
        corpus = Corpus(args.ko_token, args.en_token, args.vocab_size, args.vec_size)
        ko_word_final, en_word_final = corpus.treat_dictionary(args.dic)
        corpus.fasttext_vectorizing(args.ko_fasttext, args.en_fasttext, args.ko_emb, args.en_emb)

    # vecmap 진행
    doing = input('vecmap진행하겠습니다 아무키나 눌러주세요')
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    print(os.getcwd())
    os.system('python vecmap/map_embeddings.py --supervised {} {} {} {} \
    {} {}'.format(args.dic, args.emb_ko, args.emb_en, args.ko_mapped, args.en_mapped, '--cuda' if args.cuda else ''))

    #빈도수에 따라 wordvector cutoff 진행
    vecmap_cutoff(ko_word_final, args.ko_mapped, args.ko_mapped[:-4]+'_cutoff.txt', args.vocab_size)
    vecmap_cutoff(en_word_final, args.en_mapped, args.en_mapped[:-4]+'_cutoff.txt', args.vocab_size)
    
    print('vecmap 진행 완료')
    