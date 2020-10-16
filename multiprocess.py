from multiprocessing import Process
import multiprocessing
import time
from konlpy.tag import Kkma
import shutil
import os

def print_func(i, sentences):
    print('{} process 시작--------------------'.format(i))
    kkma = Kkma()

    temp_dir = './temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    with open('temp/{}th.txt'.format(i), 'wt', encoding='utf-8') as fw:
        for sentence in sentences:
            fw.write(' '.join(kkma.morphs(sentence.strip()))+'\n')
    print('{}번째 process 진행 완료'.format(i))
    

def batch(iterable, n):
    iterable=iter(iterable)
    while True:
        chunk=[]
        for i in range(n):
            try:
                chunk.append(next(iterable))
            except StopIteration:
                yield chunk
                return
        yield chunk

# 나누어서 처리하고 저장된 파일들을 하나로 합치고 나머지는 제거
def cleaning(result_file_name):
    temp_dir = './temp'
    temps = [temp_dir + '/' + name for name in os.listdir(temp_dir)]
    with open(result_file_name,'wt', encoding='utf-8') as fw:
        for tempfile in temps:
            with open(tempfile, 'rt', encoding='utf-8') as fr:
                fw.writelines(fr.readlines())
    shutil.rmtree(temp_dir)
    print('done')


if __name__ == "__main__":  # confirms that the code is under main function
    n_core = multiprocessing.cpu_count()
    print('컴퓨터 cpu core수 : {}'.format(n_core))
    
    with open('kor_corpus_v1(괄호제거).txt', 'rt', encoding='utf-8') as fr:
        sentences = fr.readlines()
    n_list = len(sentences)//n_core+1
    sentences_list = list(batch(sentences, n_list))
    print('전체process : {}개\n'.format(n_list))

    procs = []
    # instantiating process with arguments
    for i, sentences in enumerate(sentences_list):
        # print(name)
        proc = Process(target=print_func, args=(i+1, sentences))
        procs.append(proc)
        proc.start()

    # complete the processes

    for proc in procs:
        proc.join()

    cleaning('kor_tokend_kkma.txt')