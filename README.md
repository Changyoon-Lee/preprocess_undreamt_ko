# preprocess_undreamt_ko

undreamt github를 참고하여 한국어-영어 간의 비지도 번역을 실습하여 보았다.

사용된 데이터셋은 각기다른 corpus로 한국어는 AI hub의 자료, 영어는 WMT14의 new데이터를 이용하였다.

비지도 학습을 통한 번역은 먼저 각 언어별 corpus를 통해 embedding을 한다. 여기서 embedding은 timesave를 위하여 fasttext를 이용하였다. 

그후, 언어간 비슷한 의미를 가진 단어들을 mapping 시켜주는 과정이 필요하다. 이 과정에는 [vecmap](https://github.com/artetxem/vecmap)을 clone하여 이용하였다.

즉 preprocessing.py 의 기능은 원본 문서로 부터 [vecmap](https://github.com/artetxem/vecmap)을 진행하기 위한 자료들을 생성해주고, vecmap까지 진행하여 결과값을 생성한다.

이후 비지도학습을 이용한 번역은 [undreamt](https://github.com/Changyoon-Lee/unsupervised_nmt)을 참고하길 바란다.



## requirement

- python 3

- konlpy

- gensim

- pandas

- nltk

- fasttext(cc.ko.bin, cc.en.bin)

- 한글-영어 단어 set(한글단어와 영단어가 공백으로 구분되고, 한줄에 단어set 하나씩 들어가있으면 됨)

  

- clone vecmap

- vecmap에필요한 기본 자료들(numpy, scipy, cupy)



## usage

데이터셋을 동일하게 이용하고 올바른 경로에 위치 시켰다면 아래의 command로 실행시킬 수 있다. 

```
python preprocess.py
```



#### detail option

| tag          | 내용                                                         |
| ------------ | ------------------------------------------------------------ |
| --vec_size   | embedding할 word의 수                                        |
| --vocab_size | vecmap으로 부터 나온 자료 들 중 빈도가 높은 순으로 갯수를 필터링한다<br />갯수가 너무 많으면 비지도학습 진행시 memorry 에러가 발생 |
| --n_train    | 비지도번역에 쓰일 train set으로 몇개의 문장을 준비 할 것인지 |
| --n_test     | 비지도번역에 쓰일 test set으로 몇개의 문장을 준비 할 것인지  |
| --ko_token   | 토큰화된 문서 경로                                           |
| --en_token   | 토큰화된 문서 경로                                           |
| --cuda       | vecmap실행시 cuda사용                                        |

따로 토큰화하여 준비 된 데이터를 사용하려면 다음과 같이 작성하면 된다

ex)

```
python preprocess.py --ko_token 파일경로 --en_token 파일경로 --cuda
```

실행시  [1/2/3]중 옵션을 고를 수 있도록 하였는데 따로 준비한 데이터 셋을 사용하고자 한다면 2 를 입력하면 된다.



# 비고

영어 wmt14 data 1000만 문장 이용

한국어 AIhub data 80만 문장 이용

(주요한 단어들을 선별하는데에 문장의 갯수가 매우 부족한 것 같다.)

vecmap을 위해 20만 단어의 emb값을 사용하였는데 이렇게 한 이유는 vecmap github를 참고하였다 