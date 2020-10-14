# preprocess_undreamt_ko

undreamt github를 참고하여 한국어-영어 간의 비지도 번역을 실습하여 보았다.

사용된 데이터셋은 각기다른 corpus로 한국어는 AI hub의 자료, 영어는 WMT14의 new데이터를 이용하였다.

비지도 학습을 통한 번역은 먼저 각 언어별 corpus를 통해 embedding을 한다. 여기서 embedding은 timesave를 위하여 fasttext를 이용하였다. 

그후, 언어간 비슷한 의미를 가진 단어들을 mapping 시켜주는 과정이 필요하다. 이 과정에는 [vecmap](https://github.com/artetxem/vecmap)을 clone하여 이용하였다.

즉 preprocessing.py 의 기능은 원본 문서로 부터 [vecmap](https://github.com/artetxem/vecmap)

을 진행하기 위한 자료들을 생성해주는 역할이 있다.

이후 비지도학습을 이용한 번역은 [undreamt](https://github.com/Changyoon-Lee/unsupervised_nmt)을 참고하길 바란다.

### requirement

- python 3

- konlpy
- gensim
- pandas
- nltk
- fasttext(cc.ko.bin, cc.en.bin)

- clone vecmap

- 한글-영어 단어 set
- vecmap에필요한 기본 자료들(numpy, scipy, cupy)

### usage

```
python preprocess.py
```

