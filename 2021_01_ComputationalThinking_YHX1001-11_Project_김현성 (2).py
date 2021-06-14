from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import matplotlib.pyplot as plt

import os

def get_ngrams(wordlist, n):
    ngrams = []
    ######################################################################
    #            wordlist로부터 n-gram(들)을 ngrams에 입력하기           #
    ######################################################################
    for i in range(len(wordlist)-(n-1)):
        for k in range(i,i+n):
            ngrams[i] = ngrams[i] + wordlist[k]
    ######################################################################
    #                            코드 작성 끝                            #
    ######################################################################
    return ngrams

def get_score_from_ngrams(ngrams1, ngrams2):
    count = 0
    ######################################################################
    #         ngrams2의 원소 개수가 ngrams1의 원소 개수보다 많으면,      #
    #                     ngrams1과 ngrams2를 교환하기                   #
    ######################################################################
    if len(ngrams1) < len(ngrams2):
        temp = ngrams1
        ngrams1 = ngrams2
        ngrams2 = temp
    ######################################################################
    #                            코드 작성 끝                            #
    ######################################################################

    ######################################################################
    #          ngrams1의 각 원소별로 ngrams2의 모든 원소와 비교하여      #
    #                      같으면, count 1 증가하기                      #
    ######################################################################
    for i in range(0,len(ngrams1)):
        for k in range(0,len(ngrams2)):
            if ngrams1[i] == ngrams2[k]:
                count = count + 1
    ######################################################################
    #                            코드 작성 끝                            #
    ######################################################################
    return count / len(ngrams1)

def get_score_matrix_from_ngrmas(ngrams_list):
    score_matrix = []
    temp_score_list = []

    for i, ngrams1 in enumerate(ngrams_list):
        for j, ngrams2 in enumerate(ngrams_list):
            ######################################################################
            #            i와 j가 같지 않으면, ngrams1과 ngrams2에 대한           #
            #        get_score_from_ngrams() 결과를 temp_score_list에 입력하고,  #
            #      그렇지 않으면(i와 j가 같으면), 1.를 temp_score_list에 입력하기#
            ######################################################################
            if i != j :
                temp_score_list = get_score_from_ngrams(ngrams1,ngrams2)
            elif i == j:
                temp_score_list = 1.
            ######################################################################
            #                            코드 작성 끝                            #
            ######################################################################
        score_matrix.append(temp_score_list)
        temp_score_list = []

    return score_matrix

def get_wordlist_from_file(filename):
    wordlist = []
    ########################################################################
    #  filename 파일의 모든 내용을 공백 단위로 분리하여 wordlist에 입력하기#
    ########################################################################
    templist = []
    file = open(filename, 'r')
    lines = file.readlines()
    for i in range(0, len(lines)):
        templist = lines[i].split()
        wordlist = wordlist + templist
    ######################################################################
    #                            코드 작성 끝                            #
    ######################################################################
    return wordlist

def get_content_from_file(filename):
    content = ""

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            content = content + line[:-1]

    return content

def get_filelist_from_directory():
    filelist = []
    ######################################################################
    #            이 파일이 있는 현재 위치의 txt_dir 폴더에 있는          #
    #            .txt 파일(들)의 파일명을 filelist에 입력하기            #
    ######################################################################
    directory = 'C://Users/proto/OneDrive/바탕 화면/txt_dir/*.txt'
    filelist = os.listdir(directory)
    ######################################################################
    #                            코드 작성 끝                            #
    ######################################################################
    return filelist

def main():
    n = None
    wordlists = []
    ngrams_list = []
    ######################################################################
    #                        정수 n 입력 받기(n: )                       #
    ######################################################################
    n = int(input("정수 n을 입력하시오:"))
    ######################################################################
    #                            코드 작성 끝                            #
    ######################################################################

    filelist = get_filelist_from_directory()
    num_of_files = len(filelist)

    ######################################################################
    #     filelist를 이용하여 각 파일에 대한 get_word_list_from_file()   #
    #                    결과를 wordlists에 입력하기                     #
    ######################################################################
    for i in range(0, num_of_files):
        wordlists = wordlists + get_word_list_from_file(filelist[i])
    ######################################################################
    #                            코드 작성 끝                            #
    ######################################################################

    ######################################################################
    #          wordlists를 이용하여 각 원소에 대한 get_ngrams()          #
    #                    결과를 ngrams_list에 입력하기                   #
    ######################################################################
    for i in range(0,len(wordlists)):
        ngrams_list[i] = get_ngrams(wordlists[i],n)
    ######################################################################
    #                            코드 작성 끝                            #
    ######################################################################

    score_matrix = get_score_matrix_from_ngrmas(ngrams_list)

    if (len(score_matrix) == 0):
        return

    for row in score_matrix:
        print(row)

    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(np.array(score_matrix), interpolation='none')
    fig.colorbar(im)

    ax.set_xticks(np.arange(num_of_files))
    ax.set_xticklabels(filelist)
    ax.set_yticks(np.arange(num_of_files))
    ax.set_yticklabels(filelist)

    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
           ha="left", va="center", rotation_mode="anchor")

    fig.tight_layout()
    plt.show()

    contents = [get_content_from_file(os.path.join("./txt_dir", filename)) for filename in filelist]

    ngram_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\S+', ngram_range=(n, n))
    counts = ngram_vectorizer.fit_transform(contents)
    score_matrix = cosine_similarity(counts)

    if (len(score_matrix) == 0):
        return

    for row in score_matrix:
        print(row)

    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(np.array(score_matrix), interpolation='none')
    fig.colorbar(im)

    ax.set_xticks(np.arange(num_of_files))
    ax.set_xticklabels(filelist)
    ax.set_yticks(np.arange(num_of_files))
    ax.set_yticklabels(filelist)

    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
            ha="left", va="center", rotation_mode="anchor")

    fig.tight_layout()
    plt.show()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(contents)

    score_matrix = cosine_similarity(X)

    if (len(score_matrix) == 0):
        return

    for row in score_matrix:
        print(row)

    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(np.array(score_matrix), interpolation='none')
    fig.colorbar(im)

    ax.set_xticks(np.arange(num_of_files))
    ax.set_xticklabels(filelist)
    ax.set_yticks(np.arange(num_of_files))
    ax.set_yticklabels(filelist)

    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
            ha="left", va="center", rotation_mode="anchor")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

############################################################    
##C언어 수강신청 못했는데 열심히 했으니 선처 부탁드립니다.##
##모듈화와 예외처리도 잘 모르겠습니다...............      ##
############################################################   
