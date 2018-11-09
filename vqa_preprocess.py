import os
import sys
import time
import h5py
import json
import argparse
import itertools

import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def tokenize_ques(dataset):

    print ('Tokenizing questions ..........')
    for i, data in enumerate(tqdm(dataset)):
        ques = data['question']
        ques_tokens = word_tokenize(str(ques).lower())
        data['processed_token'] = ques_tokens

    return data


def get_num_answers(dataset, args):

    num_answers = args['num_answers']
    ans_counts = {}

    print ('Getting Top {} answers.........'.format(num_answers))
    for i, data in enumerate(tqdm(dataset)):
        ans = data['ans']
        ans_counts[ans] = ans_counts.get(ans, 0) + 1

    ans_counts = sorted(ans_counts.item(), key=lambda kv: kv[1], reverse=True)
    print ('Total number of unique answers : '.format(len(ans_counts)))
    top_answers = dict(itertools.islice(ans_counts.items(), num_answers))
    # We don't need count value of each answer
    top_ans_list = [v for k, v in top_answers.items()]
    ans2idx = {ans: idx for idx, ans in enumerate(top_ans_list)}
    idx2ans = {idx: ans for idx, ans in enumerate(top_ans_list)}

    return top_ans_list, ans2idx, idx2ans


def filter_questions(dataset, ans2idx):

    length = len(dataset)
    print ('Filtering questions based on vocabulary.........')
    for i, data in enumerate(tqdm(dataset)):
        ans = data['ans']
        if ans2idx.get(ans, 0) == 0:
            del dataset[data]  # ??????? check it

    print ('Length of dataset reduced from {} to {}'.format(length, len(dataset)))
    return dataset


def build_vocab(dataset, args):

    word_count_threshold = args['word_count_threshold']

    # Count frequency of occurrence for each word in train_data and store it in a dictionary
    word_counts = {}
    print ('Building vocabulary ................')
    for i, data in enumerate(tqdm(dataset)):
        processed_ques = data['processed_token']
        for word in processed_ques:
            word_counts[word] = word_counts.get(word, 0) + 1

    word_counts = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)
    # Filter all words based on word_count_threshold
    print ('Number of unique words in train dataset : {}'.format(len(word_counts)))
    vocab_dict = {k: v for k, v in dict(word_counts).items() if v >= word_count_threshold}
    print ('Number of unique words considering word count threshold : {}'.format(len(vocab_dict)))
    # Now we don't need the count value of words. So we will make a list instead containing
    # only words in the vocab
    vocab_list = [v for k, v in vocab_dict.items()]
    vocab_list.append('UNK')
    word2idx = {word: idx for idx, word in enumerate(vocab_list)}
    idx2word = {idx: word for idx, word in enumerate(vocab_list)}

    return word_counts, word2idx, idx2word


def encode_question(dataset, args, word2idx, word_counts):

    max_len = args['max_ques_len']
    word_count_threshold = args['word_count_threshold']
    encoded_ques = np.zeros(len(dataset), max_len)
    ques_len = np.zeros(len(dataset), dtype='uint32')

    print ('Encoding questions ............')
    for i, data in enumerate(tqdm(dataset)):
        ques = data['processed_token']
        # Below line of code will work for both train and test question. We don't need to apply
        # train vocab to test questions explicitly as Whenever a word in test question is not in
        # train_vocab its count will be zero and hence it will assigned 'UNK' token.
        ques = [word if word_counts.get(word, 0) > word_count_threshold else 'UNK' for word in ques]
        ques_len[i] = min(max_len, len(ques))  # ????????????
        for j, w in enumerate(ques):
            if j < max_len:
                encoded_ques[i][j] = word2idx[w]

    return encoded_ques, ques_len


def encode_answer(dataset, ans2idx):

    encoded_ans = np.zeros(len(dataset))  # specify dtype=int

    print ('Encoding answers ............')
    for i, data in enumerate(tqdm(dataset)):
        ans = data['ans']
        encoded_ans[i] = ans2idx[ans]

    return encoded_ans


def get_unique_img(dataset):

    all_imgs = {}  # A dictionary containing count of all image path
    print ('Getting unique images ...............')
    for data in tqdm(dataset):
        all_imgs[data['img_path']] = all_imgs.get(data['img_path'], 0) + 1

    unique_imgs = [k for k, v in all_imgs.items()]  # A list of unique image paths
    img2idx = {img: idx for idx, img in enumerate(unique_imgs)}
    img_pos = np.zeros(N)
    for i, data in enumerate(dataset):
        img_pos[i] = img2idx.get(data['img_path'])
        # img_pos[i] = unique_imgs.index(data['img_path'])

    return unique_imgs, img_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_answers', type=int, default=1000,
                        help='Total number of top answers for final classification task')

    parser.add_argument('--max_ques_len', type=int, default=26,
                        help='Maximum length of the caption, otherwise clipped')

    parser.add_argument('--word_count_threshold', type=int, default=0,
                        help='Word will be included in vocabulary only if its count exceeds threshold')

    args = parser.parse_args()

    print ('Loading json files of train & test datasets......')
    train_data = json.load(open('vqa_raw_train.json', 'r'))
    test_data = json.load(open('vqa_raw_test.json', 'r'))

    # get top 1000 answers
    top_ans_list, ans2idx, idx2ans = get_num_answers(train_data, args)

    # Filter questions answer of which is not availabel in top 1000 answers
    train_data = filter_questions(train_data, ans2idx)

    # Tokenize : Task is to tokenize all the words in question, answer, and count words
    train_data = tokenize_ques(train_data)
    test_data = tokenize_ques(test_data)

    # build vocab
    word_counts, word2idx, idx2word = build_vocab(train_data, args)

    # encode ques
    encoded_ques_train, ques_len_train = encode_question(train_data, args, word2idx, word_counts)
    encoded_ques_test, ques_len_test = encode_question(test_data, args, word2idx, word_counts)

    # encode answer
    encoded_ans = encode_answer(train_data, ans2idx)

    # get unique img
    # Since there are multiple questions for the same image. We need to have a set of unique
    # images.
    unique_img_train, img_pos_train = get_unique_img(train_data)
    unique_img_test, img_pos_test = get_unique_img(test_data)

    # encode mc ans

    with h5py.File('data_prepro.hdf5', 'w') as f:
        f.create_dataset("encoded_ques_train", dtype='uint32', data=encoded_ques_train)
        f.create_dataset("encoded_ques_test", dtype='uint32', data=encoded_ques_test)
        f.create_dataset("ques_len_train", dtype='uint32', data=ques_len_train)
        f.create_dataset("ques_len_test", dtype='uint32', data=ques_len_test)
        f.create_dataset("encoded_ans", dtype='uint32', data=encoded_ans)
        f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
        f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)

    print ('Data written to output_h5')

    out = {}
    out['idx2word'] = idx2word
    out['idx2ans'] = idx2ans
    out['unique_img_train'] = unique_img_train
    out['unique_img_test'] = unique_img_test

    json.dump(out, open('data_prepro.json'), 'w')
    print ('Data written to output_json')


if __name__ == '__main__':
    main()
