import re
import xml.etree.ElementTree as ET
import numpy as np


class Preprocessor:
    def __init__(self):
        self.text_list = []
        self.vocab = {}

    def filter_punctuation(self, line):
        line = re.sub(r"[+\.\!\/_,$%^*(+\"\']+|[+—、~@#《》￥%&*（），。]+", '', line)
        return line

    def creat_vocab(self):
        vocab = {'OOV': 0}
        word_num = 1
        for line in self.text_list:
            line = line.strip().split()
            for word in line:
                if word not in vocab:
                    vocab[word] = word_num
                    word_num += 1
                else:
                    continue
        self.vocab = vocab

    def word2id(self, text_list):
        self.text_list = text_list
        self.creat_vocab()
        id_list = []
        for line in text_list:
            line = line.strip().split()
            for i, word in enumerate(line):
                id = self.vocab[word]
                line[i] = id
            id_list.append(line)
        print('vocab length: %d' % len(self.vocab))
        return id_list


class DataLoader:
    def __init__(self, batch_size=2):
        self.file = 'ExpressionTest_cut.xml'
        self.vocab = {'None': 0, 'happiness': 1, 'like': 2, 'surprise': 3, 'fear': 4, 'sadness': 5, 'anger': 6, 'disgust': 7}
        self.batch_size = batch_size
        self.text_list, self.label_list = self.read_xml()
        self.data = zip(self.text_list, self.label_list)
        self.ratio = [0.8, 0.19, 0.01]  # 训练集、验证集、测试集比例
        self.train_data, self.val_data, self.test_data = self.split_data()
        self.train_batch = self.get_batch_data(self.train_data, self.batch_size)
        self.val_batch = self.get_batch_data(self.val_data, self.batch_size)
        self.test_batch = self.get_batch_data(self.test_data, self.batch_size)

        self.num_word = 0

    def read_xml(self):
        label_list = []
        text_list = []
        tree = ET.parse(self.file)
        root = tree.getroot()
        preprocessor = Preprocessor()
        for i in range(len(root)):
            for child in root[i]:
                line = preprocessor.filter_punctuation(child.text)
                text_list.append(line)
                if child.attrib['opinionated'] == 'Y':
                    label_list.append([1, self.vocab[child.attrib['emotion-1-type']]])
                else:
                    label_list.append([0, 0])
        id_list = preprocessor.word2id(text_list)
        self.num_word = len(preprocessor.vocab)
        return id_list, label_list

    def split_data(self):
        dataset = []
        for sample in self.data:
            dataset.append(sample)
        data_len = len(dataset)
        train_len = int(np.floor(data_len * self.ratio[0]))
        val_len = int(np.floor(data_len * self.ratio[1]))
        test_len = data_len - train_len - val_len
        return dataset[:train_len], dataset[train_len: -test_len], dataset[-test_len:]

    def get_batch_data(self, data, batch_size):
        batches = []
        for i in range(0, len(data), batch_size):
            batches.append(data[i:i + batch_size])
        return batches


if __name__ == '__main__':

    data = DataLoader(2).train_batch


    one, tot, max_lenth = 0, 0, 0
    for batch in data:
        for x in batch:
            max_lenth = max(max_lenth, len(x[0]))
            one += x[1][0]
            tot += 1

    print('have emotion: %d, total: %d' % (one, tot))
    print('max length: %d' % max_lenth)












