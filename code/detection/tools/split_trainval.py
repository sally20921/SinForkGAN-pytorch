import argparse
import os
import random


def split_datasets(xml_path, train_sample):
    xml_list = os.listdir(xml_path)
    xml_total = len(xml_list)
    train_total = int(xml_total * train_sample)
    train_sample = random.sample(range(xml_total), train_total)

    train = open("Main/train.txt", "w")
    val = open("Main/valid.txt", "w")

    for i in range(xml_total):
        name = xml_list[i][:-4] + "\n"
        if i in train_sample:
            train.write(name)
        else:
            val.write(name)

    train.close()
    val.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script tool for dividing training set and verification set in dataset.")
    parser.add_argument('--xml-path', type=str, default="./Annotations", help="Location of dimension files in dataset.")
    parser.add_argument('--train-size', type=float, default=0.8, help='Size of training set in data set')
    args = parser.parse_args()

    try:
        os.makedirs("Main")
    except OSError:
        pass

    split_datasets(args.xml_path, args.train_size)
