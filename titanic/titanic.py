#!/usr/bin/env python
import argparse
import logging as log
import pandas as pd
import re
import sys
from sklearn.ensemble import RandomForestClassifier

log.basicConfig(stream=sys.stdout, level=log.INFO,
                format='%(asctime)-15s %(threadName)s %(filename)s %(levelname)s %(message)s')


def prepare(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Cabin'].fillna('missing', inplace=True)
    df['Ticket'].fillna('missing', inplace=True)
    df['Name'].replace({'(.*)\s?,.*': r'\1'}, regex=True, inplace=True)
    df['Ticket'].replace({'(.*\s)?(\d{1,})': r'\2'}, regex=True, inplace=True)

    df['CabinDeck'] = df['Cabin'].apply(
        lambda x: re.search('(\w)(\d*).*', x).group(1))
    df['CabinNumber'] = df['Cabin'].apply(
        lambda x: re.search('(\w)(\d*).*', x).group(2))

    normalize_df = df.drop(
        ['Embarked', 'Ticket', 'Cabin', 'Sex', 'Name', 'Pclass', 'CabinDeck', 'CabinNumber'], 1)

    normalized_embarked = pd.get_dummies(
        data=df['Embarked'], prefix='Embarked')
    normalized_sex = pd.get_dummies(df['Sex'], prefix='Sex')
    normalized_class = pd.get_dummies(df['Pclass'], prefix='Pclass')
    normalized_cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    normalized_cabin_deck = pd.get_dummies(df['CabinDeck'], prefix='CabinDeck')
    normalized_cabin_number = pd.get_dummies(
        df['CabinNumber'], prefix='CabinNumber')
    normalized_ticket = pd.get_dummies(df['Ticket'], prefix='Ticket')
    normalized_name = pd.get_dummies(df['Name'], prefix='Name')

    return pd.concat([normalize_df, normalized_embarked, normalized_sex, normalized_class,
                      normalized_cabin, normalized_cabin_deck, normalized_cabin_number,
                      normalized_ticket, normalized_name], axis=1)


def load_data(train_data_path, test_data_path):
    index_columns = 'PassengerId'
    train_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex',
                     'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    test_columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age',
                    'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    train_data_df = pd.read_csv(
        train_data_path, header=0, index_col=index_columns, names=train_columns)
    test_data_df = pd.read_csv(
        test_data_path, header=0, index_col=index_columns, names=test_columns)

    return pd.concat([train_data_df, test_data_df])


def process_args():
    parser = argparse.ArgumentParser(
        description='Scikit classifier for Kaggle titanic competition')

    parser.add_argument("-trd", "--train", type=str,
                        help="Location of train data")
    parser.add_argument("-tsd", "--test", type=str,
                        help="Location of test data")

    args = parser.parse_args()
    return args


def main_wrapper():
    args = process_args()
    df = load_data(args.train, args.test)

    df = prepare(df)

    trdf = pd.DataFrame(df[df.Survived.notnull()])
    tsdf = pd.DataFrame(df[df.Survived.isnull()])

    classifier = RandomForestClassifier()
    classifier = classifier.fit(trdf.drop('Survived', 1), trdf['Survived'])

    r = classifier.predict(tsdf.drop('Survived', 1))

    print('PassengerId,Survived')
    i = 0
    for index, row in tsdf.iterrows():
        print('%s,%s' % (index, int(r[i])))
        i = i + 1


def main():
    try:
        log.info("action=main status=start")
        main_wrapper()
        log.info("action=main status=end")
    except Exception:
        log.info('action=main status=failure')
        raise

if __name__ == "__main__":
    main()
