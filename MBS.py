import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter
import glob
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def get_default(r):
    if r == 'R':
        return 1
    else:
        return 0


def first_time_transform(r):
    if r == 'Y':
        return 1
    else:
        return 0


cwd = os.getcwd()

init_cols = ['Credit Score', 'First Payment Date', 'First Timebuyer',
             'Maturity Date', 'MSA', 'MI %', 'No of Units', 'Occupancy Status',
             'OCLTV', 'DTI', 'UPB', 'Loan to Value', 'Original IR', 'Channel',
             'PPM', 'Product Type', 'Property State', 'Property Type',
             'Postal Code', 'Loan Sequence', 'Loan Purpose', 'Loan Term',
             'No Borrowers', 'Seller Name', 'Servicer', 'Super conforming',
             'Pre-HARP loan seq']

time_cols = ['Loan Sequence', 'Monthly Report Per', 'Curr UPB',
             'Current Loan Del', 'Loan Age', 'Remaining Months Mat',
             'Repo Flag', 'Modi Flag', 'Zero bal', 'Zero balance eff',
             'Current IR', 'Current deferred UPB',
             'Due Date of Last Paid Installment', 'MI Recoveries',
             'Net Sale Pro', 'Non MI recoveries', 'Expenses', 'Legal Cost',
             'Maint', 'Taxes Insur', 'Misc', 'Actual Loss', 'Mod Cost',
             'Step Mod Flag', 'Def Pay Mod', 'ELTV']

init_str = 'historical_data1_Q[1]20?8.txt'
time_str = 'historical_data1_time_Q[1]20?8.txt'

init_data = list(glob.glob(str(cwd) + '/data/' + init_str))
time_data = list(glob.glob(str(cwd) + '/data/' + time_str))


def read_data(init_id, time_id):

    # Read in data from a specific quarter then output the train test data
    # init_id: file location of the initialization data
    # time_id: file location of the loan performance data

    print(f'Reading {init_id}..')
    init_big = pd.read_csv(init_id, sep='|', names=init_cols)

    print(f'Reading {time_id}..')
    time_big = pd.read_csv(time_id, sep='|', names=time_cols)

    init_big.sort_values(by='Loan Sequence')
    time_big.sort_values(by='Loan Sequence')

    delinq = time_big['Current Loan Del']
    print(delinq.unique)
    print(init_big.shape[0])

    features = ['Credit Score', 'First Payment Date', 'First Timebuyer',
                'Maturity Date', 'MSA', 'MI %', 'Original IR', 'UPB']

    init_big = init_big[features]

    init_big['TARGET'] = delinq.map(get_default)
    init_big['First Timebuyer'] = \
        init_big['First Timebuyer'].map(first_time_transform)
    init_big.fillna(0, inplace=True)

    feat = init_big.loc[:, init_big.columns != 'TARGET']
    target = init_big.TARGET

    features_train, features_test, target_train, target_test = \
        train_test_split(feat, target, test_size=0.20, random_state=10)

    return features_train, features_test, target_train, target_test


def train_model(features_train, features_test, target_train, target_test):

    # takes in data and trains the inital quarter data
    clf = SGDClassifier()
    clf.fit(features_train, target_train)
    calibrator = CalibratedClassifierCV(clf, cv='prefit')
    model = calibrator.fit(features_train, target_train)

    target_pred = clf.predict(features_test)
    predicted_probas = model.predict_proba(features_test)

    acc = accuracy_score(target_test, target_pred)

    cnf_matrix = confusion_matrix(target_test, target_pred)
    return clf, cnf_matrix, acc


clf, cnf_matrix, acc = \
    train_model(features_train, features_test, target_train, target_test)

# Delete the dataframe before getting new data and updating the clf

del init_big, time_big
del features_train, features_test, target_train, target_test
gc.collect()


def update_model(
        features_train, features_test, target_train, target_test, clf):
    # takes in new quarter data and a clf and outputs new trained metrics/clf

    clf.partial_fit(features_train, target_train)
    calibrator = CalibratedClassifierCV(clf, cv='prefit')
    model = calibrator.fit(features_train, target_train)

    target_pred = clf.predict(features_test)
    predicted_probas = model.predict_proba(features_test)
    acc = accuracy_score(target_test, target_pred)

    cnf_matrix = confusion_matrix(target_test, target_pred)
    return clf, cnf_matrix, acc


clf, cnf_matrix, acc = \
    update_model(features_train, features_test, target_train, target_test, clf)


def plot_data(cnf_matrix):
    ax = plt.subplot()
    sns.heatmap(cnf_matrix, annot=True, cmap=plt.cm.Blues,
                ax=ax, fmt='g')  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

    plt.show()
    return 0
