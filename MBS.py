import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter
import glob
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
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
# Add Dtypes
init_dtypes = {}
time_dtypes = {}


def get_glob(quarter, year):

    init_str = f'historical_data1_Q[{quarter}]201{year}.txt'
    time_str = f'historical_data1_time_Q[{quarter}]201{year}.txt'

    init_glob = list(glob.glob(str(cwd) + '/data/' + init_str))
    time_glob = list(glob.glob(str(cwd) + '/data/' + time_str))

    return init_glob, time_glob


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

    features = ['Credit Score', 'First Payment Date', 'First Timebuyer',
                'Maturity Date', 'MSA', 'MI %', 'Original IR', 'UPB']

    init_big = init_big[features]

    init_big['TARGET'] = delinq.map(get_default)
    init_big['First Timebuyer'] = \
        init_big['First Timebuyer'].map(first_time_transform)
    init_big.fillna(0, inplace=True)

    feat = init_big.loc[:, init_big.columns != 'TARGET'].astype(np.int64)
    target = init_big.TARGET.astype(np.int64)
    total_dels = np.sum(target)

    features_train, features_test, target_train, target_test = \
        train_test_split(feat, target, test_size=0.1, random_state=10)

    return total_dels, features_train, features_test, target_train, target_test


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


def plot_data(cnf_matrix, quarter, year):
    ax = plt.subplot()
    sns.heatmap(cnf_matrix, annot=True, cmap=plt.cm.Blues,
                ax=ax, fmt='g')  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix for Q{quarter}-201{year}')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

    plt.show()
    return 0


def run_full_model(quarter_list, year_list):
    total_del_count = []
    first = True
    for quarter in quarter_list:
        for year in year_list:
            init_glob, time_glob = get_glob(quarter, year)
            for init_id, time_id in zip(init_glob, time_glob):
                if first:
                    print('Getting data...')
                    total_dels, features_train, features_test, target_train, target_test = read_data(
                        init_id, time_id)
                    total_del_count.append(total_dels)
                    print('Training Model...')
                    try:
                        clf, cnf_matrix, acc = \
                            train_model(features_train, features_test,
                                        target_train, target_test)
                        print('Plotting data')
                        plot_data(cnf_matrix, quarter, year)
                    except ValueError as e:
                        print(e)
                    first = False
                else:
                    # Delete the dataframe before getting new data
                    print('Deleting last...')
                    del features_train, features_test
                    del target_train, target_test
                    gc.collect()
                    # Read in new data
                    print('Reading new data...')
                    total_dels, features_train, features_test, target_train, target_test = read_data(
                        init_id, time_id)
                    total_del_count.append(total_dels)
                    # Update the model
                    try:
                        print('Updating model...')
                        clf, cnf_matrix, acc = \
                            update_model(features_train, features_test,
                                         target_train, target_test, clf)
                        print('Plotting data')
                        plot_data(cnf_matrix, quarter, year)
                    except ValueError as e:
                        print(e)
                    # Plot confusion Matrix for new model
    print(np.sum(total_del_count))
    return clf


quarter_list = [1, 2, 3, 4]
year_list = [6, 7, 8]

clf = run_full_model(quarter_list, year_list)
