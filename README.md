# MBS-default-prediction
A program to take in loan level data and create a model which can predict probability of default of mortgages within Agency MBS. Data from Freddie Mac Loan Level Data

Requires:
* `gc`
* `pandas` 
* `matplotlib`
* `seaborn`
* `numpy`
* `os`
* `glob`
* `sklearn`
# Usage

* main file: MBS.py

`$ python3 MBS.py` 

* data files: data/

User Guide: [Freddie Mac LLD PDF](http://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf)

Column names
```
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
```
# Issues/Todo
* Label Encoding
* Integrate with Freddie Mac API to evaluate certain MBS pools
* Need to add dtypes dictionary
* Need to run on larger datasets (small number of defaults relative to overall loan data)
* Refine model hyperparameters/feature selection
* Add prepayment risk model
* Better metrics for model (after more data added)
