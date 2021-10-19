import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing

column_names = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11','F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20']

def get_na_count(dataset):
  boolean_mask = dataset.isna()
  return boolean_mask.sum(axis=0)


def train_test_split(x, y):
  TRAIN_SPLIT = int(len(x)*0.80)
  train_x = x[:TRAIN_SPLIT]
  train_y = y[:TRAIN_SPLIT]
  test_x = x[TRAIN_SPLIT:]
  test_y = y[TRAIN_SPLIT:]
  return train_x, train_y, test_x, test_y
  
def standardizzazione(data, mm, std):
  data = data - mm
  data = data/std
  return data


def media_val_dataset(dataset):
  for nome in column_names:
    F_mean_train = dataset[nome].mean()
    dataset[nome]=dataset[nome].fillna(F_mean_train)
  return dataset

def evaluate_classifier(classifier, test_x, test_y):
    pred_y = classifier.predict(test_x)
    confusion_matrix = metrics.confusion_matrix(test_y, pred_y)
    print(confusion_matrix)
    f1_score = metrics.f1_score(test_y, pred_y, average='macro')
    acc_score = metrics.accuracy_score(test_y, pred_y)
    print('F1: ', f1_score)
    print('Accuracy: ', acc_score)

def k_fold_cross_validation_svm(scaled_train_x, k=5, C=1, kernel='linear', degree=3, gamma='auto'):
    avg_score = 0
    cv = model_selection.KFold(n_splits=k, random_state=0)
    classifier = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    for train_index, test_index in cv.split(scaled_train_x):
        fold_train_x, fold_test_x = x[train_index], x[test_index]
        fold_train_y, fold_test_y = y[train_index], y[test_index]
        classifier.fit(fold_train_x, fold_train_y)
        fold_pred_y = classifier.predict(fold_test_x)
        score = metrics.accuracy_score(fold_test_y, fold_pred_y)
        print(score)
        avg_score += score
    avg_score = avg_score / k
    return avg_score

  
def preprocessing(dataset, test_set, split):
  print ('Analisi iniziale dataset')
  counts = dataset['CLASS'].value_counts()
  print('.........')
  print('...Dataset......')
  n_false = counts[0]
  n_true = counts[1]
  print('.........')

  print('Frazione di esempi negativi', round(n_false / (n_false + n_true), 4))
  print('Frazione di esempi positivi', round(n_true / (n_false + n_true), 4))
  print('.........')
  print('.........')
  if (test_set is not None):
    print('..Test_set...')
    counts2 = test_set['CLASS'].value_counts()
    n_false2 = counts2[0]
    n_true2 = counts2[1]
    print('Frazione di esempi negativi', round(n_false2 / (n_false2 + n_true2), 4))
    print('Frazione di esempi positivi', round(n_true2 / (n_false2 + n_true2), 4))
  print('.........')
  print('.........')
  print(dataset.groupby('CLASS').size())
  if (test_set is not None):
    print('.........')
    print(test_set.groupby('CLASS').size())
  print('.........')
  print('.........')
  print('Inizio Preprocessing')
  print('.........')
  print('.........')
  #controllo duplicati e valori NAN
  dups = dataset.duplicated()
  # report if there are any duplicates
  d=dups.any()
  if(d!=True):
    print('Non sono presenti duplicati')
  else:
    print('Sono presenti duplicati')
    print(dataset[dups])
    dataset.drop_duplicates (keep = 'first', inplace = True)
  if (test_set is not None):
    dups2 = test_set.duplicated()
    d2=dups2.any()
    if(d2!=True):
      print('Non sono presenti duplicati nel test set')
    else:
      print('Sono presenti duplicati nel test set')
      print(test_set[dups2])
      test_set.drop_duplicates (keep = 'first', inplace = True)
  summary_x = get_na_count(dataset)
  print('.........')
  print('.........')
  print('Feature campi_NAN')
  print(summary_x)
  # Gestione campi vuoti
  dataset= media_val_dataset(dataset)
  summary_x = get_na_count(dataset)
  print('.........')
  print('.........')
  print('Rimozione evntuali campi NAN')
  print(summary_x)
  if (test_set is not None):
    print('.........')
    print('.........')
    print('Feature campi_NAN     Test_set')
    summary_y = get_na_count(test_set)
    print(summary_y)
    test_set= media_val_dataset(test_set)
    print('.........')
    print('.........')
    print('Rimozione evntuali campi NAN')
    summary_y = get_na_count(test_set)
    print(summary_y)
  print('.........')
  print('.........')
  x = dataset.iloc[:,0:20].values
  y = dataset.iloc[:,20].values
  if test_set is not None:
    x_t= test_set.iloc[: ,0:20].values
    y_t= test_set.iloc[:,20].values
  if(split==True):
    train_x, train_y, test_x, test_y = train_test_split(x, y)
    mean_train= np.mean(train_x)
    std_train = np.std(train_x)
    train_x_stand= standardizzazione(train_x, mean_train, std_train)
    test_x_stand= standardizzazione(test_x, mean_train, std_train)
  else: 
    mean_train= np.mean(x)
    std_train = np.std(x)
    train_x_stand= standardizzazione(x, mean_train, std_train)
    test_x_stand= standardizzazione(x_t, mean_train, std_train)
    train_y= y
    test_y= y_t
  print('Standardizzazione effettuata')
  print('.........')
  print('.........')
  print('Fine Preprocessing')
  return train_x_stand, test_x_stand, train_y, test_y












