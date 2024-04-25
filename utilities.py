from sklearn import metrics, pipeline, linear_model, model_selection, tree
import pandas as pd

# dataframe.summary()
# dataframe.describe()

def make_scorers():
  return {
    "accuracy": metrics.make_scorer(metrics.accuracy_score),
    "f1_score": metrics.make_scorer(metrics.f1_score),
  }

def make_pipeline():
  classifiers = [
    linear_model.LogisticRegression(), 
    tree.DecisionTreeClassifier(), 
  ]
  steps = [
    ("classifier", classifiers)
  ]
  return pipeline.Pipeline(steps)

def run_pipeline(pipe, grid, scorers):
  model = model_selection.GridSearchCV(pipe, grid, scoring=scorers, cv=5, return_train_score=True)
  return 

def f(results):
  rank = ["rank_test_score"]
  filter = ["params", "rank_test_score", "mean_test_score", "mean_train_score"]
  pd.DataFrame(results).sort_values(rank)[filter]
  return

def plot_results(results):
  rank = "rank_test_score"
  results = results.sort_values(rank).head(20)
  lmean = []
  lstd = []
  c1 = {}

  mean_test = "mean_test_score"
  std_test = "std_test_score"
  lmean += [mean_test]
  lstd += [std_test]
  c1[std_test] = mean_test

  mean_train = "mean_train_score"
  std_train = "std_train_score"
  lmean += [mean_train]
  lstd += [std_train]
  c1[std_train] = mean_train

  stds = pd.DataFrame(results[lstd])
  stds = stds.rename(columns=c1)
  results[lmean].plot.bar(yerr=stds, capsize=4, rot=0)
  return

def plot_diff(results):
  
  return


