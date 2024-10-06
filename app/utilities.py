from sklearn import metrics, pipeline, linear_model, model_selection, tree
import pandas as pd
import matplotlib.pyplot as plt

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

def eplot(x, y, yerr, weighted, title, xlabel, ylabel):
  fig, ax = plt.subplots(figsize=(12.8,7.2), dpi=100)
  ax.errorbar(x, y, yerr=yerr, fmt="o", label="Test Score")
  ax.scatter(x, weighted, marker="x", color="red", label="Weighted Score")

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.legend()
  # Display 'ticks' in x-axis and y-axis
  plt.xticks()
  plt.yticks()
  plt.title(title)
  # Show plot
  plt.show()
  return

def create_cross_scores(results):
  cross = pd.DataFrame(results[["rank_test_score"]])
  cross.loc[:, "mean_total_score"] = (results["mean_train_score"] + results["mean_test_score"]) / 2
  cross.loc[:, "std_total_score"] = (results["mean_train_score"] - results["mean_test_score"]) / 2
  cross.loc[:, "weighted_total_score"] = cross["mean_total_score"] * 0.5 + (1 - cross["std_total_score"]) * 0.5
  return cross

def make_weighted_score(results, postfix):
  return 0.5 * results["mean_" + postfix] + 0.5 * (1 - results["std_" + postfix])

