from sklearn.model_selection import GridSearchCV
hyper_tune = GridSearchCV (clf, {
    'C':[1,10,20,100],
    'kernel': ['rbf', 'linear', 'poly']
}, cv=5, return_train_score=False)
hyper_tune.fit(my_vectors, my_labels)
hyper_tune.cv_results_

import pandas as pd
df = pd.DataFrame(hyper_tune.cv_results_)
df[['param_C', 'param_kernel','mean_test_score']]
