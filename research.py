import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sheet_url = 'https://docs.google.com/spreadsheets/d/1OMurqWKO_E6IJHzX9HfvMssccJWpCspnA_B0KgLguPM/edit#gid=17834267'
csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
diabetes = pd.read_csv(csv_export_url)

# diabetes.head()

#############
zero_columns = [col for col in diabetes.columns if (diabetes[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
for col in zero_columns:
    diabetes[col] = np.where(diabetes[col] == 0, np.nan, diabetes[col])

for col in zero_columns:
    diabetes.loc[diabetes[col].isnull(),col] = diabetes[col].mean()

#############
# diabetes["Outcome_name"]= diabetes["Outcome"].apply(lambda x: "Diabetes" if x==1 else "Not-Diabetes")
# diabetes.head()
#############


X = diabetes.drop(["Outcome"], axis=1)
Y = diabetes.Outcome

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(X)
prediction_proba = clf.predict_proba(X)

# Saving the model
import pickle
pickle.dump(clf,open('diabetes_simple.pkl', "wb"))

def plot_importance(model, features, num=len(X), save=False):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(clf, X,save=True)