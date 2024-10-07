# import the required libraries
import pickle
from ML_Pipeline.utils import read_data,inspection,null_values
from ML_Pipeline.ml_model import prepare_model_smote,run_model
from ML_Pipeline.evaluate_metrics import confusion_matrix,roc_curve
from ML_Pipeline.feature_imp import plot_feature_importances
from ML_Pipeline.plot_model import plot_model
import matplotlib.pyplot as plt

# Read the initial datasets
df = read_data("../input/data_regression.csv")

# Inspection and cleaning the data
x = inspection(df)

# Drop the null values
df = null_values(df)

### Run the decision tree model with sklearn ###

##Selecting only the numerical columns and excluding the columns we specified in the function
X_train, X_test, y_train, y_test = prepare_model_smote(df,class_col='churn',
                                                 cols_to_exclude=['customer_id','phone_no', 'year']) 

# run the model
model_dectree,y_pred = run_model(X_train,X_test,y_train,y_test) # train the model

## performance metric ##
conf_matrix = confusion_matrix(y_test,y_pred) # generate confusion matrix
#print(conf_matrix)
roc_val = roc_curve(model_dectree,X_test,y_test) # plot the roc curve

#plot the tree
decision_tree_plot = plot_model(model_dectree,['not churn','churn'])
plt.savefig("../output/"+"Decision_Tree_plot.png")

#feature importance
fea_imp = plot_feature_importances(model_dectree)
plt.savefig("../output/"+"Feature_Importance.png") # plot the feature importance graph

## Save the model ##
pickle.dump(model_dectree, open('../output/model.pkl', 'wb'))
