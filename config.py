# Path to the train and test data files
train_file = "./data/train_more.csv"
test_file = "./data/test.csv"

# declare feature columns and target column
target_col = 'Survived'
feature_col = ['Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

# model type
model_type = 'gbm'

# output directory
outdir = "./output"

