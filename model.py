import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model():
    data = pd.read_csv("data.csv")

    x = data[["hours_studied", "attendance", "previous_score"]]
    y = data["final_score"]

    model = LinearRegression()
    model.fit(x,y)

    return model 
