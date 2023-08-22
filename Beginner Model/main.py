import pandas as pd

teams = pd.read_csv("teams.csv")

teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

teams.corr()["medals"]
