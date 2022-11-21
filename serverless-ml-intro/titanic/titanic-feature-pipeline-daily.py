import os
import modal

LOCAL = False

if LOCAL == False:
    stub = modal.Stub("Titanic Daily")
    image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("ID2223"))
    def f():
        print(os.environ["HOPSWORKS_API_KEY"])
        g()

def generate_passenger(survived, age_min, age_max, fare_min, fare_max, sex, pclass):
    """
    Returns single Titanic passenger as single row in a DataFrame
    """
    import pandas as pd
    import random
    import numpy as np 

    df = pd.DataFrame({"age": [random.uniform(age_min, age_max)],
                       "fare": [random.uniform(fare_min, fare_max)],
                       "sex": [sex],
                       "embarked": [np.round(random.uniform(0, 5))],
                       "parch": [np.round(random.uniform(0, 6))],
                       "class": [pclass],
                       "sibsp": np.round(random.uniform(0, 5))
                       
                       })
    df['survived'] = survived

    return df

def get_random_titanic_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    import pandas as pd
    import random

    survived_df = generate_passenger(
        1, 0, 20, 40, 500, 1, 2)
    died_df = generate_passenger(
        0, 30, 40, 0, 20, 0, 0)

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0, 3)
    if pick_random >= 2:
        passenger_df = survived_df
        print("survived added")
    else:
        passenger_df = died_df
        print("not-survived added")

    return passenger_df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_df = get_random_titanic_passenger()

    iris_fg = fs.get_feature_group(name="titanic_surv_modal", version=1)
    iris_fg.insert(titanic_df, write_options={"wait_for_job": False})


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("titanic_daily")
        with stub.run():
            f()
