import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("iris_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("jim-hopsworks-ai"))
   def f():
       g()


def generate_flower(name, sepal_len_max, sepal_len_min, sepal_width_max, sepal_width_min, 
                    petal_len_max, petal_len_min, petal_width_max, petal_width_min):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "sepal_length": [random.uniform(sepal_len_max, sepal_len_min)],
                       "sepal_width": [random.uniform(sepal_width_max, sepal_width_min)],
                       "petal_length": [random.uniform(petal_len_max, petal_len_min)],
                       "petal_width": [random.uniform(petal_width_max, petal_width_min)]
                      })
    df['variety'] = name
    return df


def get_random_iris_flower():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    virginica_df = generate_flower("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
    versicolor_df = generate_flower("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
    setosa_df =  generate_flower("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,3)
    if pick_random >= 2:
        iris_df = virginica_df
        print("Virginica added")
    elif pick_random >= 1:
        iris_df = versicolor_df
        print("Versicolor added")
    else:
        iris_df = setosa_df
        print("Setosa added")

    return iris_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    iris_df = get_random_iris_flower()

    iris_fg = fs.get_feature_group(name="iris_modal",version=1)
    iris_fg.insert(iris_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("iris_daily")
        with stub.run():
            f()
