import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")

iris_fg = fs.get_or_create_feature_group(
    name="iris_modal",
    version=1,
    primary_key=["sepal_length","sepal_width","petal_length","petal_width"], 
    description="Iris flower dataset")
iris_fg.insert(iris_df, write_options={"wait_for_job" : False})

#expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="iris_dimensions")    
#value_between(expectation_suite, "sepal_length", 4.5, 8.0)
#value_between(expectation_suite, "sepal_width", 2.1, 4.5)
#value_between(expectation_suite, "petal_length", 1.2, 7)
#value_between(expectation_suite, "petal_width", 0.2, 2.5)
#iris_fg.save_expectation_suite(expectation_suite=expectation_suite, validation_ingestion_policy="STRICT")    
    

