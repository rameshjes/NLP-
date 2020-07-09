
# ML-FLOW

It is an open-source platform used for managing end-to-end ML lifecycle. It helps users in handling the following tasks:
* Keep track of the experiments (e.g. maintains track of classifier's params., datasets used for training, etc)
* Managing and deployment of ML models
* Packaging ML code (code easily be shared with other members in team)


## Install Dependencies

```pip install -r requirements``` 

  Note: `conda.yaml` can also be used, but I mostly work with virtualenvs, so I don't configure that. 

## Example

   To run toy example, implemented using sklearn library. Execute following command inside `sklearn-example` directory: 

   ```mlflow run . -P run_name=linear_regression --no-conda```

   To view results inside browser execute: ```mlflow ui``` 

   This will publish results on your localhost: ```localhost:5000```
