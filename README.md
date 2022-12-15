# nd0821-c3-starter-code

The nd0821-c3-starter-code project is a deployable machine learning model leveraging a FastAPI interface. This project was created as part of the Deploying a Scalable ML Pipeline In Production Project for the Udacity Machine Learning Devops Engineer course. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) in conjunction with the requirements.txt file to setup the required environment dependencies.

```bash
pip install -r requirements.txt
```

## Usage

#### Model Training

A new model can be trained by executing the train_model script. The TrainedModel dataclass containing the trained model, encoder, label binazer and metrics is output to the `./model/` folder.

```bash
python -m ml.train_model
```

### API Deployment

The FastAPI interface can be deployed using uvicorn on a local machine using the following command.

```bash
uvicorn main:app --reload
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
