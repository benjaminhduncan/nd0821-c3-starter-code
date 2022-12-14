"""
Module for testing the API end points
"""


def test_root_get(test_client_fixture):
    """
    Function to test the root get method
    """
    response = test_client_fixture.get('/')
    assert response.status_code == 200
    assert response.json() == {
        "msg": "Welcome to the Salary Classifier API!"}


def test_infer_post_less(test_client_fixture):
    """
    Function to test the inference end point
    """
    data_model = {
        "age": 19,
        "workclass": "Never-worked",
        "fnlwgt": 77516,
        "education": "HS-grad",
        "education-num": 13,
        "marital-status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    response = test_client_fixture.post('/infer', json=data_model)
    response_value = response.json()['output']
    assert response.status_code == 200
    assert "<=50K" in response_value


def test_infer_post_more(test_client_fixture):
    """
    Function to test the inference end point
    """
    data_model = {
        "age": 50,
        "workclass": "Private",
        "fnlwgt": 77516,
        "education": "Masters",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    response = test_client_fixture.post('/infer', json=data_model)
    response_value = response.json()['output']
    assert response.status_code == 200
    assert ">50K" in response_value
