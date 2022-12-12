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


def test_infer_post(test_client_fixture):
    """
    Function to test the inference end point
    """
    data_model = {
        "age": 28,
        "workclass": "Never-worked",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Divorced",
        "occupation": "Tech-support",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = test_client_fixture.post('/infer', json=data_model)
    response_value = response.json()['output']
    assert response.status_code == 200
    assert "50K" in response_value
