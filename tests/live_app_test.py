"""
Module for testing the live Heroku API application.
"""
import requests


def live_app_test(base_url):
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

    url = base_url + '/infer'

    response = requests.post(url, json=data_model, timeout=1.5)

    status_code = response.status_code
    response_value = response.json()['output']

    return {'STATUS_CODE': status_code, 'OUTPUT': response_value}


def main():
    """
    Main module for executing live test
    """
    base_url = 'https://census-class-app.herokuapp.com'
    response_dict = live_app_test(base_url)
    print(response_dict)


if __name__ == "__main__":
    main()
