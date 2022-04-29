from django.shortcuts import render
import pandas as pd
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from .helper_functions import replace_strings, prepare_data, prepare_data_lable
from .bayes import *
from pprint import pprint
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi


# Create your views here.
class BayesView(APIView):
    @swagger_auto_schema(
        operation_description="اختبار خوارزمية bayes في توقع اصابة الشخص بالقلب حسب المعطيات التالية",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['age',
                      'max_heart_rate',
                      'exercice_angina',
                      'rest_electro',
                      'chest_pain_type',
                      'rest_blood_pressure',
                      'blood_sugar'],
            properties={
                'age': openapi.Schema(type=openapi.TYPE_NUMBER,description="عمر المريض ",),
                'max_heart_rate': openapi.Schema(type=openapi.TYPE_NUMBER,description="نبضات القلب العظمى"),
                'exercice_angina': openapi.Schema(type=openapi.TYPE_STRING,description="الألم عند التمرين { yes or no}"),
                'rest_electro': openapi.Schema(type=openapi.TYPE_STRING,description="نتيجة تخطيط القلب{'left_vent_hyper','st_t_wave_abnormality','normal'} "),
                'chest_pain_type': openapi.Schema(type=openapi.TYPE_STRING,description="نوع الم الصدر{'asympt','atyp_angina','non_anginal','typ_angina'}"),
                'rest_blood_pressure': openapi.Schema(type=openapi.TYPE_NUMBER,description="ضغط الدم الانبساطي  "),
                'blood_sugar': openapi.Schema(type=openapi.TYPE_BOOLEAN,description="سكر الدم { True or False} "),
            },
        )
        )
    def post(self, request):
        #example of data
        '''
        {"age": 44,
         "max_heart_rate": 144,
         "exercice_angina": "no",
         "rest_electro": "normal",
         "chest_pain_type": "asympt",
         "rest_blood_pressure": 130,
         "blood_sugar": "False"
         }
         '''
        # intialise data of lists.
        age = int(request.data['age'])
        max_heart_rate = int(request.data['max_heart_rate'])
        exercice_angina = request.data['exercice_angina']
        rest_electro = request.data['rest_electro']
        chest_pain_type = request.data['chest_pain_type']
        rest_blood_pressure = int(request.data['rest_blood_pressure'])
        blood_sugar = request.data['blood_sugar']

        data = {"age": [age],
                "max_heart_rate": [max_heart_rate],
                "exercice_angina": [exercice_angina],
                "rest_electro": [rest_electro],
                "chest_pain_type": [chest_pain_type],
                "rest_blood_pressure": [rest_blood_pressure],
                "blood_sugar": [blood_sugar],
                "disease": [""]
                }

        # Create DataFrame
        web_df = pd.DataFrame(data)
        web_df = prepare_data(web_df, train_set=False)
        web_df.drop(["Disease"], axis=1, inplace=True)
        web_df = replace_strings(web_df)

        # load data
        df_train = pd.read_csv("./static/data/heart_disease_male.csv")

        # prepare data
        df_train = prepare_data(df_train)
        df_train = replace_strings(df_train)
        df_train = df_train.squeeze()

        # handle missing values in training data
        Rest_Electro_mode = df_train.Rest_Electro.mode()[0]
        df_train["Rest_Electro"].fillna(Rest_Electro_mode, inplace=True)

        lookup_table = create_table(df_train, label_column="Disease")

        predictions = web_df.apply(predict_example, axis=1, args=(lookup_table,))

        prediction = predictions[0][0]
        disease_probability = predictions[0][1]
        not_disease_probability = predictions[0][2]

        if prediction == 0:
            prediction = "not disease"
        if prediction == 1:
            prediction = "disease"

        response = Response()
        response.data = {
            "prediction_expected": prediction,
            "disease_probability": disease_probability,
            "not_disease_probability": not_disease_probability
        }
        return response

class Bayes_accuracy(APIView):
    def get(self, response):
        # load data
        df_train = pd.read_csv("./static/data/heart_disease_male.csv")
        df_test = pd.read_csv("./static/data/heart_disease_male_test.csv")
        test_labels = pd.read_csv("./static/data/heart_disease_male_test_lable.csv")

        # prepare data
        df_train = prepare_data(df_train)
        df_test = prepare_data(df_test, train_set=False)
        df_train = replace_strings(df_train)
        df_train = df_train.squeeze()
        df_test = replace_strings(df_test)
        test_labels = prepare_data_lable(test_labels)
        test_labels = test_labels.squeeze(axis=None)

        lookup_table = create_table(df_train, label_column="Disease")
        predictions = df_test.apply(predict_example, axis=1, args=(lookup_table,))
        # print(df_test)
        # print(predictions)
        predictions_list = pd.Series([])
        for i in range(len(predictions)):
            predictions_list[i] = predictions[i][0]
        # print(predictions_list)
        predictions_correct = predictions_list == test_labels
        accuracy = predictions_correct.mean()
        # print(f"Accuracy: {accuracy:.3f}")

        response = Response()
        response.data = {
            "bayes_accuracy":accuracy
        }
        return response