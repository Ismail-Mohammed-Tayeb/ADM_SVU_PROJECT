from rest_framework.views import APIView
from rest_framework.response import Response
from .helper_functions import prepare_data
from .id3 import *
from pprint import pprint
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import random


# Create your views here.
class ID3View(APIView):
    @swagger_auto_schema(
        operation_description="اختبار خوارزمية ID3 في توقع اصابة الشخص بالقلب حسب المعطيات التالية",
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
                'age': openapi.Schema(type=openapi.TYPE_NUMBER, description="عمر المريض ",),
                'max_heart_rate': openapi.Schema(type=openapi.TYPE_NUMBER, description="نبضات القلب العظمى"),
                'exercice_angina': openapi.Schema(type=openapi.TYPE_STRING, description="الألم عند التمرين { yes or no}"),
                'rest_electro': openapi.Schema(type=openapi.TYPE_STRING, description="نتيجة تخطيط القلب{'left_vent_hyper','st_t_wave_abnormality','normal'} "),
                'chest_pain_type': openapi.Schema(type=openapi.TYPE_STRING, description="نوع الم الصدر{'asympt','atyp_angina','non_anginal','typ_angina'}"),
                'rest_blood_pressure': openapi.Schema(type=openapi.TYPE_NUMBER, description="ضغط الدم الانبساطي  "),
                'blood_sugar': openapi.Schema(type=openapi.TYPE_BOOLEAN, description="سكر الدم { True or False} "),
            },
        )
        )
    def post(self, request):
        # example of data
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

        data = {
                    "age": [age],
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
        print(web_df)
        web_df = prepare_data(web_df, train_set=False)
        web_df.drop(["Disease"], axis=1, inplace=True)



        # load data
        df = pd.read_csv("./static/data/heart_disease_male.csv")
        df = prepare_data(df)
        df["label"] = df.Disease
        df = df.drop("Disease", axis=1)

        random.seed(0)
        train_df, test_df = train_test_split(df, 0.2)
        train_df = train_df.squeeze()
        # test_df = test_df.squeeze()
        tree = decision_tree_algorithm(train_df, ml_task="classfication")
        # pprint(tree)
        # accuracy = calculate_accuracy(test_df, tree)
        # print(accuracy)
        prediction = make_predictions(web_df, tree)
        print(prediction)
        if prediction[0] == 0:
            result = "not Disease"
        elif prediction[0] == 1:
            result = "Disease"
        else:
            result = "unknown"

        print(result)

        response = Response()
        response.data = {
            "prediction_expected ": result,
        }
        return response

class Id3_accuracy(APIView):
    def get(self, response):
        # load data
        df = pd.read_csv("./static/data/heart_disease_male.csv")
        df = prepare_data(df)
        df["label"] = df.Disease
        df = df.drop("Disease", axis=1)

        random.seed(0)
        train_df, test_df = train_test_split(df, 0.2)
        train_df = train_df.squeeze()
        test_df = test_df.squeeze()
        tree = decision_tree_algorithm(train_df, ml_task="classfication")
        # pprint(tree)
        accuracy = calculate_accuracy(test_df, tree)

        response = Response()
        response.data = {
            "id3_accuracy ": accuracy,
        }
        return response
