from django.urls import path
from .views import BayesView, Bayes_accuracy


urlpatterns = [
    path('disease_or_not', BayesView.as_view()),
    path('accuracy', Bayes_accuracy.as_view()),
]
