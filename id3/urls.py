from django.urls import path
from .views import ID3View, Id3_accuracy


urlpatterns = [
    path('disease_or_not', ID3View.as_view()),
    path('accuracy', Id3_accuracy.as_view()),
]
