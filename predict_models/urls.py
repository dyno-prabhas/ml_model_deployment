from django.urls import path
from . import views

urlpatterns = [
    
    path('dashboard/', views.model_dashboard, name='dashboard'),
    path('predict/', views.predict_diabetes, name='predict_diabetes'),
]