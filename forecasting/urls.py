from django.urls import path

from . import views


urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('api/forecast/summary/', views.forecast_summary_view, name='forecast-summary'),
    path('api/forecast/scenario/', views.forecast_scenario_view, name='forecast-scenario'),
    path('api/forecast/booking-predict/', views.booking_prediction_view, name='booking-predict'),
]
