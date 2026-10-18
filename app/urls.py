from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/upload/', views.upload_audio, name='upload'),
    path('api/status/<uuid:job_id>/', views.status, name='status'),
    path('result/<uuid:job_id>/', views.result, name='result'),
    path('api/result/<uuid:job_id>/save/', views.save_result, name='save_result'),
    path('api/result/<uuid:job_id>/discard/', views.discard_result, name='discard_result'),
    path('transcription/<uuid:pk>/', views.detail, name='detail'),
    path('api/transcription/<uuid:pk>/update/', views.update_saved, name='update_saved'),
]
