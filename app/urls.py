from django.contrib.auth import views as auth_views
from django.urls import path
from . import views

urlpatterns = [
    # ---- احراز هویت (تمپلیت: templates/registration/login.html) ----
    path('login/', auth_views.LoginView.as_view(
        template_name='registration/login.html',
        redirect_authenticated_user=True), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),  # فقط POST

    # ---- صفحات اصلی ----
    path('', views.index, name='index'),
    path('api/upload/', views.upload_audio, name='upload'),
    path('api/status/<uuid:job_id>/', views.status, name='status'),
    path('result/<uuid:job_id>/', views.result, name='result'),
    path('api/result/<uuid:job_id>/save/', views.save_result, name='save_result'),
    path('api/result/<uuid:job_id>/discard/', views.discard_result, name='discard_result'),
    path('report/<uuid:job_id>/', views.start_report, name='start_report'),

    # ---- موارد ذخیره‌شده ----
    path('transcription/<uuid:pk>/', views.detail, name='detail'),
    path('api/transcription/<uuid:pk>/update/', views.update_saved, name='update_saved'),
    path('api/transcription/<uuid:pk>/share/', views.share_saved, name='share_saved'),  # جدید
    path('audio/<uuid:pk>/', views.audio, name='audio'),                                  # جدید

    # ---- مدیریت پزشکان (فقط ادمین) ----
    path('manage/doctors/', views.doctors_admin, name='doctors_admin'),
    path('api/doctors/create/', views.doctor_create, name='doctor_create'),
    path('api/doctors/<int:pk>/toggle/', views.doctor_toggle, name='doctor_toggle'),
    path('api/doctors/<int:pk>/password/', views.doctor_password, name='doctor_password'),
]