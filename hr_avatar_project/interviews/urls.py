from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


# urlpatterns = [
#     path('', views.home_page),
# ]

# interviews/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('interview/', views.home_page, name='interview'),
    # path('interview/<str:candidate_id>/', views.interview_page, name='interview_with_id'),
]