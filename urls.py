from django.urls import path
from .views import hello_world, hello_api, upload_video, matched_frame

urlpatterns = [
    path('api/hello', hello_world),
    path('api/test', hello_api),
    path('api/upload', upload_video),
    path('api/matchframe', matched_frame),

]