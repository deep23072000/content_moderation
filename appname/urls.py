from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import home, video_list, upload_video, preprocess_video  # ✅ Add preprocess_video

urlpatterns = [
    path('', home, name='home'),
    path('videos/', video_list, name='video_list'),
    path('upload/', upload_video, name='upload_video'),
    path('preprocess/<int:video_id>/', preprocess_video, name='preprocess_video'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
