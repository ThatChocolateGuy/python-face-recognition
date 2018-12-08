from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^clear/$',
        views.clear_database,
        name='clear_database'),
    url(r'^train/$',
        views.train_model,
        name='train_model'),
    url(r'^predict/$',
        views.predict,
        name='predict'),
    url(r'^basic-upload/$',
        views.BasicUploadView.as_view(),
        name='basic_upload'),
    url(r'^train-model/$',
        views.ProgressBarUploadView.as_view(),
        name='progress_bar_upload'),
    url(r'^drag-and-drop-upload/$',
        views.DragAndDropUploadView.as_view(),
        name='drag_and_drop_upload'),
]