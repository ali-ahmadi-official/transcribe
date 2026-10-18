import uuid
from django.db import models

class Transcription(models.Model):
    STATUS_CHOICES = [
        ('pending', 'در انتظار'),
        ('processing', 'در حال پردازش'),
        ('done', 'انجام شده'),
        ('failed', 'خطا'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    audio_file = models.FileField(upload_to='audio_uploads/')
    text = models.TextField(blank=True)
    language = models.CharField(max_length=5)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.id} - {self.status}"
