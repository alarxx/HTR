# recognition/models.py
from django.db import models


class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploads/')
    processed_image = models.ImageField(upload_to='processed/', null=True, blank=True)
    extracted_text = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processing_completed = models.BooleanField(default=False)

    def __str__(self):
        return f"Image uploaded at {self.uploaded_at}"

    def save_processed_results(self, text, visualization_bytes):
        """
        Save the processing results including text and visualization
        """
        from django.core.files.base import ContentFile
        import os

        # Save the processed image
        filename = os.path.splitext(os.path.basename(self.image.name))[0]
        self.processed_image.save(
            f"{filename}_processed.png",
            ContentFile(visualization_bytes),
            save=False
        )

        # Save the extracted text
        self.extracted_text = text
        self.processing_completed = True
        self.save()