# recognition/views.py
from django.shortcuts import render
from django.http import JsonResponse
from .forms import ImageUploadForm
from .models import ImageUpload
from .services import TextRecognitionService
from django.core.files.base import ContentFile


def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            image_upload = form.save()

            try:
                # Process the image using our ML service
                service = TextRecognitionService()
                results = service.process_image(image_upload.image.path)

                # Save the processing results
                image_upload.save_processed_results(
                    text=results['text'],
                    visualization_bytes=results['visualization']
                )

                return JsonResponse({
                    'status': 'success',
                    'text': results['text'],
                    'processed_image_url': image_upload.processed_image.url,
                    'original_image_url': image_upload.image.url
                })
            except Exception as e:
                return JsonResponse({
                    'status': 'error',
                    'error': str(e)
                })

        return JsonResponse({
            'status': 'error',
            'errors': form.errors
        })

    form = ImageUploadForm()
    return render(request, 'recognition/index.html', {'form': form})