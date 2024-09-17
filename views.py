from django.shortcuts import render
from .forms import ImageUploadForm # type: ignore
from .predict import predict_food # type: ignore

def home(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            image = request.FILES['image']
            image_path = f'media/{image.name}'

            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Call the prediction model
            predictions = predict_food(image_path)
            return render(request, 'index.html', {'form': form, 'predictions': predictions})

    else:
        form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})
