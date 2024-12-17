# Load the saved model
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model = joblib.load(os.path.join(BASE_DIR, 'train_models', 'diabetes', 'diabetes_model.pkl'))

# def diabetes_home(request):

#     if request.method == 'POST':
#             # Extract user inputs
#         data = request.POST
#         features = np.array([
#             float(data['Pregnancies']),
#             float(data['Glucose']),
#             float(data['BloodPressure']),
#             float(data['SkinThickness']),
#             float(data['Insulin']),
#             float(data['BMI']),
#             float(data['DiabetesPedigreeFunction']),
#             float(data['Age'])
#         ]).reshape(1, -1)
            
#             # Make prediction
#         prediction = model.predict(features)[0]
#         result = "Diabetic" if prediction == 1 else "Non-Diabetic"
#         print(result)
#         return render(request, 'diabetes/home.html', {'result': result})
#     else:
#         return render(request, 'diabetes/home.html')
    


import joblib
import os
from django.shortcuts import render
from django.http import JsonResponse
import matplotlib.pyplot as plt


# Load saved data
def load_model_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(BASE_DIR, 'train_models', 'diabetes_new', 'diabetes_data.pkl'))
    return model

def save_model_accuracy_graph(model_results, static_dir, filename):
    """
    Generate a bar graph comparing model accuracies and save it as an image.
    """
    models = list(model_results.keys())
    accuracies = [result['accuracy'] for result in model_results.values()]

    plot_path = os.path.join(static_dir, filename)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')

    # Save the plot
    plt.savefig(plot_path)
    plt.close()


def model_dashboard(request):
    # Load dataset and model details
    
    data = load_model_data()

    # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # static_dir = os.path.join(BASE_DIR, 'static', 'plots')

    # save_model_accuracy_graph(data["model_results"], static_dir, 'diabetes_accuracy_plot.png')

    return render(request, 'diabetes/dashboard.html', {
        "dataset_info": data['dataset_info'],
        "model_results": data['model_results'],
        "plot_path": 'model_accuracy.jpg'
    })


def predict_diabetes(request):
    if request.method == 'POST':
        data = load_model_data()
        user_input = [float(request.POST[f'feature{i}']) for i in range(1, 9)]
        
        # Scale input
        scaler = data['scaler']
        user_input_scaled = scaler.transform([user_input])
        
        # Predict using the best model
        best_model = data['best_model']
        prediction = best_model.predict(user_input_scaled)[0]
        
        return JsonResponse({"prediction": "Diabetic" if prediction == 1 else "Non-Diabetic"})
