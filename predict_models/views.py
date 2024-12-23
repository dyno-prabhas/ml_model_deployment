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

    inputs = data['dataset_info']['columns']

    return render(request, 'diabetes/dashboard.html', {
        "inputs": inputs[:-1],
        "dataset_info": data['dataset_info'],
        "model_results": data['model_results'],
    })


def predict_diabetes(request):

    data = load_model_data()
    inputs = data['dataset_info']['columns']
    if request.method == 'POST':
        user_input = [float(request.POST[i]) for i in inputs[:-1]]
        # Scale input
        scaler = data['scaler']
        user_input_scaled = scaler.transform([user_input])
        # Predict using the best model
        best_model = data['best_model']
        prediction = best_model.predict(user_input_scaled)[0]
        output = {"prediction": "Diabetic" if prediction == 1 else "Non-Diabetic"}
        return render(request, 'diabetes/diabetes_prediction.html', {
            "output": output['prediction'],
            "inputs": inputs[:-1],
        }) 
        #return JsonResponse({"prediction": "Diabetic" if prediction == 1 else "Non-Diabetic"})
    else:
        return render(request, 'diabetes/diabetes_prediction.html', {
            "inputs": inputs[:-1],
        })
        # return render(request, 'diabetes/dashboard.html', {
        #     "output": output['prediction'],
        #     "inputs": inputs[:-1],
        #     "dataset_info": data['dataset_info'],
        #     "model_results": data['model_results'],
        #     "plot_path": 'model_accuracy.jpg'
        # }) 
