#include "mlp.h"
#include <iostream>

int main() {
    // Example: 2 input features, 1 output, 2 hidden layers with 4 and 3 neurons
    MLP mlp(2, 1, 2, {4, 3}, {"relu", "sigmoid"});

    std::vector<std::vector<double>> training_data = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
    std::vector<std::vector<double>> training_targets = {{0.3}, {0.7}, {0.9}};
    std::vector<std::vector<double>> val_data = {{0.2, 0.3}, {0.4, 0.5}};
    std::vector<std::vector<double>> val_targets = {{0.5}, {0.8}};

    // Train the model with validation data
    mlp.fit(training_data, training_targets, val_data, val_targets, "mean_squared_error", 0.01, 5000, 2, 50);

    // Make a prediction
    std::vector<double> input = {0.1, 0.2};
    std::vector<double> output = mlp.predict(input);
    std::cout << "Prediction: " << output[0] << std::endl;

    return 0;
}