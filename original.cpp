#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>

class MLP {
private:
    size_t in_size;              // Size of input layer
    size_t out_size;             // Size of output layer
    int num_layers;              // Number of layers (hidden + output)
    std::vector<int> layer_sizes; // Sizes of hidden layers
    std::vector<std::vector<double>> weights;       // Weight matrices
    std::vector<std::vector<double>> biases;        // Bias vectors
    std::vector<std::vector<double>> activations;   // Post-activation values
    std::vector<std::vector<double>> pre_activations; // Pre-activation values
    bool is_training;            // Flag to indicate training or evaluation mode

    // Activation function type
    enum class ActivationFunction {
        SIGMOID,
        RELU,
        TANH,
        LINEAR
    };

    std::vector<ActivationFunction> layer_activation_funcs;

    // Activation function with layer-specific handling
    double activate(double x, int layer) {
        switch (layer_activation_funcs[layer]) {
            case ActivationFunction::SIGMOID:
                return 1.0 / (1.0 + std::exp(-x));
            case ActivationFunction::RELU:
                return std::max(0.0, x);
            case ActivationFunction::TANH:
                return std::tanh(x);
            case ActivationFunction::LINEAR:
                return x;
            default:
                return x; // Fallback to linear
        }
    }

    // Derivative of activation function
    double activate_derivative(double activated_value, double pre_activation_value, int layer) {
        switch (layer_activation_funcs[layer]) {
            case ActivationFunction::SIGMOID:
                return activated_value * (1.0 - activated_value);
            case ActivationFunction::RELU:
                return pre_activation_value > 0 ? 1.0 : 0.0;
            case ActivationFunction::TANH:
                return 1.0 - activated_value * activated_value;
            case ActivationFunction::LINEAR:
                return 1.0;
            default:
                return 1.0; // Fallback to linear
        }
    }

    // Loss function type
    enum class LossFunction {
        MEAN_SQUARED_ERROR,
        MEAN_ABSOLUTE_ERROR,
        CROSS_ENTROPY
    };

    LossFunction loss_func = LossFunction::MEAN_SQUARED_ERROR;

    // Compute loss
    double loss(const std::vector<double>& output, const std::vector<double>& target) {
        double error = 0.0;
        switch (loss_func) {
            case LossFunction::MEAN_SQUARED_ERROR:
                for (size_t i = 0; i < output.size(); i++) {
                    double diff = target[i] - output[i];
                    error += diff * diff;
                }
                return error / output.size();
            case LossFunction::MEAN_ABSOLUTE_ERROR:
                for (size_t i = 0; i < output.size(); i++) {
                    error += std::abs(target[i] - output[i]);
                }
                return error / output.size();
            case LossFunction::CROSS_ENTROPY:
                for (size_t i = 0; i < output.size(); i++) {
                    error += target[i] * std::log(output[i] + 1e-10); // Avoid log(0)
                }
                return -error / output.size();
            default:
                return 0.0;
        }
    }

    // Compute loss derivative
    std::vector<double> loss_derivative(const std::vector<double>& output, const std::vector<double>& target) {
        std::vector<double> dloss(output.size());
        switch (loss_func) {
            case LossFunction::MEAN_SQUARED_ERROR:
                for (size_t i = 0; i < output.size(); i++) {
                    dloss[i] = output[i] - target[i];
                }
                break;
            case LossFunction::MEAN_ABSOLUTE_ERROR:
                for (size_t i = 0; i < output.size(); i++) {
                    dloss[i] = (output[i] > target[i]) ? 1.0 : -1.0;
                }
                break;
            case LossFunction::CROSS_ENTROPY:
                for (size_t i = 0; i < output.size(); i++) {
                    dloss[i] = -target[i] / (output[i] + 1e-10); // Avoid division by zero
                }
                break;
            default:
                std::fill(dloss.begin(), dloss.end(), 0.0);
        }
        return dloss;
    }

    // Initialize parameters using Xavier/He methods
    void initialize_parameters() {
        std::random_device rd;
        std::mt19937 gen(rd());

        weights.resize(num_layers);
        biases.resize(num_layers);
        activations.resize(num_layers + 1);
        pre_activations.resize(num_layers + 1);

        activations[0].resize(in_size); // Input layer

        for (int i = 0; i < num_layers; i++) {
            int prev_size = (i == 0) ? in_size : layer_sizes[i - 1];
            int curr_size = (i == num_layers - 1) ? out_size : layer_sizes[i];

            double fan_in = static_cast<double>(prev_size);
            double fan_out = static_cast<double>(curr_size);
            double std;

            if (layer_activation_funcs[i] == ActivationFunction::RELU) {
                std = std::sqrt(2.0 / fan_in); // He initialization
            } else {
                std = std::sqrt(2.0 / (fan_in + fan_out)); // Xavier initialization
            }

            std::normal_distribution<> dist(0.0, std);

            weights[i].resize(prev_size * curr_size);
            biases[i].resize(curr_size);
            activations[i + 1].resize(curr_size);
            pre_activations[i + 1].resize(curr_size);

            for (auto& w : weights[i]) w = dist(gen);
            for (auto& b : biases[i]) b = dist(gen);
        }
    }

    // Helper function to compute average loss over a dataset
    double compute_average_loss(const std::vector<std::vector<double>>& data, 
                               const std::vector<std::vector<double>>& targets) {
        if (data.size() != targets.size()) {
            throw std::invalid_argument("Data and targets size mismatch in compute_average_loss");
        }
        double total_loss = 0.0;
        for (size_t i = 0; i < data.size(); i++) {
            std::vector<double> output = forward(data[i]);
            total_loss += loss(output, targets[i]);
        }
        return total_loss / data.size();
    }

public:
    // Constructor
    MLP(int in_size, int out_size, int num_layers, std::vector<int> layer_sizes,
        std::vector<std::string> activation_funcs) : is_training(false) {
        if (num_layers <= 0 || layer_sizes.size() != static_cast<size_t>(num_layers) ||
            activation_funcs.size() != static_cast<size_t>(num_layers)) {
            throw std::invalid_argument("Invalid layer or activation function configuration");
        }

        this->in_size = in_size;
        this->out_size = out_size;
        this->num_layers = num_layers;
        this->layer_sizes = layer_sizes;

        layer_activation_funcs.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            if (activation_funcs[i] == "sigmoid") {
                layer_activation_funcs[i] = ActivationFunction::SIGMOID;
            } else if (activation_funcs[i] == "relu") {
                layer_activation_funcs[i] = ActivationFunction::RELU;
            } else if (activation_funcs[i] == "tanh") {
                layer_activation_funcs[i] = ActivationFunction::TANH;
            } else if (activation_funcs[i] == "linear") {
                layer_activation_funcs[i] = ActivationFunction::LINEAR;
            } else {
                throw std::invalid_argument("Unknown activation function: " + activation_funcs[i]);
            }
        }

        initialize_parameters();
    }

    // Set the model to training mode
    void train() {
        is_training = true;
    }

    // Set the model to evaluation mode
    void eval() {
        is_training = false;
    }

    // Forward pass
    std::vector<double> forward(const std::vector<double>& input) {
        if (input.size() != in_size) {
            throw std::invalid_argument("Input size mismatch");
        }

        activations[0] = input;

        for (int layer = 0; layer < num_layers; layer++) {
            int prev_size = (layer == 0) ? in_size : layer_sizes[layer - 1];
            int curr_size = (layer == num_layers - 1) ? out_size : layer_sizes[layer];

            for (int j = 0; j < curr_size; j++) {
                double sum = biases[layer][j];
                for (int i = 0; i < prev_size; i++) {
                    sum += activations[layer][i] * weights[layer][i * curr_size + j];
                }
                // Store pre-activations only in training mode for backpropagation
                if (is_training) {
                    pre_activations[layer + 1][j] = sum;
                }
                activations[layer + 1][j] = activate(sum, layer);
            }
        }

        return activations[num_layers];
    }

    // Training function with validation data support
    void fit(const std::vector<std::vector<double>>& training_data,
             const std::vector<std::vector<double>>& training_targets,
             const std::vector<std::vector<double>>& val_data = {},
             const std::vector<std::vector<double>>& val_targets = {},
             std::string loss_function = "mean_squared_error",
             double learning_rate = 0.01,
             int epochs = 1000,
             int batch_size = 32,
             int print_epochs = 100) {
        // Set to training mode
        train();

        // Set loss function
        if (loss_function == "mean_squared_error") {
            loss_func = LossFunction::MEAN_SQUARED_ERROR;
        } else if (loss_function == "mean_absolute_error") {
            loss_func = LossFunction::MEAN_ABSOLUTE_ERROR;
        } else if (loss_function == "cross_entropy") {
            loss_func = LossFunction::CROSS_ENTROPY;
        } else {
            throw std::invalid_argument("Unknown loss function: " + loss_function);
        }

        // Validate input sizes
        if (training_data.size() != training_targets.size()) {
            throw std::invalid_argument("Training data and targets size mismatch");
        }
        if (!val_data.empty() && val_data.size() != val_targets.size()) {
            throw std::invalid_argument("Validation data and targets size mismatch");
        }

        size_t num_samples = training_data.size();
        size_t num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division

        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_error = 0.0;

            // Shuffle data indices
            std::vector<size_t> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            // Process each mini-batch
            for (size_t batch = 0; batch < num_batches; batch++) {
                size_t start = batch * batch_size;
                size_t end = std::min(start + batch_size, num_samples);

                // Initialize gradient accumulators
                std::vector<std::vector<double>> weight_gradients(num_layers);
                std::vector<std::vector<double>> bias_gradients(num_layers);
                for (int layer = 0; layer < num_layers; layer++) {
                    int prev_size = (layer == 0) ? in_size : layer_sizes[layer - 1];
                    int curr_size = (layer == num_layers - 1) ? out_size : layer_sizes[layer];
                    weight_gradients[layer].resize(prev_size * curr_size, 0.0);
                    bias_gradients[layer].resize(curr_size, 0.0);
                }

                // Process each sample in the mini-batch
                for (size_t s = start; s < end; s++) {
                    size_t sample = indices[s];

                    if (training_data[sample].size() != in_size || training_targets[sample].size() != out_size) {
                        throw std::invalid_argument("Sample " + std::to_string(sample) + " size mismatch");
                    }

                    // Forward pass (pre_activations stored because is_training is true)
                    std::vector<double> output = forward(training_data[sample]);

                    // Compute error
                    total_error += loss(output, training_targets[sample]);

                    // Backward pass
                    auto dloss_val = loss_derivative(output, training_targets[sample]);
                    std::vector<std::vector<double>> layer_deltas(num_layers);
                    for (int i = 0; i < num_layers; i++) {
                        int layer_size = (i == num_layers - 1) ? out_size : layer_sizes[i];
                        layer_deltas[i].resize(layer_size, 0.0);
                    }
                    for (size_t i = 0; i < out_size; i++) {
                        layer_deltas[num_layers - 1][i] = dloss_val[i] *
                            activate_derivative(activations[num_layers][i], pre_activations[num_layers][i], num_layers - 1);
                    }
                    for (int layer = num_layers - 2; layer >= 0; layer--) {
                        int curr_size = layer_sizes[layer];
                        int next_size = (layer == num_layers - 2) ? out_size : layer_sizes[layer + 1];
                        for (int i = 0; i < curr_size; i++) {
                            double error = 0.0;
                            for (int j = 0; j < next_size; j++) {
                                error += layer_deltas[layer + 1][j] * weights[layer + 1][i * next_size + j];
                            }
                            layer_deltas[layer][i] = error *
                                activate_derivative(activations[layer + 1][i], pre_activations[layer + 1][i], layer);
                        }
                    }

                    // Accumulate gradients
                    for (int layer = 0; layer < num_layers; layer++) {
                        int prev_size = (layer == 0) ? in_size : layer_sizes[layer - 1];
                        int curr_size = (layer == num_layers - 1) ? out_size : layer_sizes[layer];
                        for (int j = 0; j < curr_size; j++) {
                            for (int i = 0; i < prev_size; i++) {
                                weight_gradients[layer][i * curr_size + j] += 
                                    layer_deltas[layer][j] * activations[layer][i];
                            }
                            bias_gradients[layer][j] += layer_deltas[layer][j];
                        }
                    }
                }

                // Average gradients over the batch
                size_t batch_actual_size = end - start;
                for (int layer = 0; layer < num_layers; layer++) {
                    int prev_size = (layer == 0) ? in_size : layer_sizes[layer - 1];
                    int curr_size = (layer == num_layers - 1) ? out_size : layer_sizes[layer];
                    for (int j = 0; j < curr_size; j++) {
                        for (int i = 0; i < prev_size; i++) {
                            weight_gradients[layer][i * curr_size + j] /= batch_actual_size;
                        }
                        bias_gradients[layer][j] /= batch_actual_size;
                    }
                }

                // Update weights and biases
                for (int layer = 0; layer < num_layers; layer++) {
                    int prev_size = (layer == 0) ? in_size : layer_sizes[layer - 1];
                    int curr_size = (layer == num_layers - 1) ? out_size : layer_sizes[layer];
                    for (int j = 0; j < curr_size; j++) {
                        for (int i = 0; i < prev_size; i++) {
                            weights[layer][i * curr_size + j] -= 
                                learning_rate * weight_gradients[layer][i * curr_size + j];
                        }
                        biases[layer][j] -= learning_rate * bias_gradients[layer][j];
                    }
                }
            }

            // Print progress every print_epochs
            if (epoch % print_epochs == 0) {
                double train_loss = total_error / num_samples;
                std::cout << "Epoch " << epoch << ", Training Loss: " << train_loss;
                if (!val_data.empty()) {
                    // Temporarily switch to eval mode for validation
                    eval();
                    double val_loss = compute_average_loss(val_data, val_targets);
                    std::cout << ", Validation Loss: " << val_loss;
                    // Switch back to training mode
                    train();
                }
                std::cout << std::endl;
            }
        }
    }

    // Prediction function
    std::vector<double> predict(const std::vector<double>& input) {
        eval(); // Ensure evaluation mode
        return forward(input);
    }
};

// Example usage
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