#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>

class MLP {
public:
    // Constructor
    MLP(int in_size, int out_size, int num_layers, std::vector<int> layer_sizes, std::vector<std::string> activation_funcs);

    // Forward pass
    std::vector<double> forward(const std::vector<double>& input);

    // Training method
    void fit(const std::vector<std::vector<double>>& training_data,
             const std::vector<std::vector<double>>& training_targets,
             const std::vector<std::vector<double>>& val_data = {},
             const std::vector<std::vector<double>>& val_targets = {},
             std::string loss_function = "mean_squared_error",
             double learning_rate = 0.01,
             int epochs = 1000,
             int batch_size = 32,
             int print_epochs = 100);

    // Prediction
    std::vector<double> predict(const std::vector<double>& input);

    // Mode setters
    void train();
    void eval();

private:
    size_t in_size;
    size_t out_size;
    int num_layers;
    std::vector<int> layer_sizes;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> pre_activations;
    bool is_training;

    enum class ActivationFunction {
        SIGMOID, RELU, TANH, LINEAR
    };
    std::vector<ActivationFunction> layer_activation_funcs;

    enum class LossFunction {
        MEAN_SQUARED_ERROR, MEAN_ABSOLUTE_ERROR, CROSS_ENTROPY
    };
    LossFunction loss_func;

    double activate(double x, int layer);
    double activate_derivative(double activated_value, double pre_activation_value, int layer);
    double loss(const std::vector<double>& output, const std::vector<double>& target);
    std::vector<double> loss_derivative(const std::vector<double>& output, const std::vector<double>& target);
    void initialize_parameters();
    double compute_average_loss(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& targets);
};

#endif // MLP_H