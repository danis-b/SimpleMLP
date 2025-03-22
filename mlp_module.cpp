#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mlp.h"

namespace py = pybind11;

PYBIND11_MODULE(mlp_module, m) {
    py::class_<MLP>(m, "MLP")
        .def(py::init<int, int, int, std::vector<int>, std::vector<std::string>>())
        .def("forward", &MLP::forward)
        .def("fit", &MLP::fit,
             py::arg("training_data"),
             py::arg("training_targets"),
             py::arg("val_data") = std::vector<std::vector<double>>(),
             py::arg("val_targets") = std::vector<std::vector<double>>(),
             py::arg("loss_function") = "mean_squared_error",
             py::arg("learning_rate") = 0.01,
             py::arg("epochs") = 1000,
             py::arg("batch_size") = 32,
             py::arg("print_epochs") = 100)
        .def("predict", &MLP::predict)
        .def("train", &MLP::train)
        .def("eval", &MLP::eval);
}