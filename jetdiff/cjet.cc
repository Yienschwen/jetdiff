#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <ceres/jet.h>
#include <ceres/rotation.h>

namespace py = pybind11;
using jet = ceres::Jet<double, -1>;

jet jet_k(int dim, int k, double val)
{
    auto j = jet{val, Eigen::VectorXd::Zero(dim)};
    if ((k >= 0) && (k < dim))
    {
        j.v[k] = 1;
    }
    return j;
}

std::string to_string(const jet &jet)
{
    std::ostringstream oss;
    oss << "[" << jet.a << " ; ";
    auto N = jet.v.size();
    for (int i = 0; i < N; ++i)
    {
        oss << jet.v[i];
        if (i != N - 1)
        {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

using j_ = jet (*)(const jet &);
using js = jet (*)(const jet &, double);
using sj = jet (*)(double, const jet &);
using jj = jet (*)(const jet &, const jet &);

template <typename T>
T cast(T ptr) { return ptr; }

PYBIND11_MODULE(cjet, m)
{
    auto unary = &cast<j_>;

    // FIXME: release GIL to enable multithreading

    py::class_<jet>(m, "CJet")
        .def(py::init())
        .def_static("k", jet_k, py::arg("dim") = 1, py::arg("k") = -1, py::arg("val") = 0)
        .def_readwrite("f", &jet::a)
        .def_readwrite("df", &jet::v)
        // +
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self += double())
        .def(py::self + double())
        .def(double() + py::self)
        // -
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self -= double())
        .def(py::self - double())
        .def(double() - py::self)
        // *
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self *= double())
        .def(py::self * double())
        .def(double() * py::self)
        // /
        .def(py::self / py::self)
        .def(py::self /= py::self)
        .def(py::self /= double())
        .def(py::self / double())
        .def(double() / py::self)
        // self sign
        .def(+py::self)
        .def(-py::self)

        // comparison
        // <
        .def(py::self < py::self)
        .def(py::self < double())
        .def(double() < py::self)
        // <=
        .def(py::self < py::self)
        .def(py::self < double())
        .def(double() < py::self)
        // >
        .def(py::self > py::self)
        .def(py::self > double())
        .def(double() > py::self)
        // >=
        .def(py::self >= py::self)
        .def(py::self >= double())
        .def(double() >= py::self)
        // ==
        .def(py::self == py::self)
        .def(py::self == double())
        .def(double() == py::self)
        // !=
        .def(py::self != py::self)
        .def(py::self != double())
        .def(double() != py::self)

        // trig
        .def("sin", (j_)&ceres::sin)
        .def("cos", (j_)&ceres::cos)
        .def("tan", (j_)&ceres::tan)
        .def("arcsin", (j_)&ceres::asin)
        .def("arccos", (j_)&ceres::acos)
        .def("arctan", (j_)&ceres::atan)
        .def("hypot", (jj)&ceres::hypot)
        .def("arctan2", (jj)&ceres::atan2)

        // hyperbolic
        .def("sinh", (j_)&ceres::sinh)
        .def("cosh", (j_)&ceres::cosh)
        .def("tanh", (j_)&ceres::tanh)
        // TODO: arcsinh, arccosh, arctanh

        // rounding
        .def("__floor__", (j_)&ceres::floor)
        .def("__ceil__", (j_)&ceres::ceil)
        // TODO: round, trunc

        // exp, log
        .def("exp", (j_)&ceres::exp)
        .def("exp2", (j_)&ceres::exp2)
        .def("log", (j_)&ceres::log)
        .def("log2", (j_)&ceres::log2)

        // arithmetic
        .def("__pow__", (js)&ceres::pow)
        .def("__rpow__", [](const jet &j, double s)
             { return ceres::pow(s, j); })
        .def("__pow__", (jj)&ceres::pow)

        // TODO: expm1, log10, log1p, logaddexp, logaddexp2

        // misc
        .def("sqrt", (j_)&ceres::sqrt)
        .def("cbrt", (j_)&ceres::cbrt)
        .def("__abs__", (j_)&ceres::abs)
        .def("fmin", (jj)&ceres::fmin)
        .def("fmax", (jj)&ceres::fmax)

        // other python methods
        .def("__float__", [](const jet &j)
             { return j.a; })
        .def("__int__", [](const jet &j)
             { return static_cast<int>(j.a); })
        .def("__repr__", &to_string);

    // TODO: rotation
}