#include <Python.h>

int main(int argc, char* argv[]) {
    PyStatus status;
    PyConfig config;

    PyConfig_InitPythonConfig(&config);

    // It would be proper to error check these status values
    // To more robustly get these paths just go up from the CWD and look for "venv" in the CWD
    status = PyConfig_SetString(&config, &config.pythonpath_env, L"/home/dylenthomas/LiveASRonRPi-4/tests/python_embed_test:/home/dylenthomas/LiveASRonRPi-4/tests/python_embed_test/build/venv/lib/python3.14/site-packages");
    status = Py_InitializeFromConfig(&config);

    PyConfig_Clear(&config);
   
    PyRun_SimpleString("import sys; print('sys.path:', sys.path)");

    PyObject* pModule = PyImport_Import(PyUnicode_DecodeFSDefault("pythonTest"));
    if (pModule == NULL) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        fprintf(stderr, "ERROR: Failed to import module.\n");
        return -1;
    }

    PyObject *pFunc = PyObject_GetAttrString(pModule, "main");
    PyObject *pResult = PyObject_CallNoArgs(pFunc);
    pResult = PyObject_CallNoArgs(pFunc);

    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(pResult);
}