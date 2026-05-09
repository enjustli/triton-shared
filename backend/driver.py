# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT license.

import hashlib
import tempfile

import os
import subprocess
import platform
import importlib.util
import sys
import re

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import (
    DriverBase,
    decompose_descriptor,
    expand_signature,
    wrap_handle_tensordesc_impl,
    _is_descriptor as _is_tensordesc,
    _parse_descriptor as _parse_tensordesc,
)
from triton.backends.compiler import GPUTarget


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)

def _get_sanitizer_type():
    # returns "" if not set
    # throws error if set to something other than "asan" or "tsan"
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")

    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        # throw error
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")
    
    return sanitizer_type

def _sanitizer_available(sanitizer_type):
    if "LD_PRELOAD" not in os.environ:
        return False
    if f"libclang_rt.{sanitizer_type}.so" not in os.environ["LD_PRELOAD"]:
        return False
    
    return True

# -------------------- Launcher ----------------------------
def _tensordesc_base_ty(dtype, ndim):
    return f"tensordesc_base<{dtype},{ndim}>"

def _is_tensordesc_base(ty):
    return isinstance(ty, str) and ty.startswith("tensordesc_base<")

def _parse_tensordesc_base(ty):
    match = re.match(r"tensordesc_base<([^,]+),(\d+)>", ty)
    if not match:
        raise AssertionError(f"Malformed tensor descriptor base type: {ty}")
    return match.group(1), int(match.group(2))

def _ty_to_cpp(ty):
    if isinstance(ty, str) and ty.startswith("tensordesc_base<"):
        return "void*"
    if ty[0] == '*':
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        # Proper support for bfloat16 and float16 is not yet handled.
        # https://github.com/microsoft/triton-shared/issues/348
        # "fp16": "TODO",
        # "bf16": "TODO",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]

def _flatten_signature(sig, output):
    # Flatten tuples
    if isinstance(sig, tuple):
        for x in sig:
            _flatten_signature(x, output)
    else:
        output.append(sig)

def _extracted_type(ty):
    if isinstance(ty, tuple):
        val = ','.join(map(_extracted_type, ty))
        return f"[{val}]"
    if ty[0] == '*':
        return "PyObject*"
    if _is_tensordesc_base(ty):
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)

def _format_of(ty):
    if isinstance(ty, tuple):
        val = ''.join(map(_format_of, ty))
        return f"({val})"
    if ty[0] == '*':
        return "O"
    if _is_tensordesc_base(ty):
        return "O"
    if ty == "constexpr":
        return "O"
    if ty.startswith("tensordesc"):
        return "O"
    return {
      "PyObject*": "O",
      "constexpr": "O",
      "float": "f",
      "double": "d",
      "long": "l",
      "int8_t": "b",
      "int16_t": "h",
      "int32_t": "i",
      "int64_t": "l",
      "uint8_t": "B",
      "uint16_t": "H",
      "uint32_t": "I",
      "uint64_t": "K",
    }[_ty_to_cpp(ty)]

def _is_pointer_like(ty):
    return isinstance(ty, str) and (ty[0] == "*" or _is_tensordesc_base(ty))

def _mark_tensordesc_bases(sig, expanded):
    if _is_tensordesc(sig):
        dtype, ndim = _parse_tensordesc(sig)
        expanded = list(expanded)
        expanded[0] = _tensordesc_base_ty(dtype, ndim)
        return tuple(expanded)

    if isinstance(sig, tuple):
        result = []
        cursor = 0
        for elem in sig:
            if _is_tensordesc(elem):
                width = len(expand_signature((elem,), None, "tensordesc"))
                result.extend(_mark_tensordesc_bases(elem, expanded[cursor:cursor + width]))
                cursor += width
            elif isinstance(elem, tuple):
                result.append(_mark_tensordesc_bases(elem, expanded[cursor]))
                cursor += 1
            else:
                result.append(expanded[cursor])
                cursor += 1
        return tuple(result)

    return expanded

def _expand_signature_dict(signature, constants):
    expanded_signature = {}
    expanded_constants = {}
    next_idx = 0
    expanded_values = expand_signature(signature.values(), None, "tensordesc")
    cursor = 0

    def expand_top_level(sig):
        if _is_tensordesc(sig):
            width = len(expand_signature((sig,), None, "tensordesc"))
            expanded = _mark_tensordesc_bases(sig, expanded_values[cursor:cursor + width])
            return list(expanded), width
        if isinstance(sig, tuple):
            return [_mark_tensordesc_bases(sig, expanded_values[cursor])], 1
        return [expanded_values[cursor]], 1

    for idx, sig in signature.items():
        expanded, width = expand_top_level(sig)
        cursor += width
        for ty in expanded:
            expanded_signature[next_idx] = ty
            if idx in constants:
                expanded_constants[next_idx] = constants[idx]
            next_idx += 1
    return expanded_signature, expanded_constants

def _kernel_arg_decls(signature):
    decls = []
    items = list(signature.items())
    i = 0
    while i < len(items):
        _, ty = items[i]
        if ty == "constexpr":
            i += 1
            continue
        if _is_tensordesc_base(ty):
            _, ndim = _parse_tensordesc_base(ty)
            decls.extend(["int64_t", "void*"])
            i += 1
            continue
        if ty[0] == "*":
            decls.extend(["int64_t", "void*"])
        else:
            decls.append(_ty_to_cpp(ty))
        i += 1
    return ', '.join(decls)

def _kernel_parameters(signature):
    params = []
    items = list(signature.items())
    i = 0
    while i < len(items):
        idx, ty = items[i]
        if ty == "constexpr":
            i += 1
            continue
        if _is_tensordesc_base(ty):
            params.extend(["0", f"&ptr_arg{idx}"])
            i += 1
            continue
        if ty[0] == "*":
            params.append(f"0, &ptr_arg{idx}")
        else:
            params.append(f"static_cast<{_ty_to_cpp(ty)}>(arg{idx})")
        i += 1
    return ', '.join(params)

def _pointer_arg_decls(signature, constants):
    decls = []
    items = list(signature.items())
    i = 0
    while i < len(items):
        idx, ty = items[i]
        if idx in constants:
            i += 1
            continue
        if _is_tensordesc_base(ty):
            dtype, ndim = _parse_tensordesc_base(ty)
            shape_idxs = [items[i + 1 + dim][0] for dim in range(ndim)]
            stride_idxs = [items[i + 1 + ndim + dim][0] for dim in range(ndim)]
            shape_values = ', '.join(f"arg{shape_idx}" for shape_idx in shape_idxs)
            stride_values = ', '.join(f"arg{stride_idx}" for stride_idx in stride_idxs)
            cpp_ty = _ty_to_cpp(dtype)
            decls.append(
                f"StridedMemRefType<{cpp_ty}, {ndim}> ptr_arg{idx} = "
                f"{{static_cast<{cpp_ty} *>(arg{idx}), static_cast<{cpp_ty} *>(arg{idx}), 0, "
                f"{{{shape_values}}}, {{{stride_values}}}}};")
            i += 1
            continue
        if _is_pointer_like(ty):
            decls.append(
                f"StridedMemRefType<char, 0> ptr_arg{idx} = "
                f"{{static_cast<char *>(arg{idx}), static_cast<char *>(arg{idx}), 0}};")
        i += 1
    return ' '.join(decls)

def make_tensordesc_arg(arg, metadata, _):
    if metadata is None:
        return decompose_descriptor(arg)
    raise NotImplementedError("CPU tensor descriptor metadata is not supported")

def _generate_launcher(constants, signature, kernel_name):
    args_format = ''.join([_format_of(ty) for ty in signature.values()])
    format = "iiiOOOO" + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    kernel_arg_decls = _kernel_arg_decls(signature)
    kernel_arg_decls += ', ' if kernel_arg_decls else ''

    kernel_parameters = _kernel_parameters(signature)
    kernel_parameters += ', ' if kernel_parameters else ''
    pointer_arg_decls = _pointer_arg_decls(signature, constants)

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

extern "C" {{
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void {kernel_name}({kernel_arg_decls}
                       int, int, int, int, int, int);
}}

static void _launch(int gridX, int gridY, int gridZ, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    // Cast "function" to the real function type.
    // apply parallelization to the triton grid when using ThreadSanitizer (TSan) 
    // to help detect potential data races across program instances during kernel execution
    {"#pragma omp parallel for collapse(3)" if _get_sanitizer_type() == "tsan" else ""}
    for(int x = 0; x < gridX; x++) {{
      for(int y = 0; y < gridY; y++) {{
        for(int z = 0; z < gridZ; z++) {{
          {pointer_arg_decls}
          {kernel_name}({kernel_parameters}
                        gridX, gridY, gridZ, x, y, z);
        }}
      }}
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // [CPULauncher-specific]: We don't need the metadata below but just put them
  // here anyway to be consistent with others.
  // This will make updating the driver easier in the future.

  //  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  //  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
  //    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
  //    return NULL;
  //  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if _is_pointer_like(ty) else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"ptr_info{i}.dev_ptr" if _is_pointer_like(ty) else f"_arg{i}"for i, ty in signature.items())});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_shared_ref_cpu_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def compile_module(launcher_src, kernel_placeholder_name):
    py_version = sys.version_info
    if platform.system() == "Windows":
        py_include_dir = os.path.join(sys.base_prefix, 'include')
        py_lib_dir = os.path.join(sys.base_prefix, 'libs')
        py_lib = '{name}{major}{minor}.lib'.format(name="python", major=py_version.major, minor=py_version.minor)
    else:
        py_include_dir = os.path.join(sys.base_prefix, 'include', f'python{sys.version_info.major}.{sys.version_info.minor}')
        py_lib_dir = os.path.join(sys.base_prefix, 'lib')
        py_lib = '{name}{major}.{minor}'.format(name="python", major=py_version.major, minor=py_version.minor)
    cpu_backend_path = Path(__file__).resolve().parent
    include_dir = os.path.join(cpu_backend_path, "include")

    def launch(
        gridX, gridY, gridZ, stream, cu_function,
        kernel_metadata, launch_metadata,
        launch_enter_hook, launch_exit_hook, *args):
        # Unlike CUDA/HIP, we cannot easily pass function pointer across different pybind libraries.
        # Let's compile one kernel every time.
        # The cu_function parameter actually contains our kernel obj.
        # See CPUUtils.load_binary method.
        kernel_obj = cu_function
        kernel_name = kernel_metadata[6] # see pack_metadata in compiler.py
        src = launcher_src.replace(kernel_placeholder_name, kernel_name)

        key = hashlib.sha256(src.encode("utf-8") + kernel_obj).hexdigest()
        cache = get_cache_manager(key)
        name = "__triton_shared_ref_cpu_kernel_launcher"

        if platform.system() == "Windows":
          filename = f"{name}.pyd"
        else:
          filename = f"{name}.so"
        cache_path = cache.get_file(filename)

        if cache_path is None:
          with tempfile.TemporaryDirectory() as tmpdir:
              sanitizer_type = _get_sanitizer_type()

              if platform.system() == "Windows":
                  if sanitizer_type != "":
                      raise Exception("Sanitizers are not supported on Windows with triton-shared.")

                  obj_path = os.path.join(tmpdir, "kernel.obj")
                  launcher_src_path = os.path.join(tmpdir, "main.cxx")
                  so_path = os.path.join(tmpdir, "kernel.pyd")
                  Path(obj_path).write_bytes(kernel_obj)
                  Path(launcher_src_path).write_text(src)
                  # Compile it together.
                  subprocess.check_call([
                    "cl", "/LD", "/std:c++17", launcher_src_path, obj_path,
                    f"-I{py_include_dir}", f"-I{include_dir}", "/link", f"/LIBPATH:{py_lib_dir}",
                    "/link", f"{py_lib}", f"/OUT:{so_path}"
                  ])
              else:
                  obj_path = os.path.join(tmpdir, "kernel.o")
                  launcher_src_path = os.path.join(tmpdir, "main.cxx")
                  so_path = os.path.join(tmpdir, "kernel.so")
                  Path(obj_path).write_bytes(kernel_obj)
                  Path(launcher_src_path).write_text(src)

                  # Compile it together.
                  if sanitizer_type != "":
                      clang_path = _get_llvm_bin_path("clang++")

                      subprocess_args = [
                          clang_path, "-std=c++17", launcher_src_path, obj_path,
                          f"-I{py_include_dir}", f"-I{include_dir}", f"-L{py_lib_dir}",
                          "-shared", f"-l{py_lib}", "-fPIC", "-o", so_path
                      ]

                      if not _sanitizer_available(sanitizer_type):
                          raise Exception(f"Use LD_PRELOAD=\"path/to/libclang_rt.{sanitizer_type}.so\" TRITON_SHARED_SANITIZER_TYPE={sanitizer_type} python ...")

                      if sanitizer_type == "asan":
                          subprocess_args.extend(["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"])
                      elif sanitizer_type == "tsan":
                          # ensure that openmp is available
                          libomp_path = next(Path(Path(_get_llvm_bin_path("")).parent).rglob("libomp.so"), None)

                          if not libomp_path:
                              raise Exception("libomp.so does not exist.")

                          libomp_path = str(libomp_path.parent)

                          subprocess_args.extend(["-g", "-fsanitize=thread", "-fopenmp", f"-Wl,-rpath,{libomp_path}"])
                      
                      subprocess.check_call(subprocess_args)
                  else:
                      subprocess.check_call([
                        "g++", "-std=c++17", launcher_src_path, obj_path,
                        f"-I{py_include_dir}", f"-I{include_dir}", f"-L{py_lib_dir}",
                        "-shared", f"-l{py_lib}", "-fPIC", "-o", so_path
                      ])

              with open(so_path, "rb") as f:
                cache_path = cache.put(f.read(), filename, binary=True)

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel_args = args[0] if len(args) == 1 else args
        return mod.launch(gridX, gridY, gridZ,
                          kernel_metadata, launch_metadata,
                          launch_enter_hook, launch_exit_hook,
                          *kernel_args)

    return launch


class CPULauncher(object):

    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

        constants = src.constants if hasattr(src, "constants") else dict()
        def cst_key(i):
            return src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        expanded_signature, expanded_constants = _expand_signature_dict(signature, constants)
        launcher_src = _generate_launcher(expanded_constants, expanded_signature, kernel_placeholder_name)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        self.launch = wrap_handle_tensordesc_impl(compile_module(launcher_src, kernel_placeholder_name), signature,
                                              tensordesc_meta, make_tensordesc_arg)

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        self.launch(gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                    launch_exit_hook, args)



class CPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the cpu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
          "max_shared_mem": 2 ** 20,
          "multiprocessor_count": None,
          "sm_clock_rate": None,
          "mem_clock_rate": None,
          "mem_bus_width": None
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # obj of the kernel so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (
          None,       # module
          kernel_obj, # function
          None,       # n_regs
          None,        # n_spills
          sys.maxsize, # n_max_threads
        )


class CPUDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        self.binary_ext = "obj"

    # CPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(CPUDriver())
    @staticmethod
    def is_active():
        return False

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # CPU doesn't have a device to return. Return something.
        return "cpu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
    
    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)
  
