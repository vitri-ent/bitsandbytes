import errno
import platform
import site
import os
from pathlib import Path
from typing import Set, Union
from bitsandbytes.cextension import CUDASetup

from .env_vars import get_potentially_lib_path_containing_env_vars


IS_WINDOWS_PLATFORM: bool = (platform.system()=="Windows")
PATH_COLLECTION_SEPARATOR: str = ":" if not IS_WINDOWS_PLATFORM else ";"
CUDA_SHARED_LIB_NAME: str = "libcuda.so" if not IS_WINDOWS_PLATFORM else f"{os.environ['SystemRoot']}\\System32\\nvcuda.dll"
SHARED_LIB_EXTENSION: str = ".so" if not IS_WINDOWS_PLATFORM else ".dll"
CUDA_RUNTIME_LIB: str = "libcudart.so" if not IS_WINDOWS_PLATFORM else "cudart64_110.dll"
backup_path = os.path.join(os.environ.get("CUDA_PATH", os.getcwd()), "bin", CUDA_RUNTIME_LIB) if IS_WINDOWS_PLATFORM else '/usr/local/cuda/lib64'


def extract_candidate_paths(paths_list_candidate: str) -> Set[Path]:
    return {Path(ld_path) for ld_path in paths_list_candidate.split(PATH_COLLECTION_SEPARATOR) if ld_path}


def remove_non_existent_dirs(candidate_paths: Set[Path]) -> Set[Path]:
    existent_directories: Set[Path] = set()
    for path in candidate_paths:
        try:
            if path.exists():
                existent_directories.add(path)
        except OSError as exc:
            if exc.errno != errno.ENAMETOOLONG:
                raise exc

    non_existent_directories: Set[Path] = candidate_paths - existent_directories
    if non_existent_directories:
        CUDASetup.get_instance().add_log_entry("WARNING: The following directories listed in your path were found to "
            f"be non-existent: {non_existent_directories}", is_warning=True)

    return existent_directories


def get_cuda_runtime_lib_paths(candidate_paths: Set[Path]) -> Set[Path]:
    return {
        path / CUDA_RUNTIME_LIB
        for path in candidate_paths
        if (path / CUDA_RUNTIME_LIB).is_file()
    }


def resolve_paths_list(paths_list_candidate: str) -> Set[Path]:
    """
    Searches a given environmental var for the CUDA runtime library,
    i.e. `libcudart.so`.
    """
    return remove_non_existent_dirs(extract_candidate_paths(paths_list_candidate))


def find_cuda_lib_in(paths_list_candidate: str) -> Set[Path]:
    return get_cuda_runtime_lib_paths(
        resolve_paths_list(paths_list_candidate)
    )


def warn_in_case_of_duplicates(results_paths: Set[Path]) -> None:
    if len(results_paths) > 1:
        warning_msg = (
            f"Found duplicate {CUDA_RUNTIME_LIB} files: {results_paths}.. "
            "We'll flip a coin and try one of these, in order to fail forward.\n"
            "Either way, this might cause trouble in the future:\n"
            "If you get `CUDA error: invalid device function` errors, the above "
            "might be the cause and the solution is to make sure only one "
            f"{CUDA_RUNTIME_LIB} in the paths that we search based on your env.")
        CUDASetup.get_instance().add_log_entry(warning_msg, is_warning=True)


def determine_cuda_runtime_lib_path() -> Union[Path, None]:
    """
        Searches for a cuda installations, in the following order of priority:
            1. active conda env
            2. LD_LIBRARY_PATH
            3. any other env vars, while ignoring those that
                - are known to be unrelated (see `bnb.cuda_setup.env_vars.to_be_ignored`)
                - don't contain the path separator `/`

        If multiple libraries are found in part 3, we optimistically try one,
        while giving a warning message.
    """
    candidate_env_vars = get_potentially_lib_path_containing_env_vars()

    if "CONDA_PREFIX" in candidate_env_vars:
        conda_libs_path = Path(candidate_env_vars["CONDA_PREFIX"]) / "bin"

        conda_cuda_libs = find_cuda_lib_in(str(conda_libs_path))
        warn_in_case_of_duplicates(conda_cuda_libs)

        if conda_cuda_libs:
            return next(iter(conda_cuda_libs))
        
        conda_libs_path = Path(candidate_env_vars["CONDA_PREFIX"]) / "lib"

        conda_cuda_libs = find_cuda_lib_in(str(conda_libs_path))
        warn_in_case_of_duplicates(conda_cuda_libs)

        if conda_cuda_libs:
            return next(iter(conda_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CONDA_PREFIX"]} did not contain '
            f'{CUDA_RUNTIME_LIB} as expected! Searching further paths...', is_warning=True)

    for sitedir in site.getsitepackages():
        if "site-packages" in sitedir:
                site_packages_path = sitedir
                break
    if site_packages_path:
        torch_libs_path = os.path.join(site_packages_path, "torch", "lib")
        
        if os.path.isdir(torch_libs_path):
            torch_cuda_libs = find_cuda_lib_in(str(torch_libs_path))
            warn_in_case_of_duplicates(torch_cuda_libs)

            if torch_cuda_libs:
                return next(iter(torch_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{torch_cuda_libs} did not contain '
            f'{CUDA_RUNTIME_LIB} as expected! Searching further paths...', is_warning=True)
        
    if "CUDA_PATH" in candidate_env_vars:
        win_toolkit_libs_path = Path(candidate_env_vars["CUDA_PATH"]) / "bin"
    
        win_toolkit_cuda_libs = find_cuda_lib_in(str(win_toolkit_libs_path))
        warn_in_case_of_duplicates(win_toolkit_cuda_libs)

        if win_toolkit_cuda_libs:
            return next(iter(win_toolkit_cuda_libs))

        win_toolkit_libs_path = Path(candidate_env_vars["CUDA_PATH"]) / "lib"
    
        win_toolkit_cuda_libs = find_cuda_lib_in(str(win_toolkit_libs_path))
        warn_in_case_of_duplicates(win_toolkit_cuda_libs)

        if win_toolkit_cuda_libs:
            return next(iter(win_toolkit_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CONDA_PREFIX"]} did not contain '
            f'{CUDA_RUNTIME_LIB} as expected! Searching further paths...', is_warning=True)
        
    if "CUDA_HOME" in candidate_env_vars:
        lin_toolkit_libs_path = Path(candidate_env_vars["CUDA_HOME"]) / "bin"
    
        lin_toolkit_cuda_libs = find_cuda_lib_in(str(lin_toolkit_libs_path))
        warn_in_case_of_duplicates(lin_toolkit_cuda_libs)

        if lin_toolkit_cuda_libs:
            return next(iter(lin_toolkit_cuda_libs))
        
        lin_toolkit_libs_path = Path(candidate_env_vars["CUDA_HOME"]) / "lib"
    
        lin_toolkit_cuda_libs = find_cuda_lib_in(str(lin_toolkit_libs_path))
        warn_in_case_of_duplicates(lin_toolkit_cuda_libs)

        if lin_toolkit_cuda_libs:
            return next(iter(lin_toolkit_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CONDA_PREFIX"]} did not contain '
            f'{CUDA_RUNTIME_LIB} as expected! Searching further paths...', is_warning=True)

    if "LD_LIBRARY_PATH" in candidate_env_vars:
        lib_ld_cuda_libs = find_cuda_lib_in(candidate_env_vars["LD_LIBRARY_PATH"])

        if lib_ld_cuda_libs:
            cuda_runtime_libs.update(lib_ld_cuda_libs)
        warn_in_case_of_duplicates(lib_ld_cuda_libs)

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["LD_LIBRARY_PATH"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
        
    if "PATH" in candidate_env_vars:
        lib_path_cuda_libs = find_cuda_lib_in(candidate_env_vars["PATH"])
        warn_in_case_of_duplicates(lib_path_cuda_libs)

        if lib_path_cuda_libs:
            return next(iter(lib_path_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CONDA_PREFIX"]} did not contain '
            f'{CUDA_RUNTIME_LIB} as expected! Searching further paths...', is_warning=True)
        
    remaining_candidate_env_vars = {
        env_var: value for env_var, value in candidate_env_vars.items()
        if env_var not in {"CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PATH"}
    }

    cuda_runtime_libs = set()
    for env_var, value in remaining_candidate_env_vars.items():
        cuda_runtime_libs.update(find_cuda_lib_in(value))

    if len(cuda_runtime_libs) == 0:
        CUDASetup.get_instance().add_log_entry(f'CUDA_SETUP: WARNING! {CUDA_RUNTIME_LIB} not found in any environmental path. Searching {backup_path}...')
        cuda_runtime_libs.update(find_cuda_lib_in(backup_path))

    warn_in_case_of_duplicates(cuda_runtime_libs)

    return next(iter(cuda_runtime_libs)) if cuda_runtime_libs else None
