"""跨平台启动兼容入口：Linux为空操作，仅标准库，不构建模型或触碰数据。

2026-10-17：本机Python3.14.2导入torch时，platform.machine()卡在WMI查询；
25秒导入探针的栈见artifacts/startup_import_probe_20261017_0245.log。
这里使用platform.py已有的非WMI回退分支，不修改Windows服务或安装目录。
相关上游问题：https://github.com/python/cpython/issues/125315
"""
from __future__ import annotations

import sys


def prepare_runtime() -> bool:
    """在任何第三方导入前调用；主进程与Windows spawn子进程均适用。

    本地CPython的platform._wmi_query在_wmi不可用时抛OSError，调用方已有
    sys.getwindowsversion、系统命令及环境架构信息的回退逻辑。因此只禁用
    本进程的WMI查询路径，不伪造CPU架构/Windows版本，也不禁用系统WMI。

    _wmi是私有属性：只对本次已核对的Windows CPython3.14版本族生效，
    升级解释器后须重新核查；其他平台/版本不作修改。多次调用保持幂等。
    返回值表示本次是否切换了查询路径，不代表训练/多进程已通过实测。
    """
    # Linux直接返回，不导入platform，也不探测任何Windows/WMI私有接口。
    if sys.platform != "win32":
        return False
    import platform
    if (sys.implementation.name != "cpython"
            or sys.version_info[:2] != (3, 14) or not hasattr(platform, "_wmi")):
        return False
    if platform._wmi is None:
        return False
    platform._wmi = None
    return True


def prepare_windows_runtime() -> bool:
    """兼容旧B3模块的入口；Linux上与prepare_runtime一样为空操作。"""
    return prepare_runtime()
