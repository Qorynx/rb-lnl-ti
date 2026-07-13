"""Upgrade package exports.

Keep the model import lazy so data/config utilities can be inspected without
requiring the legacy timm dependency until a model is actually constructed.
"""

__all__ = ["RB_LNL_Ti", "rb_lnl_ti"]


def __getattr__(name):
    if name in __all__:
        from .rb_lnl_ti import RB_LNL_Ti, rb_lnl_ti

        return {"RB_LNL_Ti": RB_LNL_Ti, "rb_lnl_ti": rb_lnl_ti}[name]
    raise AttributeError(name)
