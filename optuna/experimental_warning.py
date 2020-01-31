import warnings


# Reference: https://github.com/cupy/cupy/blob/8d923719ced31fdff4a42702d395ec87a7e98045/cupy/util.pyx#L89  # NOQA
def experimental(api_name: str) -> None:
    """Declares that user is using an experimental feature.

    The developer of an API can mark it as *experimental* by calling
    this function. When users call experimental APIs, :class:`FutureWarning`
    is issued.
    The presentation of :class:`FutureWarning` is disabled by setting
    ``cupy.disable_experimental_warning`` to ``True``,
    which is ``False`` by default.

    The basic usage is to call it in the function or method we want to
    mark as experimental along with the API name.

    .. testcode::

        class D():
            def __init__(self):
                util.experimental('D.__init__')

        D()

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ...  FutureWarning: D.__init__ is experimental. \
The interface can change in the future. ...

    Currently, we do not have any sophisticated way to mark some usage of
    non-experimental function as experimental.
    But we can support such usage by explicitly branching it.

    Args:
        api_name: The name of an API marked as experimental.
    """

    warnings.warn(
        '{} is experimental. The interface can change in the future.'.format(api_name),
        FutureWarning)
