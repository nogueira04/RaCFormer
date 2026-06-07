# Third-party Patches

This directory records local dependency patches that should not be hidden as dirty vendored state.

- `mmcv_setup_sysroot_linker_workaround.patch`: local MMCV build workaround that filters an invalid sysroot linker argument during extension build on the cluster.

Apply manually inside the `mmcv` checkout if the cluster build environment still needs it.
