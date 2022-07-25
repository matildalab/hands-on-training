# 1. Getting Started
## Downloading the Poplar SDK
You can download the latest Poplar SDK at [Graphcore software download portal](https://downloads.graphcore.ai). (You need to request for an account to a Graphcore Field Engineer.)

## Activating the Poplar SDK
### i) Unpack the tar tarball file
First of all, unpack the tarball file downloaded from the download portal, typically named `poplar_sdk-<os-name>-<version-string>-<commit-hash>.tar.gz`.

```bash
tar -xvzf /path/to/poplar_sdk-<os-name>-<version-string>-<commit-hash>.tar.gz
```

### ii) Activate the Poplar SDK

After unpacking, you will be able to find a directory named `poplar_sdk-<os-name>-<version-string>-<commit-hash>`, where you can find two files named `enable.sh`.

```bash
find /path/to/poplar_sdk-<os-name>-<version-string>-<commit-hash> -name enable.sh
```

For example:

```
$ find /path/to/poplar_sdk-ubuntu_18_04-2.5.1+1001-64add8f33d -name enable.sh
/path/to/directory/poplar_sdk-ubuntu_18_04-2.5.1+1001-64add8f33d/poplar-ubuntu_18_04-2.5.0+4748-e94d646535/enable.sh
/path/to/poplar_sdk-ubuntu_18_04-2.5.1+1001-64add8f33d/popart-ubuntu_18_04-2.5.1+4748-e94d646535/enable.sh
```

**All you need to do is to `source` these two `enable.sh` files.**

For example,

```bash
source /path/to/poplar_sdk-ubuntu_18_04-2.5.1+1001-64add8f33d/poplar-ubuntu_18_04-2.5.0+4748-e94d646535/enable.sh
source /path/to/poplar_sdk-ubuntu_18_04-2.5.1+1001-64add8f33d/popart-ubuntu_18_04-2.5.1+4748-e94d646535/enable.sh
```

The "enable.sh" scripts will add required paths for using the Poplar SDK to the environment variables such as `PATH`, `LD_LIBRARY_PATH` and `PYTHONPATH`.

**If you add these two lines in your `~/.bashrc` or `~/.profile`, then you don't need to do this every time you login.**


<details><summary>**Alternative Trick (unofficial)**</summary><p>
Copy and paste the following bash functions in your `~/.bashrc` or `~/.profile`, and then logout and login again. (or just `source ~/.bashrc` for `source ~/.profile` depending on where you pasted the contents.)

<details><summary>***Contents to copy and paste***</summary><p>

```bash
function gc-activate () {
  source `find $1 -name "poplar-ubuntu*"`/enable.sh
  source `find $1 -name "popart-ubuntu*"`/enable.sh
}

function gc-deactivate() {
if [ -z ${POPLAR_SDK_ENABLED+x} ]
then
  echo 'ERROR: Poplar SDK is not enabled.'
else
  local POPLAR_SDK_ROOT=$(readlink -m ${POPLAR_SDK_ENABLED}/..)
  local IFS=':'

  local NEW_CMAKE_PREFIX_PATH
  local DIR
  for DIR in ${CMAKE_PREFIX_PATH} ; do
    if [[ "$DIR" != *"$POPLAR_SDK_ROOT"* ]] ; then
      NEW_CMAKE_PREFIX_PATH=${NEW_CMAKE_PREFIX_PATH:+$NEW_CMAKE_PREFIX_PATH:}$DIR
    fi
  done
  export CMAKE_PREFIX_PATH="$NEW_CMAKE_PREFIX_PATH"

  local NEW_CPATH
  local DIR
  for DIR in ${CPATH} ; do
    if [[ "$DIR" != *"$POPLAR_SDK_ROOT"* ]] ; then
      NEW_CPATH=${NEW_CPATH:+$NEW_CPATH:}$DIR
    fi
  done
  export CPATH="$NEW_CPATH"

  local NEW_LIBRARY_PATH
  local DIR
  for DIR in ${LIBRARY_PATH} ; do
    if [[ "$DIR" != *"$POPLAR_SDK_ROOT"* ]] ; then
      NEW_LIBRARY_PATH=${NEW_LIBRARY_PATH:+$NEW_LIBRARY_PATH:}$DIR
    fi
  done
  export LIBRARY_PATH="$NEW_LIBRARY_PATH"

  local NEW_LD_LIBRARY_PATH
  local DIR
  for DIR in ${LD_LIBRARY_PATH} ; do
    if [[ "$DIR" != *"$POPLAR_SDK_ROOT"* ]] ; then
      NEW_LD_LIBRARY_PATH=${NEW_LD_LIBRARY_PATH:+$NEW_LD_LIBRARY_PATH:}$DIR
    fi
  done
  export LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH"

  local NEW_PYTHONPATH
  local DIR
  for DIR in ${PYTHONPATH} ; do
    if [[ "$DIR" != *"$POPLAR_SDK_ROOT"* ]] ; then
      NEW_PYTHONPATH=${NEW_PYTHONPATH:+$NEW_PYTHONPATH:}$DIR
    fi
  done
  export PYTHONPATH="$NEW_PYTHONPATH"

  local NEW_OPAL_PREFIX
  local DIR
  for DIR in ${OPAL_PREFIX} ; do
    if [[ "$DIR" != *"$POPLAR_SDK_ROOT"* ]] ; then
      NEW_OPAL_PREFIX=${NEW_OPAL_PREFIX:+$NEW_OPAL_PREFIX:}$DIR
    fi
  done
  export OPAL_PREFIX="$NEW_OPAL_PREFIX"

  local NEW_PATH
  local DIR
  for DIR in ${PATH} ; do
    if [[ "$DIR" != *"$POPLAR_SDK_ROOT"* ]] ; then
      NEW_PATH=${NEW_PATH:+$NEW_PATH:}$DIR
    fi
  done
  export PATH="$NEW_PATH"

  unset POPLAR_SDK_ENABLED POPLAR_SDK_ROOT POPLAR_ROOT
fi
}

export -f gc-activate gc-deactivate
```

</p></details>

Then you can just activate a Poplar SDK by providing the path to the Poplar SDK directory, for example,

```bash
gc-activate /path/to/the/directory/poplar_sdk-<os-name>-<version-string>-<commit-hash>
```

Moreover, you can deactivate the Poplar SDK by running

```bash
gc-deactivate
```

</p></details>

### iii) Validating the Poplar SDK activation
Run `popc --version` at anywhere.
If the Poplar SDK is activated successfully, you will get the version string for the Poplar and Clang, for example,

```
POPLAR version 2.5.0 (76e88974fc)
clang version 14.0.0 (8139901ee03db200ba229fb0009c44d0a8ff6b25)
```

otherwise, you will see an error:

```
popc: command not found
```
