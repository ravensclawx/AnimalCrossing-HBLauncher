# Animal Crossing for HBLauncher

A Switch 2 HBLauncher port of [outusuke/ACGC-PC-Port](https://github.com/outusuke/ACGC-PC-Port),
based on the Animal Crossing GameCube decompilation. Maintained by **@ravensclawx**.

The native AArch64 game runs through HBLauncher's Linux personality. Its GX/OpenGL
translation layer uses the included `glwgpu` adapter to render through Dawn/Vulkan.
The app uses exclusive display ownership and requires your own **GAFE01 USA rev 0** disc.

## Requirements

- [HBLauncher](https://github.com/ravensclawx/HBLauncher), with ABI 3 and its Linux personality runtime.
- Linux, Python 3, Git, CMake, Ninja, and a patched `aarch64-linux-musl` toolchain.
- Your own supported game disc image. Disc images, extracted game data, saves,
  and game artwork are not included in this fork.

These are experimental Switch 2 ports. A successful build does not establish
console compatibility or gameplay performance. Hardware testing of these new
standalone packages is still required.

## Toolchain

Set the SDK and compiler locations (replace the example paths):

```sh
export HBL_ROOT=/path/to/HBLauncher
export HBXL_TOOLCHAIN=/path/to/patched-toolchain
```

An ordinary Linux cross-compiler is insufficient: musl must route syscalls
through HBLauncher's `__hbx_syscall`, and target libraries need BTI landing pads.
The known configuration uses GCC 15.1.0, musl 1.2.6, and binutils 2.44. To build
it with musl-cross-make:

```sh
git clone https://github.com/richfelker/musl-cross-make.git toolchain-src
git -C toolchain-src checkout 5e8042e63de29f8473fc3b0b831a587ba284e1a8
cp hblauncher/cmake/config.mak.example toolchain-src/config.mak
mkdir -p toolchain-src/patches/musl-1.2.6
cp "$HBL_ROOT/sdk/linux/musl-hbx-syscall-hooks.diff"   toolchain-src/patches/musl-1.2.6/9001-hbx-syscall-hooks.diff
make -C toolchain-src -j8
make -C toolchain-src install
export HBXL_TOOLCHAIN="$PWD/toolchain-src/toolchain"
```

Compiler sources and downloads remain local; they are not part of this fork.

## Build

Pinned versions are recorded in `hblauncher/dependencies.json`. The dependency
script checks out Dawn, SDL 3.4.10, and glslang, applies the Switch patches,
then builds Dawn, SDL3, glslang, and the static SDL2 compatibility layer:

```sh
python3 hblauncher/dependencies.py build --jobs 8
export DAWN_SOURCE="$PWD/hblauncher/deps/dawn"
export DAWN_BUILD="$PWD/hblauncher/deps/build-dawn"
export GLSLANG_PREFIX="$PWD/hblauncher/deps/glslang-install"
export SDL2_PREFIX="$PWD/hblauncher/deps/sdl-switch"
python3 hblauncher/build.py build --jobs 8
```

You may point those four variables to existing matching, patched dependency
builds instead. `build.py configure` only configures; `build.py pack --elf
/path/to/AnimalCrossing` packages a previously linked executable without rebuilding.
The default code/data addresses are `0x20000000` / `0x28000000`.

## Install and run

The resulting app directory is `hblauncher/dist/animalcrossing`. Copy that
whole directory into `$HBL_ROOT/apps/`, or run:

```sh
python3 hblauncher/build.py install
```

Installation refuses to overwrite an existing app directory. Only the manifest,
code/data images, and matching build report are installed; there is no bundled
commercial game artwork.

Create a private file-server root:

```sh
mkdir -p hblauncher/fsroot/rom
cp /path/to/your/AnimalCrossing.iso hblauncher/fsroot/rom/AnimalCrossing.iso
cp -R pc/shaders hblauncher/fsroot/shaders
printf 'HBXL_PRELOAD=/rom/AnimalCrossing.iso\nHBXL_ARGS=--verbose\n' > hblauncher/fsroot/hbxl-env
python3 hblauncher/build.py serve
```

The server uses `10.42.0.1:39502`, matching the app's manifest and linked
`HBXL_FS_PORT`. Keep it running, establish a verified HBLauncher session, and
select **Animal Crossing**. The server root also receives settings, caches,
and saves; keep them private and back up saves before replacing a local setup.
See `README.upstream.md` and `pc/DOCUMENTATION.md` for upstream game setup and controls.

## Source and credits

Upstream history and licensing are preserved. Read [LICENSE](LICENSE),
[the upstream README](README.upstream.md), and
[the integration notices](hblauncher/THIRD_PARTY.md). HBLauncher integration
additions use [MIT](hblauncher/LICENSE.integration). Original upstream CI is
disabled for this port branch; local build instructions above target HBLauncher.

## Source archive

`python3 hblauncher/package-source.py` writes a source-only archive under
`hblauncher/dist/`. Initialize submodules before packaging Twilight Princess.
The archive includes their source and the applied Aurora changes. It excludes
local dependencies, build output, discs, saves, and file-server data. Build the
app locally using the instructions above; generated app packages are ignored by Git.
