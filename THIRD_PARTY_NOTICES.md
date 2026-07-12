# Third-party software notices

Baby Monitor for Home Assistant is licensed under the MIT License in
[`LICENSE`](LICENSE). That license applies to this project's own source code;
it does not replace the licenses of the third-party components listed here.

This notice is informational and is not legal advice. The exact package
versions in a published image are recorded in its SPDX SBOM release assets and
SBOM attestations.

## Runtime application dependencies

The container redistributes Python and Python packages. Their license files are
preserved in the Python installation and in each installed distribution's
`.dist-info` metadata. Direct runtime dependencies include:

| Component | License |
| --- | --- |
| Python | PSF-2.0 |
| cryptography | Apache-2.0 OR BSD-3-Clause |
| FastAPI | MIT |
| HTTPX | BSD-3-Clause |
| Pydantic | MIT |
| python-multipart | Apache-2.0 |
| Uvicorn | BSD-3-Clause |

Transitive dependencies and their exact versions are listed in `uv.lock` and
in each release SBOM.

## Lit frontend libraries

The compiled frontend contains Lit, lit-html, lit-element, and
`@lit/reactive-element`, distributed under this license:

> BSD 3-Clause License
>
> Copyright (c) 2017 Google LLC. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
> this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright notice,
> this list of conditions and the following disclaimer in the documentation
> and/or other materials provided with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its contributors
> may be used to endorse or promote products derived from this software without
> specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

## Debian and FFmpeg in the container image

The final image is based on the official Debian Bookworm-based Python image and
installs Debian's `ca-certificates` and `ffmpeg` packages plus their runtime
dependencies. Those packages remain under their individual upstream and Debian
licenses; they are not relicensed under MIT.

FFmpeg is primarily licensed under LGPL-2.1-or-later and contains optional
GPL-2.0-or-later components. Debian's build may enable GPL components, so users
and redistributors should treat the installed binary according to the exact
license and build information shipped in the image. This project invokes the
`ffmpeg` executable as a separate process and does not copy FFmpeg source into,
or link its libraries with, the project's MIT-licensed Python code.

Every published image contains:

- `/usr/share/baby-monitor/debian-packages.tsv`, with exact binary package,
  binary version, source package, and source version fields;
- `/usr/share/baby-monitor/ffmpeg-build.txt`, with FFmpeg's build configuration;
- Debian copyright files under `/usr/share/doc/*/copyright`; and
- common license texts under `/usr/share/common-licenses`.

The complete corresponding source for an installed Debian package can be
located from the source package and source version recorded in
`debian-packages.tsv`:

- Debian Sources: <https://sources.debian.org/>
- Debian Snapshot, including superseded package versions:
  <https://snapshot.debian.org/package/>
- FFmpeg upstream source and license guidance: <https://ffmpeg.org/legal.html>

In a matching Debian Bookworm environment with `deb-src` entries enabled, the
source can also be retrieved with:

```text
apt-get source SOURCE_PACKAGE=SOURCE_VERSION
```

Use the release image digest, rather than a mutable tag such as `latest`, when
matching an image to its package inventory and corresponding source.
