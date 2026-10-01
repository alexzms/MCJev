# Third-party notices

MCJev is licensed under the Apache License 2.0 (see `LICENSE`). It includes code adapted from the projects below,
which keep their own licenses.

## NanoJev

`serving/server.py` (`validate_request`, the `POST /api/evaluate` request/response contract) is adapted from
NanoJev, https://github.com/TianyuCodings/NanoJev (`scripts/predict_toy_decisions.py`).

```
MIT License

Copyright (c) 2026 OpenJev contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## vLLM

`serving/fastpath.py` runs on vLLM (https://github.com/vllm-project/vllm, Apache License 2.0) and, at runtime,
re-compiles a patched copy of vLLM's `triton_unified_attention.unified_attention` with a different launch
configuration. vLLM itself is a dependency and is not redistributed here.

## stb

`harness/body/voxcam/stb_image.h` (v2.30) and `harness/body/voxcam/stb_image_write.h` (v1.16) are Sean Barrett's
stb libraries (http://nothings.org/stb), included unmodified. They are dual-licensed, public domain (Unlicense) or
MIT; the license text is at the end of each file.

## Dependencies

The harness body runs on [mineflayer](https://github.com/PrismarineJS/mineflayer) and
[prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) (MIT, PrismarineJS; its textures are used by
the voxel camera), installed from npm by `harness/body/package.json`. The model weights,
[google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it), are downloaded from
Hugging Face under their own license.
