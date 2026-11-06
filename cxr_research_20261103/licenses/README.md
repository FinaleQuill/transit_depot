# Pretrained Model License and Card Snapshots

These files are readable snapshots for compliance review. They are not a model
weight manifest and do not identify final inference members. Each HF model card
was archived at the same fixed revision used by the pretrained weight source.

## ConvNeXt Base

- Model: `timm/convnext_base.fb_in22k_ft_in1k_384`
- Revision: `2b2698d86523262ca8065d0957c7f80d963bb43f`
- Weight URL: <https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k_384/resolve/2b2698d86523262ca8065d0957c7f80d963bb43f/model.safetensors>
- Model-card URL: <https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k_384/raw/2b2698d86523262ca8065d0957c7f80d963bb43f/README.md>
- Apache-2.0 text URL: <https://www.apache.org/licenses/LICENSE-2.0.txt>
- Apache-2.0 local file SHA-256: `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`
- Model-card snapshot SHA-256: `402776a2dfa17b9dd416ab9a9fe4aa14496a551a0f0f93d1dfb70eca9f41e13a`
- Files: `convnext_base/LICENSE-Apache-2.0.txt`, `convnext_base/HF_MODEL_CARD.md`

## ConvNeXt Tiny

- Model: `timm/convnext_tiny.fb_in22k_ft_in1k_384`
- Revision: `32c741ba106f9e062174ef829354a16719c3dcac`
- Weight URL: <https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k_384/resolve/32c741ba106f9e062174ef829354a16719c3dcac/model.safetensors>
- Model-card URL: <https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k_384/raw/32c741ba106f9e062174ef829354a16719c3dcac/README.md>
- Apache-2.0 text URL: <https://www.apache.org/licenses/LICENSE-2.0.txt>
- Apache-2.0 local file SHA-256: `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`
- Model-card snapshot SHA-256: `4c72c672d524f7c38c5803d009fa6aabab9a83ce85844f5b1f150dd183ba3e6f`
- Files: `convnext_tiny/LICENSE-Apache-2.0.txt`, `convnext_tiny/HF_MODEL_CARD.md`

## Upstream ConvNeXt Code

- Repository: <https://github.com/facebookresearch/ConvNeXt>
- License URL: <https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/LICENSE>
- Local file: `convnext_upstream/LICENSE-MIT.txt`
- SHA-256: `da6d3703ed11cbe42bd212c725957c98da23cbff1998c05fa4b3d976d1a58e93`

The archived package contained no `NOTICE` file. The Hugging Face API request for
the remote file listing was rate-limited with HTTP 429 during this review, so the
remote repository's complete file list and absence of a NOTICE are not yet
independently verified. No NOTICE has been omitted from the archived files.

The copied snapshots were checked against their source archive hashes on
2026-11-02. Run `sha256sum licenses/convnext_base/* licenses/convnext_tiny/*
licenses/convnext_upstream/*` to recheck the local files.
