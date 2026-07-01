# Contribution Guide

Thankyou for helping to improve Incline. This project is a modern Rust desktop app for mine design, so good contributions need to priortise: 

- **Correctness** - Unless you have experience in mining, avoid significant changes to how to the tools function.
- **Performance** - Preformance is immensly important. Mine topologies are incredibly large and dense. We recommend making a test topology in blender by subdividing a plane many times and then shaping it into a mountainous surface. export this as an .obj
- **Simplicity** - Many mininig engineers value software that works for them. It's important to make tools and workflows easy and convient to use. For example, the auto-bench tool prefills with with the industry standard batter angle and bench height.

If your ever unsure on mining terminology, or industry design standards, please consult `docs/mining/` or email `leotimmins1974@gmail.com`.

## Project Layout

- `src/app/`: application state, commands, file operations, and event handling.
- `src/model/`: editable document model, PIDB storage, file format import/export, geometry, spatial queries, and triangulations.
- `src/rendering/`: camera, picking, snapping, GPU scene cache, and shaders.
- `src/startup/`: startup, os specific code, setting icons, etc.
- `src/ui/`: egui user interface, dialogs, toolbars, explorer, and widgets.
- `res`: contains resources that get included in the exe.
- `tutorial/`: small test files and user-facing tutorial material.
- `docs/`" documentation on code, tools, and file-formats.

## Development Workflow

- Run `cargo fmt`, `cargo check`, and `cargo clippy` before submitting. Resolve all warnings.
- Manually test the features you have changed to ensure nothing has broken or regressed.
- For file format work, add or update documentation under `docs/file_formats/`.
- For user-facing workflow changes, update tutorial or usage docs when the old instructions would become misleading.
- Avoid making large structural changes unless there is a clear definable benafit.
- Code must be readable, documented with comments, and simple. Do not overcomplicate the code for miniscule improvements in preformance.

## Target
This software must be cross-platform compatabile with MacOS, Linux, and Windows. The average mining engineer will be running this setup:

- Device:   Laptop
- OS:       Windows 11
- CPU:      Intel i7-13850HX, 20 Core, 2.1 GHz.
- GPU:      Discrete NVIDIA RTX 4090, 16 GB VRAM.
- Mem:      128 GB, DDR4.

## Code Style

Prefer existing patterns in the surrounding module. Keep changes scoped, and avoid broad refactors unless they are needed, and you know what you're doing.

Use clear names for geometry and mining concepts. A little explicitness is better than a clever abstraction when the code is describing coordinates, layers, triangulations, or design objects.

When adding comments, explain intent or non-obvious behavior. Avoid comments that simply repeat the line below them. If you're unsure about anything, document your assumptions / guesses and we can correct it.

## File Formats

PIDB files should remain stable and easy to inspect. If `format_version` changes, update `docs/file_formats/pidb-format.md`.

Triangulations are separate from PIDBs. Changes to `.00t`, `.dgd.isis`, OBJ, STL, or PLY handling should include sample-file testing where possible and should not silently discard data that Incline already preserves.

## Pull Requests

Good pull requests include:

- a short description of the problem and the fix
- document assumptions made if you're unsure of something
- screenshots or notes for visible UI changes
- any known limitations or follow-up work

Small PRs are easiest to review. If a change touches the model, rendering, and UI at once, explain the path through the code so reviewers can follow it quickly.

## AI use

AI use is permitted, but please review and test changes to ensure code functions corrrectly.

## Licence

We are utilising the [PolyForm Perimeter License 1.0.1](LICENSE.md). 

By contributing to Incline, you agree that your contributions will be licensed under the same licence as the project, unless explicitly stated otherwise.