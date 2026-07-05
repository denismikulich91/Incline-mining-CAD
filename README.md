# Incline
Incline is an open-pit mine design project to bring the ideals of free source-available software to the mining industry.

Cross-platform, lightweight, and built by mining engineers for mining engineers. Incline is free and open for everyone - aspiring students, professionals, and hobbists.

![Screenshot of the app](./tutorial/tutorial_screenshot.png)

## Status

Incline is in early development. Expect missing or incomplete features as we continue to polish for a v1.0 release.

We're working hard to bring stability and features over the coming months. However, please excerise caution during the early access - back up production data and verify exports before operational use.

## Features

- Layer-based editing for points, lines, polylines, polygons, text, roads, colours, visibility, and fill styles.
- DXF import/export for exchanging design linework with other CAD and mine planning tools.
- Triangulation loading, viewing, saving, and export.
- Block model viewing.
- Support for Vulcan `.00t`, OBJ, STL, and PLY triangulation files.
- Support for importing Vulcan `.dgd.isis` data as PIDB design geometry.
- GPU accelerated rendering with `wgpu`.
- Snapping to points, lines, and triangulation surfaces.
- Mine-design tools including offsetting, auto-bench style offsets, chamfering, line relimiting, fusing lines into polygons, polygon explosion, bezier shaping, and road creation.
- Cross-platform desktop target: Windows, Linux, and macOS.

## Getting Started

Install Rust & Cargo via rustup if you have not already done so.

Clone and run
```sh
cargo run --release
```

It is recomended you complete the [tutorial](tutorial/tutorial.md). 
It covers the basics of using Incline for mine design.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING), then look at [docs/good-first-commits.md](docs/good-first-commits.md) for approachable areas.

Before submitting changes, run:

```sh
cargo fmt
cargo check
cargo clippy
```

## License

Incline is licensed under the PolyForm Perimeter License 1.0.1. See [LICENSE](LICENSE.md) for the full terms.

Commercial licenses and permissions outside those terms are available from the copyright holders, please contact `leotimmins1974@gmail.com`.

## Maintainers

Two Western Australian mining engineer brothers - Lucas and Leo Timmins:
- **Leo Timmins** github: *leotimmins1974* email: *leotimmins1974@gmail.com*
- **Lucas Timmins** github: *trimental* email: *timmins.s.lucas@gmail.com*
