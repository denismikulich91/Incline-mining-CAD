//! Cached document text layout and rendering primitives.

use std::{
    collections::{HashMap, HashSet, hash_map},
    hash::{BuildHasher, Hash, Hasher},
};

use glam::Mat4;
use glyphon::{
    Attrs, AttrsList, BufferLine, Color, FamilyOwned, FontSystem, Style, SwashCache, TextArea,
    TextBounds, Viewport, Weight,
};

use crate::rendering::color::linear_to_srgb_byte;

#[derive(Debug, Clone, Copy, Hash)]
struct Font<'a> {
    family: glyphon::Family<'a>,
    weight: glyphon::Weight,
    style: glyphon::Style,
}

#[derive(Clone, Copy, Hash)]
pub(crate) struct SectionKey<'a> {
    content: &'a str,
    font: Font<'a>,
    color: glyphon::Color,
    index: usize,
}

#[derive(Clone)]
pub(crate) struct Key<'a> {
    lines: Vec<Vec<SectionKey<'a>>>,
    size: f32,
    line_height: f32,
    bounds: (f32, f32),
}

type KeyHash = u64;
type HashBuilder = twox_hash::xxhash64::RandomState;

#[derive(Default)]
pub(crate) struct TextCache {
    entries: HashMap<KeyHash, glyphon::Buffer>,
    recently_used: HashSet<KeyHash>,
    hasher: HashBuilder,
}

impl TextCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get(&self, key: &KeyHash) -> Option<&glyphon::Buffer> {
        self.entries.get(key)
    }

    fn allocate(
        &mut self,
        font_system: &mut glyphon::FontSystem,
        key: Key<'_>,
    ) -> (KeyHash, &mut glyphon::Buffer) {
        let hash = {
            let mut hasher = self.hasher.build_hasher();

            key.lines.hash(&mut hasher);
            key.size.to_bits().hash(&mut hasher);
            key.line_height.to_bits().hash(&mut hasher);
            key.bounds.0.to_bits().hash(&mut hasher);
            key.bounds.1.to_bits().hash(&mut hasher);

            hasher.finish()
        };

        let paragraph = match self.entries.entry(hash) {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => {
                let metrics = glyphon::Metrics::new(key.size, key.line_height);
                let mut buffer = glyphon::Buffer::new(font_system, metrics);

                buffer.set_size(Some(key.bounds.0), Some(key.bounds.1.max(key.line_height)));

                buffer.lines.clear();

                for line in key.lines {
                    let mut line_str = String::new();
                    let mut attrs_list = AttrsList::new(&Attrs::new());
                    for section in line {
                        let start = line_str.len();
                        line_str.push_str(section.content);
                        let end = line_str.len();
                        attrs_list.add_span(
                            start..end,
                            &Attrs::new()
                                .family(section.font.family)
                                .weight(section.font.weight)
                                .style(section.font.style)
                                .color(section.color)
                                .metadata(0),
                        )
                    }
                    let buffer_line = BufferLine::new(
                        line_str,
                        glyphon::cosmic_text::LineEnding::CrLf,
                        attrs_list,
                        glyphon::Shaping::Advanced,
                    );
                    buffer.lines.push(buffer_line);
                }

                buffer.shape_until_scroll(font_system, true);

                entry.insert(buffer)
            }
        };

        let _ = self.recently_used.insert(hash);

        (hash, paragraph)
    }

    pub(crate) fn trim(&mut self) {
        self.entries
            .retain(|key, _| self.recently_used.contains(key));

        self.recently_used.clear();
    }
}

#[derive(Clone)]
pub(crate) struct CachedTextArea {
    key: KeyHash,
    left: f32,
    top: f32,
    bounds: TextBounds,
    default_color: glyphon::Color,
    transform: Mat4,
    zoom: f32,
}

impl CachedTextArea {
    pub(crate) fn text_area<'a>(&self, cache: &'a TextCache) -> Option<TextArea<'a>> {
        let buffer = cache.get(&self.key)?;
        Some(TextArea {
            buffer,
            left: self.left,
            top: self.top,
            bounds: self.bounds,
            default_color: self.default_color,
            scale: 1.,
            transform: self.transform,
            zoom: self.zoom,
            custom_glyphs: &[],
        })
    }
}

pub(crate) struct TextSystem {
    pub(crate) font_system: FontSystem,
    pub(crate) text_renderer: glyphon::TextRenderer,
    pub(crate) text_atlas: glyphon::TextAtlas,
    pub(crate) text_cache: TextCache,
    pub(crate) swash_cache: SwashCache,
    pub(crate) viewport: Viewport,
}

#[derive(Clone)]
pub(crate) struct Text {
    pub(crate) text: String,
    pub(crate) color: Option<[f32; 4]>,
    pub(crate) is_bold: bool,
    pub(crate) is_italic: bool,
    pub(crate) font_family: FamilyOwned,
    pub(crate) default_color: [f32; 4],
}

impl Text {
    pub(crate) fn new(text: String, default_text_color: [f32; 4]) -> Self {
        Self {
            text,
            default_color: default_text_color,
            color: None,
            is_bold: false,
            is_italic: false,
            font_family: FamilyOwned::Monospace,
        }
    }

    fn color(&self) -> [f32; 4] {
        self.color.unwrap_or(self.default_color)
    }

    fn style(&self) -> Style {
        if self.is_italic {
            Style::Italic
        } else {
            Style::Normal
        }
    }

    fn weight(&self) -> Weight {
        if self.is_bold {
            Weight::BOLD
        } else {
            Weight::NORMAL
        }
    }

    pub(crate) fn section_keys(&self, index: usize) -> Vec<SectionKey<'_>> {
        let color = self.color();
        let color = Color::rgba(
            linear_to_srgb_byte(color[0]),
            linear_to_srgb_byte(color[1]),
            linear_to_srgb_byte(color[2]),
            (color[3].clamp(0.0, 1.0) * 255.) as u8,
        );
        let font = Font {
            family: self.font_family.as_family(),
            weight: self.weight(),
            style: self.style(),
        };
        self.text
            .lines()
            .map(|line| SectionKey {
                content: line,
                font,
                color,
                index,
            })
            .collect()
    }
}

#[derive(Clone)]
pub(crate) struct TextBox {
    pub(crate) font_size: f32,
    pub(crate) line_height_factor: f32,
    pub(crate) texts: Vec<Text>,
    pub(crate) hidpi_scale: f32,
}

impl Default for TextBox {
    fn default() -> Self {
        Self {
            font_size: 16.0,
            line_height_factor: 1.1,
            texts: Vec::new(),
            hidpi_scale: 1.0,
        }
    }
}

impl TextBox {
    pub(crate) fn new(texts: Vec<Text>, hidpi_scale: f32) -> TextBox {
        TextBox {
            texts,
            hidpi_scale,
            ..Default::default()
        }
    }

    pub(crate) fn line_height(&self, zoom: f32) -> f32 {
        self.font_size * self.line_height_factor * self.hidpi_scale * zoom
    }

    pub(crate) fn key(&self, bounds: (f32, f32)) -> Key<'_> {
        let mut lines = Vec::new();
        let mut sections = Vec::new();
        for (i, text) in self.texts.iter().enumerate() {
            let text_lines = text.section_keys(i);
            for (line_index, line_section) in text_lines.into_iter().enumerate() {
                if line_index > 0 {
                    lines.push(std::mem::take(&mut sections));
                }
                sections.push(line_section);
            }
            if text.text.ends_with('\n') {
                lines.push(std::mem::take(&mut sections));
            }
        }
        if !sections.is_empty() {
            lines.push(std::mem::take(&mut sections));
        }

        Key {
            lines,
            size: self.font_size * self.hidpi_scale,
            line_height: self.line_height(1.),
            bounds,
        }
    }

    pub(crate) fn text_areas(
        &self,
        text_system: &mut TextSystem,
        screen_position: (f32, f32),
        bounds: (f32, f32),
        zoom: f32,
        transform: Mat4,
    ) -> CachedTextArea {
        let cache = &mut text_system.text_cache;

        let key = {
            let (key, _paragraph) = cache.allocate(&mut text_system.font_system, self.key(bounds));
            key
        };

        CachedTextArea {
            key,
            left: screen_position.0,
            top: screen_position.1,
            bounds: TextBounds::default(),
            default_color: Color::rgb(255, 255, 255),
            transform,
            zoom,
        }
    }
}
