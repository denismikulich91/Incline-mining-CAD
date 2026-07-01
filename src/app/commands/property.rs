use crate::{
    app::App,
    model::{Command, FillStyle, Object, ObjectColor, ObjectId},
    userspace_log,
};

macro_rules! batch_property {
    ($self:expr, $ids:expr, $update_fn:expr, $log:literal) => {{
        let Some(project) = $self.workspace.active_project_mut() else {
            return;
        };
        let doc = &mut project.pidb.document;
        let cmds: Vec<Command> = $ids
            .iter()
            .filter_map(|&id| {
                let before = doc.get_object(id)?.clone();
                let mut after = before.clone();
                $update_fn(&mut after);
                if before != after {
                    Some(Command::Replace { before, after })
                } else {
                    None
                }
            })
            .collect();
        if !cmds.is_empty() {
            $self.history.execute(doc, Command::Batch(cmds));
            project.dirty = true;
        }
        userspace_log!($log, $ids.len());
        $self.invalidate_geometry();
    }};
}

impl<'a> App<'a> {
    pub(crate) fn batch_set_object_color(&mut self, ids: Vec<ObjectId>, new_color: ObjectColor) {
        batch_property!(
            self,
            ids,
            |obj: &mut Object| {
                match obj {
                    Object::Point { color, .. }
                    | Object::Polyline { color, .. }
                    | Object::Text { color, .. }
                    | Object::Road { color, .. } => *color = new_color,
                }
            },
            "Batch-set color on {} object(s)"
        );
    }

    pub(crate) fn batch_set_polyline_closed(&mut self, ids: Vec<ObjectId>, closed: bool) {
        batch_property!(
            self,
            ids,
            |obj: &mut Object| {
                if let Object::Polyline { closed: c, .. } = obj {
                    *c = closed;
                }
            },
            "Batch-set closed on {} polyline(s)"
        );
    }

    pub(crate) fn batch_set_object_fill(&mut self, ids: Vec<ObjectId>, new_fill: FillStyle) {
        batch_property!(
            self,
            ids,
            |obj: &mut Object| {
                if let Object::Polyline { fill, .. } = obj {
                    *fill = new_fill;
                }
            },
            "Batch-set fill style on {} object(s)"
        );
    }

    pub(crate) fn batch_set_polyline_line_weight(&mut self, ids: Vec<ObjectId>, weight: f32) {
        batch_property!(
            self,
            ids,
            |obj: &mut Object| {
                if let Object::Polyline { line_weight, .. } = obj {
                    *line_weight = weight;
                }
            },
            "Batch-set line weight on {} polyline(s)"
        );
    }
}
