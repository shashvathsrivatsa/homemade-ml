use crate::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};


// ——— Draw Dot ———————————————————————————————————————————————————————————————————————————————————————————————————————

// ── Types mirroring your Value ─────────────────────────────────────────────

#[derive(Clone)]
struct GNode {
    id: usize,
    label: String,
    data: f64,
    grad: f64,
    op: &'static str,
}

struct GEdge {
    from: usize, // parent id
    to: usize,   // parent id
}

// ── Trace: walk the DAG collecting nodes + edges ───────────────────────────

fn trace(root: &crate::Value) -> (Vec<GNode>, Vec<GEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    fn build(
        v: &crate::Value,
        nodes: &mut Vec<GNode>,
        edges: &mut Vec<GEdge>,
        visited: &mut HashSet<usize>,
    ) {
        if !visited.insert(v.id) { return; }
        nodes.push(GNode { id: v.id, label: v.label.clone(), data: v.data, grad: *v.grad.borrow(), op: v.op });
        for parent in &v.parents {
            edges.push(GEdge { from: parent.id, to: v.id });
            build(parent, nodes, edges, visited);
        }
    }

    build(root, &mut nodes, &mut edges, &mut visited);
    (nodes, edges)
}

// ── Node width from content ────────────────────────────────────────────────

fn node_width(node: &GNode) -> i32 {
    let char_w  = 7_i32;
    let padding = 14_i32; // left + right per section
    let data_w  = format!("data={:.3}", node.data).len() as i32 * char_w + padding;
    let grad_w  = format!("grad={:.3}", node.grad).len() as i32 * char_w + padding;
    let label_w = if node.label.is_empty() { 0 } else {
        (node.label.len() as i32 * char_w + padding).max(20)
    };
    label_w + data_w + grad_w
}

// ── Layout: topological rank (x) + vertical slot (y) ─────────────────────
//
// We assign rank = longest path from any leaf to this node.
// Leaves get rank 0 (left), root gets the highest rank (right).
// Within each rank column, nodes are stacked vertically.
//
// Each value node with a non-empty op gets a paired "op node" rendered as
// an ellipse to its left, between the inputs and the value box.

fn layout(
    nodes: &[GNode],
    edges: &[GEdge],
    node_widths: &HashMap<usize, i32>,
) -> (HashMap<usize, (i32, i32)>, HashMap<usize, (i32, i32)>) {
    let mut downstream: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut inputs: HashMap<usize, Vec<usize>> = HashMap::new();
    for e in edges {
        downstream.entry(e.from).or_default().push(e.to);
        inputs.entry(e.to).or_default().push(e.from);
    }

    // Rank = longest path from leaf
    let mut rank: HashMap<usize, i32> = nodes.iter().map(|n| (n.id, 0)).collect();
    let mut in_deg: HashMap<usize, usize> = nodes.iter()
        .map(|n| (n.id, inputs.get(&n.id).map_or(0, |v| v.len())))
        .collect();
    let mut queue: VecDeque<usize> = nodes.iter()
        .filter(|n| in_deg[&n.id] == 0).map(|n| n.id).collect();
    while let Some(id) = queue.pop_front() {
        let r = rank[&id];
        for &ds in downstream.get(&id).unwrap_or(&vec![]) {
            let new_r = r + 1;
            if new_r > rank[&ds] { rank.insert(ds, new_r); }
            let deg = in_deg.get_mut(&ds).unwrap();
            *deg -= 1;
            if *deg == 0 { queue.push_back(ds); }
        }
    }

    // Pull leaves rightward to sit adjacent to their consumer
    for n in nodes {
        if inputs.get(&n.id).map_or(true, |v| v.is_empty()) {
            if let Some(ds_ids) = downstream.get(&n.id) {
                let min_r = ds_ids.iter().map(|id| rank[id]).min().unwrap_or(1);
                rank.insert(n.id, min_r - 1);
            }
        }
    }

    // Group by rank
    let mut by_rank: HashMap<i32, Vec<usize>> = HashMap::new();
    for n in nodes { by_rank.entry(rank[&n.id]).or_default().push(n.id); }

    // Max node width per column
    let mut col_max_w: HashMap<i32, i32> = HashMap::new();
    for n in nodes {
        let r = rank[&n.id];
        let w = node_widths[&n.id];
        let e = col_max_w.entry(r).or_insert(0);
        if w > *e { *e = w; }
    }

    // Sorted rank list → cumulative column x-centers
    let mut sorted_ranks: Vec<i32> = col_max_w.keys().cloned().collect();
    sorted_ranks.sort();

    let col_gap = 110_i32; // gap between column edges (op ellipse sits here)
    let row_h   =  50_i32;

    let mut col_x: HashMap<i32, i32> = HashMap::new();
    let mut cursor = 0_i32;
    for (i, &r) in sorted_ranks.iter().enumerate() {
        let half_w = col_max_w[&r] / 2;
        if i == 0 {
            cursor = half_w;
        } else {
            let prev_r = sorted_ranks[i - 1];
            cursor += col_max_w[&prev_r] / 2 + col_gap + half_w;
        }
        col_x.insert(r, cursor);
    }

    // Assign positions
    let mut val_pos: HashMap<usize, (i32, i32)> = HashMap::new();
    let mut op_pos:  HashMap<usize, (i32, i32)> = HashMap::new();

    for (&r, ids) in &by_rank {
        let column_left = col_x[&r] - col_max_w[&r] / 2;
        let n = ids.len() as i32;
        for (slot, &id) in ids.iter().enumerate() {
            let node = nodes.iter().find(|n| n.id == id).unwrap();
            let cx = column_left + node_widths[&id] / 2;
            let y = (slot as i32 - n / 2) * row_h + row_h / 2;
            val_pos.insert(id, (cx, y));

            if !node.op.is_empty() {
                // op ellipse x = midpoint between this column's left edge and prev column's right edge
                let left_edge = column_left;
                let prev_r = sorted_ranks.iter().rev().find(|&&pr| pr < r).copied();
                let op_x = if let Some(pr) = prev_r {
                    let prev_right = col_x[&pr] + col_max_w[&pr] / 2;
                    (prev_right + left_edge) / 2
                } else {
                    left_edge - col_gap / 2
                };
                op_pos.insert(id, (op_x, y));
            }
        }
    }

    // Spread overlapping nodes in a column downward. Do not re-center the
    // column afterward: derived nodes have already been placed at the vertical
    // midpoint of their inputs, and translating the whole column would break
    // that relationship.
    let resolve = |ids: &[usize], val_pos: &mut HashMap<usize, (i32, i32)>, op_pos: &mut HashMap<usize, (i32, i32)>| {
        if ids.len() < 2 { return; }
        let mut sorted: Vec<usize> = ids.to_vec();
        sorted.sort_by_key(|&id| val_pos[&id].1);

        for i in 1..sorted.len() {
            let prev_y = val_pos[&sorted[i - 1]].1;
            let curr_y = val_pos[&sorted[i]].1;
            if curr_y - prev_y < row_h {
                let ny = prev_y + row_h;
                let (cx, _) = val_pos[&sorted[i]];
                val_pos.insert(sorted[i], (cx, ny));
                if let Some(&(ox, _)) = op_pos.get(&sorted[i]) {
                    op_pos.insert(sorted[i], (ox, ny));
                }
            }
        }

    };

    // Phase 1: fix the leftmost column.
    let min_rank = sorted_ranks[0];
    if let Some(ids) = by_rank.get(&min_rank) {
        let ids = ids.clone();
        resolve(&ids, &mut val_pos, &mut op_pos);
    }

    // Phase 2: for each subsequent column left→right, center each node at
    // avg(y of its parents), then resolve overlaps in that column.
    for &r in sorted_ranks.iter().filter(|&&r| r > min_rank) {
        let ids = match by_rank.get(&r) {
            Some(v) => v.clone(),
            None => continue,
        };

        for &id in &ids {
            let in_ids = inputs.get(&id).cloned().unwrap_or_default();
            if in_ids.is_empty() { continue; }
            let avg_y = in_ids.iter().map(|&iid| val_pos[&iid].1).sum::<i32>()
                / in_ids.len() as i32;
            let (cx, _) = val_pos[&id];
            val_pos.insert(id, (cx, avg_y));
            if let Some(&(ox, _)) = op_pos.get(&id) {
                op_pos.insert(id, (ox, avg_y));
            }
        }

        resolve(&ids, &mut val_pos, &mut op_pos);
    }

    (val_pos, op_pos)
}

// ── draw_dot ───────────────────────────────────────────────────────────────

pub fn draw_dot(root: &crate::Value) {
    let path = "visualization.svg";

    let (nodes, edges) = trace(root);
    let node_widths: HashMap<usize, i32> = nodes.iter().map(|n| (n.id, node_width(n))).collect();
    let (val_pos, op_pos) = layout(&nodes, &edges, &node_widths);

    // Canvas bounds: pad around all positions
    let pad    = 120_i32;
    let node_h =  26_i32;
    let op_rx  = 22_i32; // circle radius

    let max_nw  = node_widths.values().cloned().max().unwrap_or(200);
    let all_x: Vec<i32> = val_pos.values().map(|&(x,_)| x).collect();
    let all_y: Vec<i32> = val_pos.values().map(|&(_,y)| y).collect();
    let min_x = all_x.iter().cloned().min().unwrap_or(0) - pad - max_nw;
    let max_x = all_x.iter().cloned().max().unwrap_or(0) + pad + max_nw;
    let min_y = all_y.iter().cloned().min().unwrap_or(0) - pad - node_h;
    let max_y = all_y.iter().cloned().max().unwrap_or(0) + pad + node_h;

    let w = (max_x - min_x) as u32;
    let h = (max_y - min_y) as u32;

    // Shift everything so min is at (pad, pad)
    let sx = |x: i32| x - min_x;
    let sy = |y: i32| y - min_y;

    let root_area = SVGBackend::new(path, (w as u32, h as u32)).into_drawing_area();
    root_area.fill(&BLACK).unwrap();

    let node_map: HashMap<usize, &GNode> = nodes.iter().map(|n| (n.id, n)).collect();

    // ── helper: draw an arrow from (x1,y1) → (x2,y2) ────────────────────
    let arrow = |x1: i32, y1: i32, x2: i32, y2: i32| {
        let dx = (x2 - x1) as f64;
        let dy = (y2 - y1) as f64;
        let len = (dx*dx + dy*dy).sqrt().max(1.0);
        let (ux, uy) = (dx/len, dy/len);

        let head_len  = 10.0_f64;
        let head_half =  4.0_f64;
        let base_x = x2 as f64 - head_len * ux;
        let base_y = y2 as f64 - head_len * uy;

        root_area.draw(&PathElement::new(
            vec![(x1, y1), (base_x as i32, base_y as i32)],
            ShapeStyle::from(&WHITE).stroke_width(1),
        )).unwrap();

        let lx = (base_x + head_half * uy) as i32;
        let ly = (base_y - head_half * ux) as i32;
        let rx = (base_x - head_half * uy) as i32;
        let ry = (base_y + head_half * ux) as i32;
        root_area.draw(&Polygon::new(
            vec![(x2, y2), (lx, ly), (rx, ry)],
            ShapeStyle::from(&WHITE).filled(),
        )).unwrap();
    };

    // ── draw edges ────────────────────────────────────────────────────────
    for edge in &edges {
        let (fx, fy) = val_pos[&edge.from];
        let (tx, ty) = val_pos[&edge.to];
        let to_node  = node_map[&edge.to];
        let from_hw  = node_widths[&edge.from] / 2;

        let x1 = sx(fx + from_hw);
        let y1 = sy(fy);

        if !to_node.op.is_empty() {
            let (ox, oy) = op_pos[&edge.to];
            arrow(x1, y1, sx(ox - op_rx), sy(oy));
        } else {
            let to_hw = node_widths[&edge.to] / 2;
            arrow(x1, y1, sx(tx - to_hw), sy(ty));
        }
    }

    // op ellipse → value box
    for (id, &(ox, oy)) in &op_pos {
        let (vx, vy) = val_pos[id];
        let to_hw = node_widths[id] / 2;
        arrow(sx(ox + op_rx), sy(oy), sx(vx - to_hw), sy(vy));
    }

    // ── draw value boxes ──────────────────────────────────────────────────
    for node in &nodes {
        let (cx, cy) = val_pos[&node.id];
        let nw = node_widths[&node.id];
        let x0 = sx(cx - nw / 2);
        let y0 = sy(cy - node_h / 2);
        let x1 = sx(cx + nw / 2);
        let y1 = sy(cy + node_h / 2);

        root_area.draw(&Rectangle::new(
            [(x0, y0), (x1, y1)],
            ShapeStyle::from(&BLACK).filled(),
        )).unwrap();
        root_area.draw(&Rectangle::new(
            [(x0, y0), (x1, y1)],
            ShapeStyle::from(&WHITE).stroke_width(1),
        )).unwrap();

        // approximate glyph dimensions for "sans-serif" 13pt
        let char_w = 7_i32;

        // Draw text centered within an individual cell.
        let center_text = |text: &str, left: i32, right: i32| {
            let style = ("sans-serif", 13)
                .into_font()
                .color(&WHITE)
                .pos(Pos::new(HPos::Center, VPos::Center));
            root_area.draw(&Text::new(
                text.to_string(),
                ((left + right) / 2, sy(cy)),
                style,
            )).unwrap();
        };

        let data_str = format!("data={:.3}", node.data);
        let grad_str = format!("grad={:.3}", node.grad);
        let grad_w = grad_str.len() as i32 * char_w + 14;

        // divider between data and grad sections
        let div2 = x1 - grad_w;
        root_area.draw(&PathElement::new(
            vec![(div2, y0), (div2, y1)],
            ShapeStyle::from(&WHITE).stroke_width(1),
        )).unwrap();
        center_text(&grad_str, div2, x1);

        if node.label.is_empty() {
            center_text(&data_str, x0, div2);
        } else {
            let label_w = (node.label.len() as i32 * char_w + 14).max(20);
            let div1 = x0 + label_w;
            root_area.draw(&PathElement::new(
                vec![(div1, y0), (div1, y1)],
                ShapeStyle::from(&WHITE).stroke_width(1),
            )).unwrap();
            center_text(&node.label, x0, div1);
            center_text(&data_str, div1, div2);
        }
    }

    // ── draw op ellipses ──────────────────────────────────────────────────
    for (id, &(ox, oy)) in &op_pos {
        let node = node_map[id];
        let cx = sx(ox);
        let cy = sy(oy);

        // filled black circle + white outline — SVGBackend emits a real <circle>
        root_area.draw(&Circle::new((cx, cy), op_rx, ShapeStyle::from(&BLACK).filled())).unwrap();
        root_area.draw(&Circle::new((cx, cy), op_rx, ShapeStyle::from(&WHITE).stroke_width(1))).unwrap();

        let label = node.op;
        let font_size = match label {
            "*" => 22_u32,
            "tanh" => 12_u32,
            _ => 16_u32,
        };
        let style = ("sans-serif", font_size)
            .into_font()
            .color(&WHITE)
            .pos(Pos::new(HPos::Center, VPos::Center));
        // `*` sits high within the font's line box, so nudge it down to make
        // the visible glyph (rather than just its metrics) optically centered.
        let label_y = if label == "*" { cy + 4 } else { cy };
        root_area.draw(&Text::new(
            label,
            (cx, label_y),
            style,
        )).unwrap();
    }

    root_area.present().unwrap();
    drop(root_area);

    // Plotters emits a fixed-size SVG. When it is opened directly in a
    // browser, expand the SVG viewport so the unused browser area is black as
    // well. The viewBox keeps the graph scaled proportionally.
    let svg = std::fs::read_to_string(path).unwrap();
    let svg = svg.replacen(
        "<svg ",
        "<svg style=\"width:100vw;height:100vh;background:#000;display:block\" ",
        1,
    );
    std::fs::write(path, svg).unwrap();
    println!("Saved → {}", path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_nodes_stay_vertically_centered_between_their_parents() {
        let nodes = vec![
            GNode { id: 9, label: "o".into(), data: 0.0, grad: 0.0, op: "tanh" },
            GNode { id: 8, label: "n".into(), data: 0.0, grad: 0.0, op: "+" },
            GNode { id: 5, label: "sum".into(), data: 0.0, grad: 0.0, op: "+" },
            GNode { id: 3, label: "left".into(), data: 0.0, grad: 0.0, op: "*" },
            GNode { id: 1, label: "x1".into(), data: 0.0, grad: 0.0, op: "" },
            GNode { id: 2, label: "w1".into(), data: 0.0, grad: 0.0, op: "" },
            GNode { id: 4, label: "right".into(), data: 0.0, grad: 0.0, op: "*" },
            GNode { id: 6, label: "x2".into(), data: 0.0, grad: 0.0, op: "" },
            GNode { id: 7, label: "w2".into(), data: 0.0, grad: 0.0, op: "" },
            GNode { id: 10, label: "b".into(), data: 0.0, grad: 0.0, op: "" },
        ];
        let edges = vec![
            GEdge { from: 1, to: 3 }, GEdge { from: 2, to: 3 },
            GEdge { from: 6, to: 4 }, GEdge { from: 7, to: 4 },
            GEdge { from: 3, to: 5 }, GEdge { from: 4, to: 5 },
            GEdge { from: 5, to: 8 }, GEdge { from: 10, to: 8 },
            GEdge { from: 8, to: 9 },
        ];
        let widths = nodes.iter().map(|node| (node.id, node_width(node))).collect();

        let (positions, operation_positions) = layout(&nodes, &edges, &widths);

        for child in [3, 4, 5, 8, 9] {
            let parents: Vec<_> = edges.iter().filter(|edge| edge.to == child).collect();
            let expected_y = parents.iter().map(|edge| positions[&edge.from].1).sum::<i32>()
                / parents.len() as i32;
            assert_eq!(positions[&child].1, expected_y, "node {child} is not centered");
            assert_eq!(operation_positions[&child].1, expected_y);
        }

        assert_eq!(
            positions[&5].0 - widths[&5] / 2,
            positions[&10].0 - widths[&10] / 2,
            "nodes in the same column should share a left edge",
        );
    }
}
