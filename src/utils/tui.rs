use crate::*;

// ——— TUI ————————————————————————————————————————————————————————————————————————————————————————————————————————————

const LOSS_HISTORY_SIZE: usize = 200;

pub struct LossGraph {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    points: Vec<(f64, f64)>,
}

impl LossGraph {
    pub fn new() -> std::io::Result<Self> {
        enable_raw_mode()?;
        let mut output = stdout();
        if let Err(error) = execute!(output, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error);
        }

        let terminal = match Terminal::new(CrosstermBackend::new(output)) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = disable_raw_mode();
                let _ = execute!(stdout(), LeaveAlternateScreen);
                return Err(error);
            }
        };

        Ok(Self {
            terminal,
            points: Vec::with_capacity(LOSS_HISTORY_SIZE),
        })
    }

    pub fn draw(&mut self, step: usize, loss: f32) -> std::io::Result<()> {
        if loss.is_finite() {
            self.points.push((step as f64, loss as f64));
            if self.points.len() > LOSS_HISTORY_SIZE {
                self.points.remove(0);
            }
        }

        let x_min = self.points.first().map_or(step as f64, |point| point.0);
        let x_max = (step as f64).max(x_min + 1.0);
        let (mut y_min, mut y_max) = if self.points.is_empty() {
            (0.0, 1.0)
        } else {
            self.points
                .iter()
                .map(|point| point.1)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
                    (min.min(value), max.max(value))
                })
        };

        if (y_max - y_min).abs() < f64::EPSILON {
            let padding = (y_max.abs() * 0.05).max(0.001);
            y_min -= padding;
            y_max += padding;
        } else {
            let padding = (y_max - y_min) * 0.05;
            y_min -= padding;
            y_max += padding;
        }

        self.terminal.draw(|frame| {
            let areas = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(3), Constraint::Length(1)])
                .split(frame.area());

            let datasets = vec![
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&self.points),
            ];

            let chart = Chart::new(datasets)
                .x_axis(
                    Axis::default()
                        .title("Step")
                        .bounds([x_min, x_max])
                        .labels([
                            Line::from(format!("{x_min:.0}")),
                            Line::from(format!("{x_max:.0}")),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Loss")
                        .bounds([y_min, y_max])
                        .labels([
                            Line::from(format!("{y_min:.4}")),
                            Line::from(format!("{y_max:.4}")),
                        ]),
                );

            let status = Paragraph::new(format!("Step: {step}  Loss: {loss:.6}"));

            frame.render_widget(chart, areas[0]);
            frame.render_widget(status, areas[1]);
        })?;

        Ok(())
    }

    pub fn should_quit(&self) -> std::io::Result<bool> {
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
                && key.code == KeyCode::Char('q')
            {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

impl Drop for LossGraph {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}
