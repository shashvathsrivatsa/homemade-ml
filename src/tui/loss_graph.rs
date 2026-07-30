use crate::*;

// ——— TUI ————————————————————————————————————————————————————————————————————————————————————————————————————————————

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
            points: Vec::new(),
        })
    }

    pub fn draw(&mut self, step: usize, loss: f32) -> std::io::Result<()> {
        if loss.is_finite() {
            self.points.push((step as f64, loss as f64));
        }

        let x_min = self.points.first().map_or(step as f64, |point| point.0);
        let x_max = (step as f64).max(x_min + 1.0);
        let (y_min, y_max) = if self.points.is_empty() {
            (0.0, 1.0)
        } else {
            self.points
                .iter()
                .map(|point| point.1)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
                    (min.min(value), max.max(value))
                })
        };

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
                .x_axis(Axis::default().bounds([x_min, x_max]).labels([
                    Line::from(format!("{x_min:.0}")),
                    Line::from(format!("{x_max:.0}")),
                ]))
                .y_axis(Axis::default().bounds([y_min, y_max]).labels([
                    Line::from(format!("{y_min:.4}")),
                    Line::from(format!("{y_max:.4}")),
                ]));

            let status = Paragraph::new(format!("Loss: {loss:.6}")).alignment(Alignment::Center);

            frame.render_widget(chart, areas[0]);
            frame.render_widget(status, areas[1]);
        })?;

        Ok(())
    }

    pub fn should_quit(&self) -> std::io::Result<bool> {
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
                && (key.code == KeyCode::Char('q')
                    || key.code == KeyCode::Char('c')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL))
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        if self.points.is_empty() {
            return Ok(());
        }

        let x_min = self.points.first().unwrap().0;
        let x_max = self.points.last().unwrap().0.max(x_min + 1.0);
        let (y_min, y_max) = self
            .points
            .iter()
            .map(|point| point.1)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
                (min.min(value), max.max(value))
            });

        let root = BitMapBackend::new(path.as_ref(), (1200, 700)).into_drawing_area();
        root.fill(&RGBColor(10, 14, 20))?;

        let mut chart = ChartBuilder::on(&root)
            .margin(30)
            .x_label_area_size(50)
            .y_label_area_size(80)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart
            .configure_mesh()
            .y_labels(2)
            .axis_style(WHITE)
            .label_style(("sans-serif", 20).into_font().color(&WHITE))
            .draw()?;

        chart.draw_series(LineSeries::new(self.points.iter().copied(), &CYAN))?;
        root.present()?;

        Ok(())
    }
}

impl Drop for LossGraph {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}
