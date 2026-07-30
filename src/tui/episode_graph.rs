use crate::*;

// ——— TUI ————————————————————————————————————————————————————————————————————————————————————————————————————————————

const MOVING_AVG_WINDOW: usize = 50;

pub struct EpisodeGraph {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    points: Vec<(f64, f64)>,
}

impl EpisodeGraph {
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

    pub fn draw(&mut self, length: usize) -> std::io::Result<bool> {
        let episode = self.points.len();
        self.points.push((episode as f64, length as f64));

        let x_min = 0.0;
        let x_max = (episode as f64).max(1.0);
        let (y_min, y_max) = self.points
            .iter()
            .map(|point| point.1)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
                (min.min(value), max.max(value))
            });

        let moving_avg: Vec<(f64, f64)> = self.points
            .windows(MOVING_AVG_WINDOW.min(self.points.len()))
            .enumerate()
            .map(|(i, window)| {
                let avg = window.iter().map(|p| p.1).sum::<f64>() / window.len() as f64;
                (i as f64 + (window.len() - 1) as f64, avg)
            })
            .collect();

        self.terminal.draw(|frame| {
            let areas = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(3), Constraint::Length(1)])
                .split(frame.area());

            let datasets = vec![
                Dataset::default()
                    .marker(symbols::Marker::Dot)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(Color::Yellow))
                    .data(&self.points),
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&moving_avg),
            ];

            let chart = Chart::new(datasets)
                .x_axis(Axis::default().bounds([x_min, x_max]).labels([
                    Line::from(format!("{x_min:.0}")),
                    Line::from(format!("{x_max:.0}")),
                ]))
                .y_axis(Axis::default().bounds([y_min, y_max]).labels([
                    Line::from(format!("{y_min:.0}")),
                    Line::from(format!("{y_max:.0}")),
                ]));

            let status = Paragraph::new(format!("Episode: {episode}  Length: {length}")).alignment(Alignment::Center);

            frame.render_widget(chart, areas[0]);
            frame.render_widget(status, areas[1]);
        })?;

        self.check_quit();
        Ok(false)
    }

    fn check_quit(&self) {
        while event::poll(Duration::ZERO).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read()
                && key.kind == KeyEventKind::Press
                && (key.code == KeyCode::Char('q')
                    || key.code == KeyCode::Char('c')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL))
            {
                let _ = disable_raw_mode();
                let _ = execute!(stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
                std::process::exit(0);
            }
        }
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        if self.points.is_empty() {
            return Ok(());
        }

        let x_min = 0.0;
        let x_max = (self.points.len() - 1) as f64;
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

impl Drop for EpisodeGraph {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}
