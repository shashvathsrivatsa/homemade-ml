use crate::*;

// ——— TUI ————————————————————————————————————————————————————————————————————————————————————————————————————————————

const MOVING_AVG_WINDOW: usize = 50;

fn non_degenerate_bounds(min: f64, max: f64) -> (f64, f64) {
    if min == max {
        let padding = min.abs().mul_add(0.05, 1.0);
        (min - padding, max + padding)
    } else {
        (min, max)
    }
}

pub struct FruitsGraph {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    points: Vec<(f64, f64)>,
    current: (f64, f64),
    frame_interval: Duration,
    last_render: Option<Instant>,
    cleaned_up: bool,
}

impl FruitsGraph {
    pub fn new(fps: u32) -> std::io::Result<Self> {
        if fps == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "fruits graph FPS must be greater than zero",
            ));
        }

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
            current: (0.0, 0.0),
            frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
            last_render: None,
            cleaned_up: false,
        })
    }

    pub fn update(
        &mut self,
        fruits_eaten: usize,
        done: bool,
    ) -> std::io::Result<Option<char>> {
        let episode = self.points.len();
        self.current = (episode as f64, fruits_eaten as f64);

        if done {
            self.points.push(self.current);
        }

        let render_due = self
            .last_render
            .is_none_or(|last_render| last_render.elapsed() >= self.frame_interval);
        if done || render_due {
            self.render(!done)?;
            self.last_render = Some(Instant::now());
        }

        self.check_key()
    }

    fn render(&mut self, show_current: bool) -> std::io::Result<()> {
        let x_min = 0.0;
        let x_max = self.current.0.max(1.0);
        let (mut y_min, mut y_max) = self
            .points
            .iter()
            .map(|p| p.1)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
                (min.min(v), max.max(v))
            });
        if show_current {
            y_min = y_min.min(self.current.1);
            y_max = y_max.max(self.current.1);
        }
        let (y_min, y_max) = if y_min == f64::INFINITY {
            (0.0, 1.0)
        } else {
            non_degenerate_bounds(y_min, y_max)
        };

        let moving_avg: Vec<(f64, f64)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let window = &self.points[(i + 1).saturating_sub(MOVING_AVG_WINDOW)..=i];
                let avg = window.iter().map(|p| p.1).sum::<f64>() / window.len() as f64;
                (point.0, avg)
            })
            .collect();

        self.terminal.draw(|frame| {
            let current: &[(f64, f64)] = if show_current {
                std::slice::from_ref(&self.current)
            } else {
                &[]
            };
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
                Dataset::default()
                    .marker(symbols::Marker::Dot)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(Color::White))
                    .data(current),
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

            frame.render_widget(chart, frame.area());
        })?;

        Ok(())
    }

    fn check_key(&mut self) -> std::io::Result<Option<char>> {
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                if let KeyCode::Char(character) = key.code {
                    let character = if character == 'c'
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        '\u{3}'
                    } else {
                        character
                    };
                    return Ok(Some(character));
                }
            }
        }
        Ok(None)
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        if self.points.is_empty() {
            return Ok(());
        }

        let x_min = 0.0;
        let x_max = ((self.points.len() - 1) as f64).max(x_min + 1.0);
        let (y_min, y_max) = self
            .points
            .iter()
            .map(|point| point.1)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
                (min.min(value), max.max(value))
            });
        let (y_min, y_max) = non_degenerate_bounds(y_min, y_max);

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

impl StateVisualization for FruitsGraph {
    fn update_state(
        &mut self,
        _segments: &VecDeque<(f32, f32)>,
        _fruit: (f32, f32),
        fruits_eaten: usize,
        _episode_len: usize,
        done: bool,
    ) -> std::io::Result<Option<char>> {
        self.update(fruits_eaten, done)
    }
}

impl Drop for FruitsGraph {
    fn drop(&mut self) {
        if self.cleaned_up {
            return;
        }
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}
