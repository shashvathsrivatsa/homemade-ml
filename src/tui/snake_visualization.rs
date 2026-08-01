use crate::*;
use crossterm::event::{
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use ratatui::{
    layout::Rect,
    widgets::canvas::{Canvas, Circle, Line as CanvasLine, Rectangle},
};

const WORLD_SIZE: f64 = 100.0;
const WORLD_MIN: f64 = -WORLD_SIZE / 2.0;
const WORLD_MAX: f64 = WORLD_SIZE / 2.0;
const FRUIT_RADIUS: f64 = 2.5;
const MOVING_AVG_WINDOW: usize = 50;
const KEY_REPEAT_TIMEOUT: Duration = Duration::from_millis(50);

pub struct SnakeVisualization {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    state: Vec<f32>,
    segments: Vec<(f64, f64)>,
    fruit: (f64, f64),
    fruits_eaten: usize,
    episode_len: usize,
    points: Vec<(f64, f64)>,
    current: (f64, f64),
    frame_interval: Duration,
    last_render: Option<Instant>,
    held_keys: Vec<char>,
    last_key_activity: Option<Instant>,
    keyboard_enhancement_enabled: bool,
    cleaned_up: bool,
}

impl SnakeVisualization {
    pub fn new(fps: u32) -> std::io::Result<Self> {
        if fps == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "snake visualization FPS must be greater than zero",
            ));
        }

        enable_raw_mode()?;
        let mut output = stdout();
        let keyboard_enhancement_enabled = matches!(
            crossterm::terminal::supports_keyboard_enhancement(),
            Ok(true)
        );
        if let Err(error) = execute!(output, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error);
        }
        if keyboard_enhancement_enabled {
            if let Err(error) = execute!(
                output,
                PushKeyboardEnhancementFlags(
                    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                        | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
                        | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
                        | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                )
            ) {
                let _ = execute!(stdout(), LeaveAlternateScreen);
                let _ = disable_raw_mode();
                return Err(error);
            }
        }

        let terminal = match Terminal::new(CrosstermBackend::new(output)) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = disable_raw_mode();
                if keyboard_enhancement_enabled {
                    let _ = execute!(stdout(), PopKeyboardEnhancementFlags);
                }
                let _ = execute!(stdout(), LeaveAlternateScreen);
                return Err(error);
            }
        };

        Ok(Self {
            terminal,
            state: Vec::new(),
            segments: Vec::new(),
            fruit: (0.0, 0.0),
            fruits_eaten: 0,
            episode_len: 0,
            points: Vec::new(),
            current: (0.0, 0.0),
            frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
            last_render: None,
            held_keys: Vec::new(),
            last_key_activity: None,
            keyboard_enhancement_enabled,
            cleaned_up: false,
        })
    }

    pub fn update(
        &mut self,
        state: Vec<f32>,
        segments: &VecDeque<(f32, f32)>,
        fruit: (f32, f32),
        fruits_eaten: usize,
        episode_len: usize,
        done: bool,
    ) -> std::io::Result<Option<char>> {
        self.state = state;
        self.segments = segments
            .iter()
            .map(|&(x, y)| (f64::from(x), f64::from(y)))
            .collect();
        self.fruit = (f64::from(fruit.0), f64::from(fruit.1));
        self.fruits_eaten = fruits_eaten;
        self.episode_len = episode_len;
        self.current = (self.points.len() as f64, fruits_eaten as f64);
        if done {
            self.points.push(self.current);
        }

        let render_due = self
            .last_render
            .is_none_or(|last_render| last_render.elapsed() >= self.frame_interval);
        if done || render_due {
            self.render()?;
            self.last_render = Some(Instant::now());
        }

        self.check_key()
    }

    pub fn poll_key(&mut self) -> std::io::Result<Option<char>> {
        self.check_key()
    }

    fn render(&mut self) -> std::io::Result<()> {
        let x_min = 0.0;
        let x_max = self.current.0.max(1.0);
        let y_min = 0.0;
        let y_max = self
            .points
            .iter()
            .map(|point| point.1)
            .fold(self.current.1.max(1.0), f64::max);
        let moving_avg: Vec<(f64, f64)> = self
            .points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let window = &self.points[(index + 1).saturating_sub(MOVING_AVG_WINDOW)..=index];
                let average = window.iter().map(|point| point.1).sum::<f64>() / window.len() as f64;
                (point.0, average)
            })
            .collect();

        self.terminal.draw(|frame| {
            let vertical_areas = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(5), Constraint::Length(2)])
                .split(frame.area());
            let main_areas = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(vertical_areas[0]);
            let game_area = centered_game_area(main_areas[1]);

            let canvas = Canvas::default()
                .marker(symbols::Marker::Braille)
                .x_bounds([WORLD_MIN, WORLD_MAX])
                .y_bounds([WORLD_MIN, WORLD_MAX])
                .paint(|context| {
                    context.draw(&Rectangle {
                        x: WORLD_MIN,
                        y: WORLD_MIN,
                        width: WORLD_SIZE,
                        height: WORLD_SIZE,
                        color: Color::DarkGray,
                    });

                    context.draw(&Circle {
                        x: self.fruit.0,
                        y: self.fruit.1,
                        radius: FRUIT_RADIUS,
                        color: Color::Red,
                    });

                    for segment in self.segments.windows(2) {
                        context.draw(&CanvasLine::new(
                            segment[0].0,
                            segment[0].1,
                            segment[1].0,
                            segment[1].1,
                            Color::Green,
                        ));
                    }
                });

            let current = [self.current];
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
                    .data(&current),
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

            let status = Paragraph::new(format!(
                "fruits: {}    episode steps: {}    snake length: {}\nstate: {:.3?}",
                self.fruits_eaten,
                self.episode_len,
                self.segments.len(),
                self.state,
            ))
            .alignment(Alignment::Center);

            frame.render_widget(canvas, game_area);
            frame.render_widget(chart, main_areas[0]);
            frame.render_widget(status, vertical_areas[1]);
        })?;

        Ok(())
    }

    fn check_key(&mut self) -> std::io::Result<Option<char>> {
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
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

                    match key.kind {
                        KeyEventKind::Press | KeyEventKind::Repeat => {
                            self.held_keys.retain(|held| *held != character);
                            self.held_keys.push(character);
                            self.last_key_activity = Some(Instant::now());
                        }
                        KeyEventKind::Release => {
                            self.held_keys.retain(|held| *held != character);
                        }
                    }
                }
            }
        }

        if !self.keyboard_enhancement_enabled
            && self
                .last_key_activity
                .is_some_and(|activity| activity.elapsed() >= KEY_REPEAT_TIMEOUT)
        {
            self.held_keys.clear();
        }

        Ok(self.held_keys.last().copied())
    }
}

fn centered_game_area(area: Rect) -> Rect {
    let width = area.width.min(area.height.saturating_mul(2)).max(1);
    let height = area.height.min(width / 2).max(1);

    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

impl StateVisualization for SnakeVisualization {
    fn update_state(
        &mut self,
        state: Vec<f32>,
        segments: &VecDeque<(f32, f32)>,
        fruit: (f32, f32),
        fruits_eaten: usize,
        episode_len: usize,
        done: bool,
    ) -> std::io::Result<Option<char>> {
        self.update(state, segments, fruit, fruits_eaten, episode_len, done)
    }
}

impl Drop for SnakeVisualization {
    fn drop(&mut self) {
        if self.cleaned_up {
            return;
        }
        let _ = disable_raw_mode();
        if self.keyboard_enhancement_enabled {
            let _ = execute!(self.terminal.backend_mut(), PopKeyboardEnhancementFlags);
        }
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}
