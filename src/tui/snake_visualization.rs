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
const FRUIT_REWARD: f32 = 2.0;
const COLLISION_REWARD: f32 = -1.0;
const APPROACH_REWARD_SCALE: f32 = 0.1;

pub struct SnakeGraph<const SHOW_CHART: bool> {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    state: Vec<f32>,
    segments: Vec<(f64, f64)>,
    fruit: (f64, f64),
    fruits_eaten: usize,
    episode_len: usize,
    current_reward: f32,
    points: Vec<(f64, f64)>,
    current: (f64, f64),
    frame_interval: Duration,
    last_render: Option<Instant>,
    held_keys: Vec<char>,
    last_key_activity: Option<Instant>,
    keyboard_enhancement_enabled: bool,
    cleaned_up: bool,
}

pub type SnakeTrainGraph = SnakeGraph<true>;
pub type SnakeTestGraph = SnakeGraph<false>;

// Keep existing callers source-compatible; the old name retains its original
// chart + game behavior.
pub type SnakeVisualization = SnakeTrainGraph;

impl<const SHOW_CHART: bool> SnakeGraph<SHOW_CHART> {
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
            current_reward: 0.0,
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
        self.current_reward = if done {
            COLLISION_REWARD
        } else if fruits_eaten > self.fruits_eaten {
            FRUIT_REWARD
        } else if segments.len() >= 2 {
            let previous_head = segments[1];
            let current_head = segments[0];
            let previous_distance = (previous_head.0 - fruit.0).hypot(previous_head.1 - fruit.1);
            let current_distance = (current_head.0 - fruit.0).hypot(current_head.1 - fruit.1);
            (previous_distance - current_distance) * APPROACH_REWARD_SCALE
        } else {
            0.0
        };
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
            let game_area = if SHOW_CHART {
                centered_game_area(main_areas[1])
            } else {
                centered_game_area(vertical_areas[0])
            };
            // A Braille canvas has 2 horizontal by 4 vertical drawable dots per
            // terminal cell. These dimensions let us measure the stroke in dots
            // instead of world units, keeping its width independent of direction.
            let dot_width = WORLD_SIZE / (f64::from(game_area.width.max(1)) * 2.0);
            let dot_height = WORLD_SIZE / (f64::from(game_area.height.max(1)) * 4.0);

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

                    // Stem first: apple (drawn after) repaints shared cells red,
                    // leaving only the above-apple portion of the stem green.
                    context.draw(&CanvasLine::new(
                        self.fruit.0,
                        self.fruit.1 + FRUIT_RADIUS,
                        self.fruit.0 + dot_width,
                        self.fruit.1 + FRUIT_RADIUS + dot_height * 4.0,
                        Color::Green,
                    ));

                    // Filled apple: concentric rings from center to edge
                    for i in 1..=6u8 {
                        context.draw(&Circle {
                            x: self.fruit.0,
                            y: self.fruit.1,
                            radius: FRUIT_RADIUS * f64::from(i) / 6.0,
                            color: Color::Red,
                        });
                    }

                    // Draw the two fringe edges perpendicular to each segment.
                    // Unlike axis-aligned copies of the whole path, this gives
                    // horizontal, vertical, and diagonal sections equal width.
                    for segment in self.segments.windows(2) {
                        let dx_dots = (segment[1].0 - segment[0].0) / dot_width;
                        let dy_dots = (segment[1].1 - segment[0].1) / dot_height;
                        let length_dots = dx_dots.hypot(dy_dots);
                        if length_dots == 0.0 {
                            continue;
                        }
                        let normal_x = -dy_dots / length_dots * dot_width;
                        let normal_y = dx_dots / length_dots * dot_height;

                        for side in [-1.0, 1.0] {
                            context.draw(&CanvasLine::new(
                                segment[0].0 + normal_x * side,
                                segment[0].1 + normal_y * side,
                                segment[1].0 + normal_x * side,
                                segment[1].1 + normal_y * side,
                                Color::Green,
                            ));
                        }
                    }

                    // Round caps also close the wedges where adjacent segments
                    // have different normals, without making turns extra thick.
                    for &(x, y) in &self.segments {
                        for (offset_x, offset_y) in [
                            (-dot_width, 0.0),
                            (dot_width, 0.0),
                            (0.0, -dot_height),
                            (0.0, dot_height),
                        ] {
                            context.draw(&CanvasLine::new(
                                x + offset_x,
                                y + offset_y,
                                x + offset_x,
                                y + offset_y,
                                Color::Green,
                            ));
                        }
                    }

                    for segment in self.segments.windows(2) {
                        context.draw(&CanvasLine::new(
                            segment[0].0,
                            segment[0].1,
                            segment[1].0,
                            segment[1].1,
                            Color::Green,
                        ));
                    }

                    if self.segments.len() >= 2 {
                        let head = self.segments[0];
                        let neck = self.segments[1];
                        let dx_dots = (head.0 - neck.0) / dot_width;
                        let dy_dots = (head.1 - neck.1) / dot_height;
                        let length_dots = dx_dots.hypot(dy_dots);
                        if length_dots > 0.0 {
                            let norm_dx = dx_dots / length_dots;
                            let norm_dy = dy_dots / length_dots;
                            let fwd_x = norm_dx * dot_width;
                            let fwd_y = norm_dy * dot_height;
                            let perp_x = -norm_dy * dot_width;
                            let perp_y = norm_dx * dot_height;
                            for side in [-0.7_f64, 0.7] {
                                let eye_x = head.0 + fwd_x * 0.5 + perp_x * side;
                                let eye_y = head.1 + fwd_y * 0.5 + perp_y * side;
                                context.draw(&CanvasLine::new(eye_x, eye_y, eye_x, eye_y, Color::White));
                            }
                        }
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
                "fruits: {}    reward: {:+.6}    episode steps: {}    snake length: {}\nstate: {:.3?}",
                self.fruits_eaten,
                self.current_reward,
                self.episode_len,
                self.segments.len(),
                self.state,
            ))
            .alignment(Alignment::Center);

            frame.render_widget(canvas, game_area);
            if SHOW_CHART {
                frame.render_widget(chart, main_areas[0]);
            }
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

impl<const SHOW_CHART: bool> StateVisualization for SnakeGraph<SHOW_CHART> {
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

impl<const SHOW_CHART: bool> Drop for SnakeGraph<SHOW_CHART> {
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
